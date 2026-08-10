from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Mapping

from .actuators import weighted_fair_decoder
from .safety import RoutingSafetyConfig
from .schema import ControlCommand, ControlValidationError
from .state import AppliedControl, PreparedControl


ROUTING_ACTUATOR = "decode_routing_ratio"


class RuntimeControlManager:
    """State machine used inside the Proxy's centralized scheduler process."""

    def __init__(
        self,
        *,
        enabled: bool,
        safety: RoutingSafetyConfig,
        wall_clock: Callable[[], int] = time.time_ns,
        mono_clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.enabled = enabled
        self.safety = safety
        self._wall_clock = wall_clock
        self._mono_clock = mono_clock
        self.control_generation = 0
        self.control_mode = "baseline"
        self.control_status = "BASELINE"
        self.rho_a = 0.5
        self.current_command_id: str | None = None
        self.previous_safe_value: float | str = "baseline"
        self.prepared: PreparedControl | None = None
        self.command_records: dict[str, AppliedControl | PreparedControl | dict[str, Any]] = {}
        self.last_applied_mono_ns = 0
        self.assignment_counts: dict[str, int] = {}
        self.assignment_tokens: dict[str, float] = {}
        self.recent_assignments: deque[tuple[str, float]] = deque(maxlen=safety.recent_window_size)
        self.safety_fallback_count = 0

    def _current_value(self) -> float | str:
        return self.rho_a if self.control_mode == "controlled" else "baseline"

    def _response(
        self,
        *,
        accepted: bool,
        reason: str,
        command: ControlCommand | None = None,
        old_value: float | str | None = None,
        effective_value: float | str | None = None,
        validated_wall_ns: int | None = None,
        applied_wall_ns: int | None = None,
        effective_from: str | None = None,
    ) -> dict[str, Any]:
        return {
            "accepted": accepted,
            "rejected": not accepted,
            "old_value": self._current_value() if old_value is None else old_value,
            "requested_value": None if command is None else command.requested_value,
            "effective_value": self._current_value() if effective_value is None else effective_value,
            "validated_wall_ns": validated_wall_ns,
            "applied_wall_ns": applied_wall_ns,
            "effective_from": effective_from,
            "control_generation": self.control_generation if command is None else command.control_generation,
            "control_command_id": None if command is None else command.control_command_id,
            "actuator_name": None if command is None else command.actuator_name,
            "requested_wall_ns": None if command is None else command.requested_wall_ns,
            "reason": reason,
        }

    def _reject(self, command: ControlCommand | None, reason: str) -> dict[str, Any]:
        return self._response(accepted=False, reason=reason, command=command)

    def _expected_matches(self, expected: float | str) -> bool:
        current = self._current_value()
        if isinstance(current, str) or isinstance(expected, str):
            return str(current) == str(expected)
        return abs(float(current) - float(expected)) <= 1e-12

    def prepare(self, raw_command: Mapping[str, Any], *, decoders_healthy: bool) -> dict[str, Any]:
        try:
            command = ControlCommand.from_mapping(raw_command)
        except ControlValidationError as exc:
            return self._reject(None, f"schema_invalid:{exc}")
        if not self.enabled:
            return self._reject(command, "control_plane_disabled")
        if command.control_command_id in self.command_records:
            return self._reject(command, "duplicate_command_id")
        if command.actuator_name != ROUTING_ACTUATOR:
            return self._reject(command, "actuator_not_supported")
        if command.control_generation != self.control_generation + 1:
            return self._reject(command, "stale_or_nonmonotonic_generation")
        if not self._expected_matches(command.expected_current_value):
            return self._reject(command, "compare_and_swap_mismatch")
        if not decoders_healthy:
            return self._reject(command, "decode_health_guard_failed")
        try:
            requested = float(command.requested_value)
        except (TypeError, ValueError):
            return self._reject(command, "requested_value_not_numeric")
        if not self.safety.minimum_rho_a <= requested <= self.safety.maximum_rho_a:
            return self._reject(command, "requested_value_out_of_range")
        if abs(requested - self.rho_a) > self.safety.maximum_step + 1e-12:
            return self._reject(command, "maximum_step_exceeded")
        now_mono = self._mono_clock()
        if self.last_applied_mono_ns and now_mono - self.last_applied_mono_ns < self.safety.minimum_dwell_ns:
            return self._reject(command, "minimum_dwell_time_not_met")
        now_wall = self._wall_clock()
        prepared = PreparedControl(
            command=command,
            validated_wall_ns=now_wall,
            validated_mono_ns=now_mono,
            expires_mono_ns=now_mono + self.safety.prepare_ttl_ns,
            old_mode=self.control_mode,
            old_value=self._current_value(),
        )
        self.prepared = prepared
        self.command_records[command.control_command_id] = prepared
        return self._response(
            accepted=True,
            reason="prepared",
            command=command,
            old_value=prepared.old_value,
            effective_value=prepared.old_value,
            validated_wall_ns=now_wall,
            effective_from="pending_commit",
        )

    def commit(self, raw_command: Mapping[str, Any]) -> dict[str, Any]:
        try:
            command = ControlCommand.from_mapping(raw_command)
        except ControlValidationError as exc:
            return self._reject(None, f"schema_invalid:{exc}")
        record = self.command_records.get(command.control_command_id)
        if isinstance(record, AppliedControl):
            return {**record.response, "idempotent_replay": True}
        if not isinstance(record, PreparedControl) or self.prepared is not record:
            return self._reject(command, "command_not_prepared")
        if command != record.command:
            return self._reject(command, "prepared_command_mismatch")
        now_mono = self._mono_clock()
        if now_mono > record.expires_mono_ns:
            self.prepared = None
            self.command_records[command.control_command_id] = {"phase": "expired"}
            return self._reject(command, "prepared_command_expired")
        old_value = record.old_value
        self.previous_safe_value = old_value
        self.rho_a = float(command.requested_value)
        self.control_mode = "controlled"
        self.control_status = "CONTROLLED"
        self.control_generation = command.control_generation
        self.current_command_id = command.control_command_id
        self.last_applied_mono_ns = now_mono
        self.prepared = None
        self.assignment_counts.clear()
        self.assignment_tokens.clear()
        self.recent_assignments.clear()
        now_wall = self._wall_clock()
        response = self._response(
            accepted=True,
            reason="applied",
            command=command,
            old_value=old_value,
            effective_value=self.rho_a,
            validated_wall_ns=record.validated_wall_ns,
            applied_wall_ns=now_wall,
            effective_from="next_decode_route",
        )
        self.command_records[command.control_command_id] = AppliedControl(command, response)
        return response

    def rollback(self, raw_command: Mapping[str, Any]) -> dict[str, Any]:
        try:
            command = ControlCommand.from_mapping(raw_command)
        except ControlValidationError as exc:
            return self._reject(None, f"schema_invalid:{exc}")
        if command.control_command_id in self.command_records:
            record = self.command_records[command.control_command_id]
            if isinstance(record, AppliedControl):
                return {**record.response, "idempotent_replay": True}
            return self._reject(command, "duplicate_command_id")
        if command.control_generation != self.control_generation + 1:
            return self._reject(command, "stale_or_nonmonotonic_generation")
        if command.actuator_name != ROUTING_ACTUATOR:
            return self._reject(command, "actuator_not_supported")
        if not self._expected_matches(command.expected_current_value):
            return self._reject(command, "compare_and_swap_mismatch")
        target = command.requested_value
        if target == "previous_safe_value":
            target = self.previous_safe_value
        old_value = self._current_value()
        if target == "baseline":
            self.control_mode = "baseline"
            self.control_status = "BASELINE"
            self.rho_a = 0.5
            effective: float | str = "baseline"
        else:
            try:
                numeric = float(target)
            except (TypeError, ValueError):
                return self._reject(command, "rollback_target_invalid")
            if not self.safety.minimum_rho_a <= numeric <= self.safety.maximum_rho_a:
                return self._reject(command, "rollback_target_out_of_range")
            self.control_mode = "controlled"
            self.control_status = "CONTROLLED"
            self.rho_a = numeric
            effective = numeric
        self.previous_safe_value = old_value
        self.control_generation = command.control_generation
        self.current_command_id = command.control_command_id
        self.last_applied_mono_ns = self._mono_clock()
        self.prepared = None
        self.assignment_counts.clear()
        self.assignment_tokens.clear()
        self.recent_assignments.clear()
        now_wall = self._wall_clock()
        response = self._response(
            accepted=True,
            reason="rollback_applied",
            command=command,
            old_value=old_value,
            effective_value=effective,
            validated_wall_ns=now_wall,
            applied_wall_ns=now_wall,
            effective_from="next_decode_route",
        )
        self.command_records[command.control_command_id] = AppliedControl(command, response)
        return response

    def force_safety_fallback(self, reason: str) -> dict[str, Any] | None:
        if self.control_mode != "controlled":
            return None
        old_value = self.rho_a
        self.previous_safe_value = old_value
        self.control_mode = "baseline"
        self.control_status = "SAFE_BASELINE"
        self.rho_a = 0.5
        self.control_generation += 1
        self.current_command_id = f"safety-{self.control_generation}-{self._wall_clock()}"
        self.last_applied_mono_ns = self._mono_clock()
        self.prepared = None
        self.assignment_counts.clear()
        self.assignment_tokens.clear()
        self.recent_assignments.clear()
        self.safety_fallback_count += 1
        return {
            "accepted": True,
            "rejected": False,
            "old_value": old_value,
            "requested_value": "baseline",
            "effective_value": "baseline",
            "validated_wall_ns": self._wall_clock(),
            "applied_wall_ns": self._wall_clock(),
            "effective_from": "next_decode_route",
            "control_generation": self.control_generation,
            "control_command_id": self.current_command_id,
            "actuator_name": ROUTING_ACTUATOR,
            "requested_wall_ns": self._wall_clock(),
            "reason": reason,
        }

    def choose_decoder(
        self,
        ordered_keys: list[str],
        active_tokens: Mapping[str, float],
        tainted: set[str],
        request_tokens: float,
    ) -> dict[str, Any] | None:
        if not self.enabled or self.control_mode != "controlled":
            return None
        healthy = [key for key in ordered_keys if key not in tainted]
        if len(ordered_keys) != 2 or len(healthy) != 2:
            return {"selected": None, "safety_fallback": self.force_safety_fallback("decode_unhealthy_or_unavailable")}
        decoder_a, decoder_b = ordered_keys
        picked, deficits = weighted_fair_decoder(decoder_a, decoder_b, self.rho_a, self.assignment_counts)
        alternative = decoder_b if picked == decoder_a else decoder_a
        route_reason = "controlled_weighted_fair_deficit"
        if active_tokens[picked] > active_tokens[alternative] + self.safety.maximum_load_skew_tokens:
            picked = alternative
            route_reason = "controlled_load_guard_bypass"
        self.assignment_counts[picked] = self.assignment_counts.get(picked, 0) + 1
        self.assignment_tokens[picked] = self.assignment_tokens.get(picked, 0.0) + request_tokens
        self.recent_assignments.append((picked, request_tokens))
        return {
            "selected": picked,
            "route_reason": route_reason,
            "assignment_deficit": deficits,
            "control_command_id": self.current_command_id,
            "control_generation": self.control_generation,
            "requested_rho_A": self.rho_a,
            "effective_rho_A": self.rho_a,
            **self.ratio_snapshot(ordered_keys),
        }

    def ratio_snapshot(self, ordered_keys: list[str]) -> dict[str, Any]:
        if len(ordered_keys) != 2:
            return {"actual_recent_request_ratio": None, "actual_recent_token_ratio": None}
        decoder_a = ordered_keys[0]
        recent_count = len(self.recent_assignments)
        recent_tokens = sum(tokens for _, tokens in self.recent_assignments)
        return {
            "actual_request_ratio": (
                self.assignment_counts.get(decoder_a, 0) / max(sum(self.assignment_counts.values()), 1)
            ),
            "actual_token_ratio": (
                self.assignment_tokens.get(decoder_a, 0.0) / max(sum(self.assignment_tokens.values()), 1.0)
            ),
            "actual_recent_request_ratio": (
                sum(key == decoder_a for key, _ in self.recent_assignments) / max(recent_count, 1)
            ),
            "actual_recent_token_ratio": (
                sum(tokens for key, tokens in self.recent_assignments if key == decoder_a) / max(recent_tokens, 1.0)
            ),
            "controlled_assignment_counts": dict(self.assignment_counts),
            "controlled_assignment_tokens": dict(self.assignment_tokens),
        }

    def snapshot(self, ordered_keys: list[str], *, decoders_healthy: bool) -> dict[str, Any]:
        prepared = self.prepared
        return {
            "enabled": self.enabled,
            "control_mode": self.control_mode,
            "control_status": self.control_status,
            "control_generation": self.control_generation,
            "actuator_name": ROUTING_ACTUATOR,
            "effective_rho_A": self.rho_a if self.control_mode == "controlled" else None,
            "baseline_policy": "minimum_expected_remaining_tokens_with_fair_ties",
            "current_command_id": self.current_command_id,
            "previous_safe_value": self.previous_safe_value,
            "prepared_command_id": None if prepared is None else prepared.command.control_command_id,
            "decoders_healthy": decoders_healthy,
            "safety_fallback_count": self.safety_fallback_count,
            "safety": {
                "rho_A_range": [self.safety.minimum_rho_a, self.safety.maximum_rho_a],
                "maximum_step": self.safety.maximum_step,
                "minimum_dwell_ns": self.safety.minimum_dwell_ns,
                "maximum_load_skew_tokens": self.safety.maximum_load_skew_tokens,
            },
            **self.ratio_snapshot(ordered_keys),
        }
