from __future__ import annotations

from unittest import TestCase

from servingrom_control import RoutingSafetyConfig, RuntimeControlManager


class FakeClock:
    def __init__(self) -> None:
        self.wall = 1_000_000_000_000
        self.mono = 1_000_000_000

    def wall_ns(self) -> int:
        return self.wall

    def mono_ns(self) -> int:
        return self.mono

    def advance(self, seconds: float) -> None:
        delta = int(seconds * 1_000_000_000)
        self.wall += delta
        self.mono += delta


class RuntimeControlManagerTest(TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.manager = RuntimeControlManager(
            enabled=True,
            safety=RoutingSafetyConfig(minimum_dwell_ns=5_000_000_000),
            wall_clock=self.clock.wall_ns,
            mono_clock=self.clock.mono_ns,
        )

    def command(self, command_id: str, generation: int, requested, expected):
        return {
            "control_command_id": command_id,
            "control_generation": generation,
            "actuator_name": "decode_routing_ratio",
            "requested_value": requested,
            "expected_current_value": expected,
            "requested_wall_ns": self.clock.wall_ns(),
        }

    def apply(self, command):
        prepared = self.manager.prepare(command, decoders_healthy=True)
        self.assertTrue(prepared["accepted"], prepared)
        committed = self.manager.commit(command)
        self.assertTrue(committed["accepted"], committed)
        return committed

    def test_weighted_fair_ratio_converges_deterministically(self):
        command = self.command("ratio-03", 1, 0.3, "baseline")
        self.apply(command)
        keys = ["decode-a", "decode-b"]
        selected = [
            self.manager.choose_decoder(keys, {key: 0.0 for key in keys}, set(), 128.0)["selected"]
            for _ in range(100)
        ]
        self.assertEqual(selected.count("decode-a"), 30)
        snapshot = self.manager.snapshot(keys, decoders_healthy=True)
        self.assertAlmostEqual(snapshot["actual_recent_request_ratio"], 0.3)
        self.assertAlmostEqual(snapshot["actual_recent_token_ratio"], 0.3)

    def test_prepare_is_fail_closed_and_commit_is_idempotent(self):
        bad = self.command("out-of-range", 1, 0.1, "baseline")
        self.assertEqual(
            self.manager.prepare(bad, decoders_healthy=True)["reason"],
            "requested_value_out_of_range",
        )
        command = self.command("ratio-05", 1, 0.5, "baseline")
        self.assertTrue(self.manager.prepare(command, decoders_healthy=True)["accepted"])
        self.assertEqual(
            self.manager.prepare(command, decoders_healthy=True)["reason"],
            "duplicate_command_id",
        )
        first = self.manager.commit(command)
        replay = self.manager.commit(command)
        self.assertTrue(first["accepted"])
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(first["applied_wall_ns"], replay["applied_wall_ns"])

    def test_generation_dwell_and_rollback(self):
        self.apply(self.command("ratio-05", 1, 0.5, "baseline"))
        too_soon = self.command("ratio-03", 2, 0.3, 0.5)
        self.assertEqual(
            self.manager.prepare(too_soon, decoders_healthy=True)["reason"],
            "minimum_dwell_time_not_met",
        )
        self.clock.advance(5)
        stale = self.command("stale", 1, 0.3, 0.5)
        self.assertEqual(
            self.manager.prepare(stale, decoders_healthy=True)["reason"],
            "stale_or_nonmonotonic_generation",
        )
        self.apply(self.command("ratio-03-valid", 2, 0.3, 0.5))
        rollback = self.command("rollback", 3, "baseline", 0.3)
        result = self.manager.rollback(rollback)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["effective_value"], "baseline")
        self.assertEqual(self.manager.control_mode, "baseline")

    def test_unhealthy_decoder_forces_safe_baseline(self):
        self.apply(self.command("ratio-07", 1, 0.7, "baseline"))
        result = self.manager.choose_decoder(
            ["decode-a", "decode-b"],
            {"decode-a": 0.0, "decode-b": 0.0},
            {"decode-b"},
            128.0,
        )
        self.assertIsNone(result["selected"])
        self.assertEqual(self.manager.control_mode, "baseline")
        self.assertEqual(self.manager.control_status, "SAFE_BASELINE")
        self.assertEqual(result["safety_fallback"]["reason"], "decode_unhealthy_or_unavailable")

    def test_load_guard_can_override_ratio_without_losing_accounting(self):
        self.apply(self.command("ratio-08", 1, 0.7, "baseline"))
        result = self.manager.choose_decoder(
            ["decode-a", "decode-b"],
            {"decode-a": 4096.0, "decode-b": 0.0},
            set(),
            128.0,
        )
        self.assertEqual(result["selected"], "decode-b")
        self.assertEqual(result["route_reason"], "controlled_load_guard_bypass")
