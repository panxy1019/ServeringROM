from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


TERMINAL_EVENTS = frozenset(
    {"request_rejected", "request_complete", "request_cancel", "request_error"}
)


@dataclass(slots=True)
class ProxyLifecycleAnalysis:
    trace_rows: list[dict[str, Any]]
    attempt_rows: list[dict[str, Any]]
    violations: list[dict[str, Any]]
    metrics: dict[str, Any]


def _violation(
    violations: list[dict[str, Any]],
    code: str,
    *,
    trace_id: str | None = None,
    attempt_id: int | None = None,
    detail: str = "",
) -> None:
    violations.append(
        {"code": code, "trace_id": trace_id, "attempt_id": attempt_id, "detail": detail}
    )


def _first(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    return next((event for event in events if event["event_type"] == event_type), None)


def _last(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    return next((event for event in reversed(events) if event["event_type"] == event_type), None)


def _ns(event: dict[str, Any] | None) -> int | None:
    return None if event is None else int(event["ts_mono_ns"])


def _payload(event: dict[str, Any] | None, name: str, default: Any = None) -> Any:
    return default if event is None else event.get("payload", {}).get(name, default)


def _json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_process_sequences(
    events: list[dict[str, Any]], violations: list[dict[str, Any]]
) -> dict[str, int]:
    by_process: dict[str, list[int]] = defaultdict(list)
    for event in events:
        by_process[event["process_instance_id"]].append(event["event_seq"])
    gaps: dict[str, int] = {}
    for process_id, sequence in by_process.items():
        expected = list(range(1, max(sequence) + 1)) if sequence else []
        gap_count = len(set(expected) - set(sequence)) + len(sequence) - len(set(sequence))
        gaps[process_id] = gap_count
        if gap_count:
            _violation(
                violations,
                "event_seq_gap",
                detail=f"process={process_id} gap_or_duplicate_count={gap_count}",
            )
    return gaps


def _validate_writer_summaries(
    summaries: list[dict[str, Any]], violations: list[dict[str, Any]]
) -> dict[str, int]:
    totals = Counter()
    for summary in summaries:
        for field in (
            "events_attempted",
            "events_enqueued",
            "events_written",
            "events_dropped_queue_full",
            "events_dropped_writer_failed",
            "serialization_errors",
            "write_errors",
        ):
            totals[field] += int(summary.get(field, 0))
        if int(summary.get("events_written", 0)) != int(summary.get("events_enqueued", 0)):
            _violation(
                violations,
                "writer_counter_mismatch",
                detail=f"process={summary.get('process_instance_id')}",
            )
    return dict(totals)


def analyze_proxy_lifecycle(
    events: Iterable[dict[str, Any]],
    summaries: Iterable[dict[str, Any]] = (),
    damaged_lines: Iterable[dict[str, Any]] = (),
) -> ProxyLifecycleAnalysis:
    ordered = list(events)
    summary_list = list(summaries)
    damaged = list(damaged_lines)
    violations: list[dict[str, Any]] = []
    sequence_gaps = _validate_process_sequences(ordered, violations)
    writer_totals = _validate_writer_summaries(summary_list, violations)
    for item in damaged:
        _violation(violations, "jsonl_damaged_line", detail=_json(item) or "")

    trace_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in ordered:
        trace_id = event.get("trace_id")
        if trace_id is not None:
            trace_events[trace_id].append(event)

    trace_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for trace_id, trace in sorted(trace_events.items()):
        trace.sort(key=lambda event: (event["ts_mono_ns"], event["event_seq"]))
        arrivals = [event for event in trace if event["event_type"] == "request_arrival"]
        terminals = [event for event in trace if event["event_type"] in TERMINAL_EVENTS]
        admissions = [event for event in trace if event["event_type"] == "admission_decision"]
        accepted = any(bool(_payload(event, "accepted", False)) for event in admissions)
        rejected = any(event["event_type"] == "request_rejected" for event in terminals)

        if not arrivals:
            _violation(violations, "arrival_missing", trace_id=trace_id)
        if len(terminals) > 1:
            _violation(violations, "duplicate_terminal_event", trace_id=trace_id)
        if not terminals:
            _violation(violations, "terminal_event_missing", trace_id=trace_id)
        if accepted and not terminals:
            _violation(violations, "accepted_request_without_terminal_state", trace_id=trace_id)
        if rejected and any(event["event_type"] == "prefill_http_submit" for event in trace):
            _violation(violations, "rejected_request_with_prefill_submit", trace_id=trace_id)

        attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in trace:
            attempt_id = event.get("attempt_id")
            if attempt_id is not None:
                attempts[int(attempt_id)].append(event)
        attempt_ids = sorted(attempts)
        if attempt_ids and attempt_ids != list(range(attempt_ids[-1] + 1)):
            _violation(
                violations,
                "attempt_id_gap",
                trace_id=trace_id,
                detail=f"attempt_ids={attempt_ids}",
            )

        request_to_attempts: dict[str, set[int]] = defaultdict(set)
        for attempt_id, attempt in attempts.items():
            for event in attempt:
                if event.get("request_id"):
                    request_to_attempts[event["request_id"]].add(attempt_id)
        for request_id, used_by in request_to_attempts.items():
            if len(used_by) > 1:
                _violation(
                    violations,
                    "request_id_reused_across_attempts",
                    trace_id=trace_id,
                    detail=f"request_id={request_id} attempts={sorted(used_by)}",
                )

        for attempt_id, attempt in sorted(attempts.items()):
            attempt.sort(key=lambda event: (event["ts_mono_ns"], event["event_seq"]))
            event_types = [event["event_type"] for event in attempt]
            prefill_submit = _first(attempt, "prefill_http_submit")
            prefill_complete = _last(attempt, "prefill_http_complete")
            route = _last(attempt, "p_to_d_route")
            decode_submit = _last(attempt, "decode_http_submit")
            first_byte = _first(attempt, "decode_first_byte")
            terminal = next((event for event in attempt if event["event_type"] in TERMINAL_EVENTS), None)
            recomputed = _first(attempt, "attempt_recomputed")

            checks = (
                (prefill_complete, prefill_submit, "prefill_complete_without_submit"),
                (route, prefill_complete, "route_without_prefill_complete"),
                (decode_submit, route, "decode_submit_without_route"),
                (first_byte, decode_submit, "first_byte_without_decode_submit"),
            )
            for later, earlier, code in checks:
                if later is not None and earlier is None:
                    _violation(violations, code, trace_id=trace_id, attempt_id=attempt_id)
                elif later is not None and earlier is not None and _ns(later) < _ns(earlier):
                    _violation(
                        violations,
                        "event_order_violation",
                        trace_id=trace_id,
                        attempt_id=attempt_id,
                        detail=f"{later['event_type']} before {earlier['event_type']}",
                    )
            if terminal and terminal["event_type"] == "request_complete":
                if first_byte is None or _ns(terminal) < _ns(first_byte):
                    _violation(
                        violations,
                        "completion_before_first_byte",
                        trace_id=trace_id,
                        attempt_id=attempt_id,
                        detail="first byte missing" if first_byte is None else "completion timestamp precedes first byte",
                    )
            if recomputed:
                payload = recomputed["payload"]
                if payload.get("new_attempt_id") != attempt_id or payload.get("new_request_id") != recomputed.get("request_id"):
                    _violation(
                        violations,
                        "trace_id_changed_after_recompute",
                        trace_id=trace_id,
                        attempt_id=attempt_id,
                        detail="recompute context does not match event identity",
                    )

            request_ids = {event["request_id"] for event in attempt if event.get("request_id")}
            request_id = next(iter(request_ids)) if len(request_ids) == 1 else None
            if len(request_ids) > 1:
                _violation(
                    violations,
                    "event_order_violation",
                    trace_id=trace_id,
                    attempt_id=attempt_id,
                    detail=f"multiple request IDs within attempt: {sorted(request_ids)}",
                )

            attempt_rows.append(
                {
                    "trace_id": trace_id,
                    "attempt_id": attempt_id,
                    "request_id": request_id,
                    "external_request_id": attempt[0].get("external_request_id"),
                    "process_instance_id": attempt[0]["process_instance_id"],
                    "prefill_submit_mono_ns": _ns(prefill_submit),
                    "prefill_complete_mono_ns": _ns(prefill_complete),
                    "prefill_duration_ns": _payload(prefill_complete, "duration_ns"),
                    "prefill_backend": _payload(prefill_submit, "backend_endpoint"),
                    "route_mono_ns": _ns(route),
                    "decode_submit_mono_ns": _ns(decode_submit),
                    "decode_first_byte_mono_ns": _ns(first_byte),
                    "decoder_backend": _payload(route, "selected_decoder"),
                    "route_reason": _payload(route, "route_reason"),
                    "route_snapshot_json": _json(route.get("payload") if route else None),
                    "decode_stream_chunks": event_types.count("decode_stream_chunk"),
                    "backend_retries": event_types.count("backend_retry"),
                    "recomputed_from_attempt_id": _payload(recomputed, "previous_attempt_id"),
                    "terminal_event": terminal["event_type"] if terminal else None,
                    "attempt_start_mono_ns": min(event["ts_mono_ns"] for event in attempt),
                    "attempt_end_mono_ns": max(event["ts_mono_ns"] for event in attempt),
                    "event_count": len(attempt),
                }
            )

        arrival = arrivals[0] if arrivals else None
        terminal = terminals[0] if len(terminals) == 1 else None
        first_byte = _first(trace, "decode_first_byte")
        trace_rows.append(
            {
                "trace_id": trace_id,
                "external_request_id": trace[0].get("external_request_id"),
                "process_instance_id": trace[0]["process_instance_id"],
                "arrival_wall_ns": _payload(arrival, "arrival_wall_ns", arrival.get("ts_wall_ns") if arrival else None),
                "arrival_mono_ns": _payload(arrival, "arrival_mono_ns", _ns(arrival)),
                "input_tokens": _payload(arrival, "input_tokens"),
                "expected_output_tokens": _payload(arrival, "expected_output_tokens"),
                "stream": _payload(arrival, "stream"),
                "accepted": accepted,
                "terminal_event": terminal["event_type"] if terminal else None,
                "terminal_wall_ns": terminal.get("ts_wall_ns") if terminal else None,
                "terminal_mono_ns": _ns(terminal),
                "output_tokens": _payload(terminal, "output_tokens"),
                "output_sha256": _payload(terminal, "output_sha256"),
                "attempt_count": len(attempts),
                "backend_retry_count": sum(event["event_type"] == "backend_retry" for event in trace),
                "first_byte_mono_ns": _ns(first_byte),
                "ttft_proxy_ns": (
                    _ns(first_byte) - _payload(arrival, "arrival_mono_ns", _ns(arrival))
                    if first_byte and arrival
                    else None
                ),
                "e2e_ns": (
                    _ns(terminal) - _payload(arrival, "arrival_mono_ns", _ns(arrival))
                    if terminal and arrival
                    else None
                ),
                "event_count": len(trace),
            }
        )

    violation_counts = Counter(item["code"] for item in violations)
    metrics = {
        "event_count": len(ordered),
        "trace_count": len(trace_rows),
        "attempt_count": len(attempt_rows),
        "accepted_trace_count": sum(bool(row["accepted"]) for row in trace_rows),
        "terminal_trace_count": sum(row["terminal_event"] is not None for row in trace_rows),
        "violation_count": len(violations),
        "violation_counts": dict(sorted(violation_counts.items())),
        "event_seq_gap_count": sum(sequence_gaps.values()),
        "damaged_line_count": len(damaged),
        "writer_totals": writer_totals,
    }
    return ProxyLifecycleAnalysis(trace_rows, attempt_rows, violations, metrics)
