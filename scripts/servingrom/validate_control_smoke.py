#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTROL_TYPES = {
    "actuator_command_received",
    "actuator_command_validated",
    "actuator_applied",
    "actuator_rejected",
    "actuator_rollback",
    "actuator_safety_fallback",
}
CONTROL_FIELDS = {
    "control_command_id",
    "control_generation",
    "actuator_name",
    "old_value",
    "requested_value",
    "effective_value",
    "requested_wall_ns",
    "applied_wall_ns",
    "reason",
}


def load_events(paths: list[Path]):
    events = []
    damaged = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    events.append(json.loads(line))
                except Exception as exc:
                    damaged.append({"path": str(path), "line": line_number, "error": repr(exc)})
    return events, damaged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-report", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    smoke = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
    paths = sorted(Path(args.raw_dir).glob("*.jsonl"))
    events, damaged = load_events(paths)
    by_process: dict[str, list[int]] = defaultdict(list)
    type_counts = Counter()
    field_errors = []
    attempt_routes: dict[tuple[Any, Any], set[str]] = defaultdict(set)
    command_routes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    applied = {}
    for event in events:
        by_process[event["process_instance_id"]].append(event["event_seq"])
        type_counts[event["event_type"]] += 1
        payload = event.get("payload") or {}
        if event["event_type"] in CONTROL_TYPES:
            missing = sorted(CONTROL_FIELDS - set(payload))
            if missing:
                field_errors.append({"event_seq": event["event_seq"], "missing": missing})
        if event["event_type"] == "actuator_applied":
            applied[payload["control_command_id"]] = payload
        if event["event_type"] == "p_to_d_route":
            key = (event.get("trace_id"), event.get("attempt_id"))
            attempt_routes[key].add(payload["selected_decoder"])
            command_id = payload.get("control_command_id")
            if command_id:
                command_routes[command_id].append(event)

    sequence_gaps = {}
    sequence_duplicates = {}
    for process, values in by_process.items():
        ordered = sorted(values)
        expected = set(range(ordered[0], ordered[-1] + 1)) if ordered else set()
        missing = sorted(expected - set(ordered))
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if missing:
            sequence_gaps[process] = missing
        if duplicates:
            sequence_duplicates[process] = duplicates

    first_effect_errors = []
    for command_id, payload in applied.items():
        routes = command_routes.get(command_id) or []
        if not routes:
            first_effect_errors.append({"command_id": command_id, "reason": "no_controlled_route"})
            continue
        if routes[0]["ts_wall_ns"] < payload["applied_wall_ns"]:
            first_effect_errors.append({"command_id": command_id, "reason": "route_before_apply"})

    stage_ratios = {}
    for stage in smoke["stages"]:
        state = stage["state"]
        if state.get("effective_rho_A") is not None:
            stage_ratios[stage["name"]] = {
                "target_request_ratio": state["effective_rho_A"],
                "actual_request_ratio": state["actual_request_ratio"],
                "actual_token_ratio": state["actual_token_ratio"],
                "assignment_counts": state["controlled_assignment_counts"],
            }

    writer = smoke["telemetry_health"]
    checks = {
        "smoke_driver_passed": smoke["passed"],
        "output_sha256_consistent": smoke["output_sha256_consistent"],
        "final_baseline": smoke["final_state"]["control_mode"] == "baseline",
        "jsonl_corruption_zero": not damaged,
        "event_seq_gap_zero": not sequence_gaps and not sequence_duplicates,
        "control_fields_complete": not field_errors,
        "first_effect_after_apply": not first_effect_errors,
        "attempt_decoder_migration_zero": all(len(value) == 1 for value in attempt_routes.values()),
        "writer_written_equals_enqueued": writer["events_written"] == writer["events_enqueued"],
        "writer_drop_zero": writer["events_dropped_queue_full"] == 0 and writer["events_dropped_writer_failed"] == 0,
        "writer_errors_zero": all(writer[name] == 0 for name in ("serialization_errors", "event_build_errors", "write_errors", "flush_errors")),
        "all_control_event_types_present": CONTROL_TYPES <= set(type_counts),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "raw_files": [str(path) for path in paths],
        "event_count": len(events),
        "event_type_counts": dict(sorted(type_counts.items())),
        "sequence_gaps": sequence_gaps,
        "sequence_duplicates": sequence_duplicates,
        "damaged_lines": damaged,
        "control_field_errors": field_errors,
        "first_effect_errors": first_effect_errors,
        "attempts_with_decoder_migration": [
            {"trace_id": key[0], "attempt_id": key[1], "decoders": sorted(value)}
            for key, value in attempt_routes.items() if len(value) != 1
        ],
        "output_sha256_values": smoke["output_sha256_values"],
        "stage_ratios": stage_ratios,
        "writer": writer,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output), "events": len(events)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
