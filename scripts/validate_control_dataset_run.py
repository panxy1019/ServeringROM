#!/usr/bin/env python3
"""Fail-closed quality gates for one Round 14.2 formal run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_proxy_events(root: Path):
    for path in sorted((root / "raw" / "proxy").glob("*.jsonl")):
        for number, line in enumerate(path.open(encoding="utf-8"), 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"damaged JSONL {path}:{number}: {exc}") from exc


def main() -> int:
    import numpy as np
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    reports = root / "reports"
    control = root / "derived" / "control"
    base = json.loads((reports / "snapshot_data_quality.json").read_text())
    run = json.loads((root / "metadata" / "run.json").read_text())
    workload = json.loads((root / "metadata" / "workload_result.json").read_text())
    u = np.load(control / "control_input.npy")[:, 0]
    windows = pq.read_table(control / "control_windows.parquet").to_pylist()
    slow = pq.read_table(control / "slow_control_kpi_windows.parquet").to_pylist()
    events = list(read_proxy_events(root))
    violations: list[str] = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            violations.append(reason)

    require(bool(base.get("valid")), "base_snapshot_invalid")
    require(len(windows) == len(u) == 3000, "fast_window_count_not_3000")
    require(len(slow) == 120, "slow_window_count_not_120")
    require(np.isfinite(u).all(), "control_array_nonfinite")
    require(all(row["u_source_event_type"] == "actuator_applied" for row in windows), "u_not_from_actuator_applied")
    require(all(str(row["control_command_id"]).startswith(run["run_id"]) for row in windows), "cross_run_control_command")
    require(set(round(float(value), 6) for value in u) >= {0.3, 0.5, 0.7}, "control_levels_incomplete")
    require(float(np.var(u)) >= 0.005, "control_variance_too_low")
    delta = np.diff(u)
    require(bool((np.abs(delta) <= 0.2000000001).all()), "control_delta_violation")
    transitions = np.flatnonzero(np.abs(delta) > 1e-12) + 1
    require(len(transitions) >= 15, "insufficient_control_transitions")
    transition_times = [int(windows[index]["control_applied_wall_ns"]) for index in transitions]
    if len(transition_times) > 1:
        require(min(np.diff(transition_times)) >= 5_000_000_000, "actual_dwell_below_5_seconds")
    command_by_id = {}
    control_rejects = 0
    safety_fallbacks = 0
    for event in events:
        payload = event.get("payload") or {}
        if event.get("event_type") == "actuator_applied":
            cid = str(payload.get("control_command_id"))
            if cid in command_by_id and command_by_id[cid] != payload:
                violations.append("conflicting_actuator_applied")
            command_by_id[cid] = payload
        if event.get("event_type") == "actuator_rejected":
            control_rejects += 1
        if event.get("event_type") == "actuator_safety_fallback":
            safety_fallbacks += 1
    used_ids = {str(row["control_command_id"]) for row in windows}
    require(all(cid in command_by_id for cid in used_ids), "u_without_unique_actuator_event")
    generations = [int(command_by_id[cid]["control_generation"]) for cid in command_by_id if cid.startswith(run["run_id"])]
    require(generations == sorted(set(generations)), "control_generation_not_strictly_monotonic")
    require(control_rejects == 0, "control_reject_present")
    require(safety_fallbacks == 0, "safety_fallback_present")
    require(bool(workload.get("drain", {}).get("drained")), "inventory_not_drained")
    require(run.get("topology") == {"prefill_tp": 2, "decode_a_tp": 2, "decode_b_tp": 2}, "topology_not_tp2_tp2_tp2")
    require(all(key in slow[0] for key in ("u_start", "u_mean", "u_end", "delta_u")), "slow_control_fields_missing")

    summary_files = list((root / "raw").glob("**/*.summary.json"))
    writer_drops = 0
    writer_mismatches = 0
    for path in summary_files:
        summary = json.loads(path.read_text())
        counters = summary.get("counters", summary)
        writer_drops += int(counters.get("events_dropped_queue_full", 0)) + int(counters.get("events_dropped_writer_failed", 0))
        if counters.get("events_written") != counters.get("events_enqueued"):
            writer_mismatches += 1
    require(writer_drops == 0, "writer_drop_nonzero")
    require(writer_mismatches == 0, "writer_counter_mismatch")

    report = {
        "schema_version": "servingrom.control_dataset_run_quality.v1",
        "valid": not violations,
        "violations": sorted(set(violations)),
        "metrics": {
            "fast_windows": len(windows), "slow_windows": len(slow), "u_defined_ratio": len(u) / 3000,
            "control_levels": sorted(set(float(value) for value in u)), "control_variance": float(np.var(u)),
            "control_transitions": int(len(transitions)), "max_abs_delta_u": float(np.max(np.abs(delta))),
            "minimum_actual_dwell_seconds": min(np.diff(transition_times)) / 1e9 if len(transition_times) > 1 else None,
            "control_rejects": control_rejects, "safety_fallbacks": safety_fallbacks,
            "writer_drops": writer_drops, "writer_mismatches": writer_mismatches,
            "base_snapshot_valid": bool(base.get("valid")),
        },
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "control_dataset_run_quality.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Formal Control Run 质量报告", "", f"- valid: `{report['valid']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in report["metrics"].items())
    if violations:
        lines.extend(["", "## Violations", "", *(f"- `{value}`" for value in report["violations"])])
    (reports / "control_dataset_run_quality.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
