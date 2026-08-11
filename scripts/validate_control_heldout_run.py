#!/usr/bin/env python3
"""Fail-closed per-run quality gates for Round 14.3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_LEVELS = {
    "interpolation": {0.4, 0.5, 0.6},
    "unseen-composite": {0.3, 0.4, 0.5, 0.6, 0.7},
    "slow-ramp": {0.3, 0.4, 0.5, 0.6, 0.7},
    "boundary-near": {0.2, 0.4, 0.5, 0.6, 0.8},
}


def proxy_events(root: Path):
    for path in sorted((root / "raw" / "proxy").glob("*.jsonl")):
        for number, line in enumerate(path.open(encoding="utf-8"), 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"damaged JSONL {path}:{number}: {exc}") from exc


def main() -> int:
    import numpy as np
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(); parser.add_argument("run_root", type=Path)
    root = parser.parse_args().run_root
    reports, control = root / "reports", root / "derived" / "control"
    base = json.loads((reports / "snapshot_data_quality.json").read_text())
    run = json.loads((root / "metadata" / "run.json").read_text())
    workload = json.loads((root / "metadata" / "workload_result.json").read_text())
    u = np.load(control / "control_input.npy")[:, 0]
    windows = pq.read_table(control / "control_windows.parquet").to_pylist()
    slow = pq.read_table(control / "slow_control_kpi_windows.parquet").to_pylist()
    events = list(proxy_events(root)); violations = []

    def require(condition, reason):
        if not condition: violations.append(reason)

    family = run["trajectory_family"]
    levels = {round(float(value), 6) for value in u}
    require(bool(base.get("valid")), "base_snapshot_invalid")
    require(len(windows) == len(u) == 3000, "fast_window_count_not_3000")
    require(len(slow) == 120, "slow_window_count_not_120")
    require(np.isfinite(u).all(), "control_array_nonfinite")
    require(EXPECTED_LEVELS[family] <= levels, "trajectory_levels_incomplete")
    require(float(np.var(u)) >= 0.002, "control_variance_too_low")
    delta = np.diff(u); transitions = np.flatnonzero(np.abs(delta) > 1e-12) + 1
    require(bool((np.abs(delta) <= 0.2000000001).all()), "control_delta_violation")
    require(len(transitions) >= 10, "insufficient_control_transitions")
    require(all(row["u_source_event_type"] == "actuator_applied" for row in windows), "invalid_u_source")
    require(all(str(row["control_command_id"]).startswith(run["run_id"]) for row in windows), "cross_run_control_command")
    applied_times = [int(windows[index]["control_applied_wall_ns"]) for index in transitions]
    if len(applied_times) > 1:
        require(min(np.diff(applied_times)) >= 5_000_000_000, "actual_dwell_below_5_seconds")
    event_types = [event["event_type"] for event in events]
    require("actuator_rejected" not in event_types, "control_reject_present")
    require("actuator_safety_fallback" not in event_types, "safety_fallback_present")
    require(bool(workload.get("drain", {}).get("drained")), "inventory_not_drained")
    require(workload.get("summary", {}).get("error_count") == 0, "workload_error_present")
    require(all(sample.get("status") == 200 for sample in workload.get("health_samples", [])), "health_sample_failure")
    require(run.get("split") == "test/control-heldout", "invalid_split")
    require(run.get("topology") == {"prefill_tp": 2, "decode_a_tp": 2, "decode_b_tp": 2}, "topology_drift")
    require(run.get("pod_uid_before") == run.get("pod_uid_after"), "pod_uid_changed")
    require(run.get("restart_before") == run.get("restart_after") == 0, "pod_restart_detected")
    writer_drops = writer_mismatches = 0
    for path in (root / "raw").glob("**/*.summary.json"):
        counters = json.loads(path.read_text()).get("counters", json.loads(path.read_text()))
        writer_drops += int(counters.get("events_dropped_queue_full", 0)) + int(counters.get("events_dropped_writer_failed", 0))
        if counters.get("events_written") != counters.get("events_enqueued"): writer_mismatches += 1
    require(writer_drops == 0, "writer_drop_nonzero")
    require(writer_mismatches == 0, "writer_counter_mismatch")
    report = {
        "schema_version": "servingrom.control_heldout_run_quality.v1", "valid": not violations,
        "heldout_control_quality_pass": not violations, "violations": sorted(set(violations)),
        "metrics": {"family": family, "fast_windows": len(windows), "slow_windows": len(slow),
                    "u_defined_ratio": len(u) / 3000, "control_levels": sorted(levels),
                    "control_variance": float(np.var(u)), "transitions": int(len(transitions)),
                    "max_abs_delta_u": float(np.abs(delta).max()),
                    "minimum_actual_dwell_seconds": min(np.diff(applied_times)) / 1e9 if len(applied_times) > 1 else None,
                    "writer_drops": writer_drops, "writer_mismatches": writer_mismatches,
                    "base_snapshot_valid": bool(base.get("valid"))},
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "control_heldout_run_quality.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Held-out Control Run 质量报告", "", f"- valid: `{report['valid']}`"]
    lines += [f"- {key}: `{value}`" for key, value in report["metrics"].items()]
    if violations: lines += ["", "## Violations", "", *(f"- `{value}`" for value in report["violations"])]
    (reports / "control_heldout_run_quality.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2)); return 0 if report["valid"] else 1


if __name__ == "__main__": raise SystemExit(main())
