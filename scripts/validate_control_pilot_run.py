#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    import numpy as np
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    reports = root / "reports"
    control_dir = root / "derived" / "control"
    base = json.loads((reports / "snapshot_data_quality.json").read_text())
    u = np.load(control_dir / "control_input.npy")
    aux = np.load(control_dir / "control_auxiliary.npy")
    windows = pq.read_table(control_dir / "control_windows.parquet").to_pylist()
    slow = pq.read_table(control_dir / "slow_control_kpi_windows.parquet").to_pylist()
    measurement = json.loads((root / "metadata" / "measurement.json").read_text())
    expected = (int(measurement["measurement_end_wall_ns"]) - int(measurement["measurement_start_wall_ns"])) // 200_000_000
    violations = []
    def require(condition: bool, reason: str):
        if not condition:
            violations.append(reason)
    require(bool(base.get("valid")), "base_snapshot_invalid")
    require(len(windows) == expected == len(u) == len(aux), "fast_window_count_mismatch")
    require(expected == 3000, "pilot_measurement_not_600_seconds")
    require(len(slow) == 120, "slow_window_count_mismatch")
    require(np.isfinite(u).all() and np.isfinite(aux).all(), "control_array_nonfinite")
    require(bool(((u[:, 0] >= 0.2) & (u[:, 0] <= 0.8)).all()), "control_out_of_range")
    require(bool((np.abs(aux[:, 1]) <= 0.2000000001).all()), "control_delta_violation")
    require(all(row["u_source_event_type"] == "actuator_applied" for row in windows), "invalid_u_source")
    require(all(row["control_command_id"] for row in windows), "control_command_id_missing")
    require(all(row["control_generation"] >= 1 for row in windows), "control_generation_missing")
    changed = [row for row in windows if abs(float(row["delta_u"])) > 1e-12]
    require(len(changed) >= 4, "insufficient_control_transitions")
    unique_levels = sorted({round(float(value), 6) for value in u[:, 0]})
    require(set(unique_levels) >= {0.3, 0.5, 0.7}, "control_levels_incomplete")
    report = {
        "schema_version": "servingrom.control_pilot_quality.v1",
        "valid": not violations,
        "violations": violations,
        "metrics": {
            "fast_windows": len(windows), "slow_windows": len(slow),
            "control_levels": unique_levels, "control_transition_windows": len(changed),
            "u_min": float(u[:, 0].min()), "u_max": float(u[:, 0].max()),
            "max_abs_delta_u": float(np.abs(aux[:, 1]).max()),
            "base_snapshot_valid": bool(base.get("valid")),
        },
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "control_pilot_quality.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Control Pilot Run 质量报告", "", f"- valid: `{report['valid']}`"]
    lines += [f"- {name}: `{value}`" for name, value in report["metrics"].items()]
    if violations:
        lines += ["", "## Violations", ""] + [f"- `{value}`" for value in violations]
    (reports / "control_pilot_quality.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
