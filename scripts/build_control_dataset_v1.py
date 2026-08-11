#!/usr/bin/env python3
"""Seal the immutable Round 14.2 Control Dataset v1 from 36 sealed runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cohen_d(low: np.ndarray, high: np.ndarray) -> float | None:
    low, high = low[np.isfinite(low)], high[np.isfinite(high)]
    if len(low) < 2 or len(high) < 2:
        return None
    pooled = math.sqrt(((len(low) - 1) * low.var(ddof=1) + (len(high) - 1) * high.var(ddof=1)) / (len(low) + len(high) - 2))
    return float((high.mean() - low.mean()) / pooled) if pooled > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = manifest["runs"]
    if len(rows) != 36 or any(row["status"] != "SEALED" for row in rows):
        raise RuntimeError("formal dataset requires 36/36 SEALED runs")
    output = args.output
    if output.exists():
        raise FileExistsError(f"immutable dataset output already exists: {output}")
    output.mkdir(parents=True)

    by_split: dict[str, dict[str, list[np.ndarray]]] = {
        split: {name: [] for name in ("X", "D", "U", "X_next", "U_aux")}
        for split in ("train", "validation", "test")
    }
    slow_rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []
    run_index = []
    offsets = defaultdict(int)
    reference_files = None
    for row in rows:
        root = args.runs_root / row["run_id"]
        quality = json.loads((root / "reports" / "control_dataset_run_quality.json").read_text())
        run_status = json.loads((root / "metadata" / "run_status.json").read_text())
        if not quality["valid"] or run_status.get("status") != "SEALED":
            raise RuntimeError(f"run is not valid and sealed: {row['run_id']}")
        snap, control = root / "derived" / "snapshots", root / "derived" / "control"
        arrays = {
            "X": np.load(snap / "full_state.npy"), "D": np.load(snap / "disturbance.npy"),
            "U": np.load(control / "control_input.npy"), "X_next": np.load(snap / "next_state.npy"),
            "U_aux": np.load(control / "control_auxiliary.npy"),
        }
        if any(len(value) != 3000 for value in arrays.values()):
            raise RuntimeError(f"unexpected row count: {row['run_id']}")
        split = row["split"]
        start, stop = offsets[split], offsets[split] + 3000
        offsets[split] = stop
        run_index.append({
            **{key: row[key] for key in (
                "plan_id", "run_id", "split", "workload", "load_fraction",
                "arrival_process", "arrival_seed",
            )},
            # Control seeds use the complete unsigned 64-bit SHA256-derived
            # range. Store the audit value losslessly instead of relying on
            # Arrow's signed int64 inference.
            "control_seed": str(row["control_seed"]),
            "row_start": start,
            "row_stop": stop,
        })
        for name, value in arrays.items():
            by_split[split][name].append(value)
        controls = pq.read_table(control / "control_windows.parquet").to_pylist()
        states = arrays["X"]
        state_index = {item["name"]: int(item["index"]) for item in json.loads((snap / "state_index.json").read_text())}
        for index, control_row in enumerate(controls):
            authority_rows.append({
                "workload": row["workload"], "load_fraction": row["load_fraction"],
                "arrival_process": row["arrival_process"], "split": split, "run_id": row["run_id"],
                "u": float(control_row["u_rho_A"]),
                "routed": int(control_row["routed_request_count"]), "routed_a": int(control_row["routed_A_request_count"]),
                "running_imbalance": float(states[index, state_index["decode_d1_running_count"]] - states[index, state_index["decode_d2_running_count"]]),
                "waiting_imbalance": float(states[index, state_index["decode_d1_waiting_count"]] - states[index, state_index["decode_d2_waiting_count"]]),
                "remaining_imbalance": float(states[index, state_index["decode_d1_expected_remaining_tokens"]] - states[index, state_index["decode_d2_expected_remaining_tokens"]]),
            })
        for slow in pq.read_table(control / "slow_control_kpi_windows.parquet").to_pylist():
            slow_rows.append({"plan_id": row["plan_id"], "run_id": row["run_id"], "split": split,
                              "workload": row["workload"], "load_fraction": row["load_fraction"],
                              "arrival_process": row["arrival_process"], **slow})
        current_refs = {
            name: source for name, source in {
                "state_index.json": snap / "state_index.json", "disturbance_index.json": snap / "disturbance_index.json",
                "output_index.json": snap / "output_index.json", "control_index.json": control / "control_index.json",
                "control_auxiliary_index.json": control / "control_auxiliary_index.json",
            }.items()
        }
        hashes = {name: sha256(path) for name, path in current_refs.items()}
        if reference_files is None:
            reference_files = (current_refs, hashes)
        elif hashes != reference_files[1]:
            raise RuntimeError(f"schema drift in {row['run_id']}")

    for split, parts in by_split.items():
        split_dir = output / split
        split_dir.mkdir()
        for name, values in parts.items():
            np.save(split_dir / f"{name}.npy", np.concatenate(values, axis=0))
    if offsets != {"train": 36000, "validation": 36000, "test": 36000}:
        raise RuntimeError(f"split isolation/count failure: {dict(offsets)}")
    pq.write_table(pa.Table.from_pylist(slow_rows), output / "slow_kpi_windows.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(run_index), output / "run_index.parquet", compression="zstd")
    for name, source in reference_files[0].items():
        shutil.copy2(source, output / name)

    grouped = defaultdict(list)
    for row in authority_rows:
        grouped[(row["workload"], row["load_fraction"], row["arrival_process"])].append(row)
    authority = []
    for key, group in sorted(grouped.items()):
        low = [item for item in group if abs(item["u"] - 0.3) < 1e-9]
        high = [item for item in group if abs(item["u"] - 0.7) < 1e-9]
        routed_low = sum(item["routed_a"] for item in low) / max(1, sum(item["routed"] for item in low))
        routed_high = sum(item["routed_a"] for item in high) / max(1, sum(item["routed"] for item in high))
        effects = {}
        for name in ("running_imbalance", "waiting_imbalance", "remaining_imbalance"):
            lo = np.asarray([item[name] for item in low]); hi = np.asarray([item[name] for item in high])
            effects[name] = {"low_mean": float(lo.mean()), "high_mean": float(hi.mean()), "cohen_d": cohen_d(lo, hi)}
        effect_pass = any(value["cohen_d"] is not None and value["cohen_d"] >= 0.25 for value in effects.values())
        authority.append({"workload": key[0], "load_fraction": key[1], "arrival_process": key[2],
                          "routed_fraction_low": routed_low, "routed_fraction_high": routed_high,
                          "route_direction_pass": routed_high > routed_low, "state_effect_pass": effect_pass,
                          "effects": effects})
    authority_pass = len(authority) == 12 and all(row["route_direction_pass"] and row["state_effect_pass"] for row in authority)
    quality = {
        "schema_version": "servingrom.control_dataset_quality.v1", "runs": 36,
        "fast_windows": sum(offsets.values()), "slow_windows": len(slow_rows), "split_fast_windows": dict(offsets),
        "all_run_quality_pass": True, "control_authority_groups": authority,
        "control_dataset_ready": True, "control_identifiability_ready": authority_pass,
    }
    write_json(output / "quality_summary.json", quality)
    write_json(output / "dataset_manifest.json", {
        "dataset_id": "servingrom-control-dataset-v1", "immutable": True,
        "source_campaign_manifest_sha256": sha256(args.manifest), "run_count": 36,
        "split_policy": "whole-run isolation; seed 101=train, 202=validation, 303=test",
        "dimensions": {"X": 1804, "D": 31, "U": 1, "X_next": 1804},
    })
    report = ["# ServingROM Control Dataset v1", "", "## 结论", "",
              "- `control_dataset_ready=true`", f"- `control_identifiability_ready={str(authority_pass).lower()}`",
              "- 36/36 run 已通过单 run 质量门并封存。", "- 拓扑：Prefill TP2 + Decode A TP2 + Decode B TP2。",
              "- 未采集 held-out actuator trajectory，未训练 Control-ROM，未实现 MPC。"]
    (output / "CONTROL_DATASET_V1_REPORT.md").write_text("\n".join(report) + "\n")
    coverage = ["# Control Excitation Coverage", "", "- fast windows: `108000`", "- slow KPI windows: `4320`",
                "- train/validation/test: 每个 split `12 runs / 36000 fast / 1440 slow`",
                "- U levels: `0.3, 0.5, 0.7`; formal input is actuator_applied.effective_value."]
    (output / "CONTROL_EXCITATION_COVERAGE.md").write_text("\n".join(coverage) + "\n")
    lines = ["# Control Identifiability Audit", "", f"- overall pass: `{authority_pass}`", "",
             "| Workload | Load | Arrival | route direction | state d>=0.25 |", "|---|---:|---|---:|---:|"]
    for row in authority:
        lines.append(f"| {row['workload']} | {row['load_fraction']:.2f} | {row['arrival_process']} | {row['route_direction_pass']} | {row['state_effect_pass']} |")
    (output / "CONTROL_IDENTIFIABILITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    files = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    write_json(output / "SHA256SUMS.json", files)
    print(json.dumps(quality, indent=2))
    return 0 if authority_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
