#!/usr/bin/env python3
"""Build immutable Dataset v1.1 by replaying sealed runs with one SLO policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from servingrom_pipeline.snapshot_builder import SnapshotConfig, build_snapshots


ARRAYS = {"X": "full_state.npy", "X_next": "next_state.npy", "D": "disturbance.npy", "Y": "output.npy"}
SPLITS = ("train", "validation", "test", "test/transient")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def split_name(row: dict[str, Any]) -> str:
    return "test/transient" if row.get("transient_pattern") else row["split"]


def validate_run(snapshot_dir: Path, expected_windows: int, state_index_hash: str) -> None:
    state = np.load(snapshot_dir / "full_state.npy", mmap_mode="r")
    next_state = np.load(snapshot_dir / "next_state.npy", mmap_mode="r")
    disturbance = np.load(snapshot_dir / "disturbance.npy", mmap_mode="r")
    output = np.load(snapshot_dir / "output.npy", mmap_mode="r")
    if not (len(state) == len(next_state) == len(disturbance) == len(output) == expected_windows):
        raise ValueError(f"canonical row mismatch in {snapshot_dir}")
    for name, array in (("X", state), ("X_next", next_state), ("D", disturbance), ("Y", output)):
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"invalid canonical numeric values: {name}")
    if not np.array_equal(state[1:], next_state[:-1]):
        raise ValueError("canonical next-state shift mismatch")
    if sha256(snapshot_dir / "state_index.json") != state_index_hash:
        raise ValueError("canonical state index drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=100.0)
    args = parser.parse_args()

    source_manifest_before = sha256(args.source_dataset / "dataset_manifest.json")
    source_rows = read_parquet(args.source_dataset / "run_manifest.parquet")
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    planned = [row for row in progress["runs"] if row.get("status") == "SEALED"]
    if len(planned) != 84:
        raise RuntimeError(f"canonicalization requires 84 SEALED runs, got {len(planned)}")
    by_id = {row["run_id"]: dict(row) for row in source_rows}
    if set(by_id) != {row["run_id"] for row in planned}:
        raise RuntimeError("progress and source run manifest disagree")
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_index_hash = sha256(args.source_dataset / "state_index.json")
    split_sizes = {split: int(np.load(args.source_dataset / split / "X.npy", mmap_mode="r").shape[0]) for split in SPLITS}
    dimensions = {"X": 1804, "X_next": 1804, "D": 31, "Y": 19, "MU": 12}
    outputs: dict[str, dict[str, np.memmap]] = {}
    for split in SPLITS:
        directory = args.output_root / split
        directory.mkdir(parents=True, exist_ok=True)
        outputs[split] = {
            name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=np.float32, shape=(split_sizes[split], dimension))
            for name, dimension in dimensions.items()
        }
    offsets = {split: 0 for split in SPLITS}
    reports = []
    canonical_rows = []
    config = SnapshotConfig(
        period_ms=200,
        force_ttft_slo_ms=args.ttft_slo_ms,
        force_tpot_slo_ms=args.tpot_slo_ms,
    )
    for position, plan in enumerate(planned, 1):
        run_id = plan["run_id"]
        source_row = by_id[run_id]
        split = split_name(source_row)
        run_root = args.runs_root / run_id
        temporary = run_root / "derived" / "snapshots_slo_canonical_tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        result = build_snapshots(run_root, config, snapshot_dir_override=temporary)
        windows = int(source_row["window_count"])
        validate_run(temporary, windows, state_index_hash)
        start, end = offsets[split], offsets[split] + windows
        for name, filename in ARRAYS.items():
            outputs[split][name][start:end] = np.asarray(np.load(temporary / filename, mmap_mode="r"), dtype=np.float32)
        static = np.asarray(np.load(temporary / "static_config.npy"), dtype=np.float32)
        if static.shape != (12,) or static[10] != args.ttft_slo_ms or static[11] != args.tpot_slo_ms:
            raise ValueError(f"canonical MU mismatch in {run_id}: {static}")
        outputs[split]["MU"][start:end] = static
        output_index = {row["name"]: int(row["index"]) for row in json.loads((temporary / "output_index.json").read_text(encoding="utf-8"))}
        canonical_y = np.load(temporary / "output.npy", mmap_mode="r")
        source_y = np.load(args.source_dataset / split / "Y.npy", mmap_mode="r")[start:end]
        report = {
            "run_id": run_id, "split": split, "workload": source_row["workload"],
            "source_ttft_slo_ms": float(np.load(args.source_dataset / split / "MU.npy", mmap_mode="r")[start, 10]),
            "canonical_ttft_slo_ms": args.ttft_slo_ms,
            "source_goodput_requests": float(source_y[:, output_index["goodput_request_count"]].sum()),
            "canonical_goodput_requests": float(canonical_y[:, output_index["goodput_request_count"]].sum()),
            "source_goodput_tokens": float(source_y[:, output_index["goodput_output_tokens"]].sum()),
            "canonical_goodput_tokens": float(canonical_y[:, output_index["goodput_output_tokens"]].sum()),
            "window_count": windows, "snapshot_manifest_sha256": sha256(temporary / "snapshot_manifest.json"),
            "valid_window_count": int(result["valid_window_count"]),
        }
        reports.append(report)
        canonical_row = dict(source_row)
        canonical_row["goodput_requests"] = report["canonical_goodput_requests"]
        canonical_row["goodput_output_tokens"] = report["canonical_goodput_tokens"]
        canonical_rows.append(canonical_row)
        offsets[split] = end
        shutil.rmtree(temporary)
        print(f"CANONICALIZED {position}/84 run_id={run_id} split={split}", flush=True)
    for split in SPLITS:
        if offsets[split] != split_sizes[split]:
            raise RuntimeError(f"split coverage mismatch: {split} {offsets[split]}/{split_sizes[split]}")
        for array in outputs[split].values():
            array.flush()
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(canonical_rows), args.output_root / "run_manifest.parquet", compression="zstd")
    for name in ("state_index.json", "bin_schema.yaml", "collection_progress.json", "capacity_summary.json"):
        shutil.copy2(args.source_dataset / name, args.output_root / name)
    first_run = args.runs_root / planned[0]["run_id"] / "derived" / "snapshots"
    for name in ("disturbance_index.json", "output_index.json", "static_config.json"):
        shutil.copy2(first_run / name, args.output_root / name)
    schema = {
        "dataset_id": "servingrom-qwen36-1p2d-d2-rom-v1.1-slo2000",
        "version": "1.1", "parent_dataset": "servingrom-qwen36-1p2d-d2-rom-v1",
        "derivation": "sealed telemetry replay with canonical SLO",
        "canonical_ttft_slo_ms": args.ttft_slo_ms, "canonical_tpot_slo_ms": args.tpot_slo_ms,
        "snapshot_period_ms": 200, "state_dimension": 1804,
        "disturbance_dimension": 31, "output_dimension": 19, "static_dimension": 12,
        "run_level_split": True, "window_semantics": "[t_k,t_k+1)",
        "source_dataset_manifest_sha256": source_manifest_before,
    }
    atomic_json(args.output_root / "snapshot_schema.json", schema)
    atomic_json(args.output_root / "canonicalization_report.json", {
        "schema_version": "servingrom.slo_canonicalization.v1", "runs": reports,
        "source_dataset_manifest_before": source_manifest_before,
        "source_dataset_manifest_after": sha256(args.source_dataset / "dataset_manifest.json"),
    })
    if source_manifest_before != sha256(args.source_dataset / "dataset_manifest.json"):
        raise RuntimeError("Dataset v1 changed during canonicalization")
    manifest_path = args.output_root / "dataset_manifest.json"
    files = []
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path != manifest_path:
            files.append({"path": path.relative_to(args.output_root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic_json(manifest_path, {"dataset_id": schema["dataset_id"], "schema_version": "1.1", "file_count": len(files), "files": files})
    lines = [
        "# Dataset v1.1 统一 SLO 派生报告", "",
        f"- Parent manifest：`{source_manifest_before}`",
        f"- Runs：`{len(reports)}/84`", f"- TTFT SLO：`{args.ttft_slo_ms} ms`",
        f"- TPOT SLO：`{args.tpot_slo_ms} ms`",
        f"- Source manifest 前后一致：`True`",
        f"- 输出：`{args.output_root}`", "",
        "该过程只重放封存 telemetry，不发送推理请求，不修改 Dataset v1。",
    ]
    (args.output_root / "DERIVED_DATASET_V1_1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DATASET_V1_1_SEALED", "runs": len(reports), "output": str(args.output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

