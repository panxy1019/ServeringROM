#!/usr/bin/env python3
"""Merge SEALED ServingROM runs into the immutable ROM Dataset v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ARRAY_FILES = {
    "X": "full_state.npy",
    "X_next": "next_state.npy",
    "D": "disturbance.npy",
    "Y": "output.npy",
    "MU": "static_config.npy",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist() if path.exists() else []


def run_metrics(run_root: Path, progress: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    traces = read_parquet(run_root / "derived" / "trace_lifecycle.parquet")
    attempts = read_parquet(run_root / "derived" / "attempt_lifecycle.parquet")
    transfers = read_parquet(run_root / "derived" / "kv_transfers.parquet")
    snapshots = run_root / "derived" / "snapshots"
    quality = json.loads((run_root / "reports" / "snapshot_data_quality.json").read_text(encoding="utf-8"))
    workload = json.loads((run_root / "metadata" / "workload_result.json").read_text(encoding="utf-8"))
    static = np.load(snapshots / "static_config.npy")
    ttft = [float(row["ttft_proxy_ns"]) / 1e6 for row in traces if row.get("ttft_proxy_ns") is not None and row.get("terminal_event") == "request_complete"]
    tpot = []
    for row in traces:
        timestamps = [int(event["ts_wall_ns"]) for event in (row.get("token_events") or []) if event.get("ts_wall_ns") is not None]
        if len(timestamps) >= 2:
            tpot.append(sum(b - a for a, b in zip(timestamps, timestamps[1:])) / (len(timestamps) - 1) / 1e6)
    completed = [row for row in traces if row.get("terminal_event") == "request_complete"]
    rejected = [row for row in traces if row.get("terminal_event") == "request_rejected"]
    route_counts = Counter(row.get("decoder_backend") for row in attempts if row.get("decoder_backend"))
    output = np.load(snapshots / "output.npy", mmap_mode="r")
    output_index = {row["name"]: int(row["index"]) for row in json.loads((snapshots / "output_index.json").read_text(encoding="utf-8"))}
    measurement = workload["summary"]
    return {
        "run_id": progress["run_id"], "workload": progress["workload"],
        "arrival_process": progress["arrival_process"], "target_arrival_rate": progress["target_arrival_rate"],
        "actual_arrival_rate": measurement.get("actual_arrival_rate"),
        "lambda_stable": progress["lambda_stable"], "load_fraction": progress.get("load_fraction"),
        "seed": progress["seed"], "split": progress["split"],
        "transient_pattern": progress.get("transient_pattern"),
        "window_count": int(output.shape[0]), "valid_ratio": quality["metrics"]["valid_window_ratio"],
        "request_count": len(traces), "accepted_request_count": sum(bool(row.get("accepted")) for row in traces),
        "completed_request_count": len(completed), "rejected_request_count": len(rejected),
        "prompt_tokens": sum(int(row.get("input_tokens") or 0) for row in traces),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in completed),
        "kv_bytes": sum(int(row.get("actual_total_bytes") or 0) for row in transfers),
        "decode_d1_requests": route_counts.get("0.0.0.0:13701", 0),
        "decode_d2_requests": route_counts.get("0.0.0.0:13702", 0),
        "rejection_rate": len(rejected) / len(traces) if traces else 0.0,
        "ttft_p50_ms": percentile(ttft, 0.50), "ttft_p95_ms": percentile(ttft, 0.95),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_p50_ms": percentile(tpot, 0.50), "tpot_p95_ms": percentile(tpot, 0.95),
        "tpot_p99_ms": percentile(tpot, 0.99),
        "goodput_requests": float(output[:, output_index["goodput_request_count"]].sum()),
        "goodput_output_tokens": float(output[:, output_index["goodput_output_tokens"]].sum()),
        "image_digest": progress["image_digest"], "git_commit": progress["git_commit"],
        "schema_hash": progress["schema_hash"], "seal_status": "SEALED",
        "static_dimension": int(static.shape[0]),
    }


def quantity_policy(index_rows: list[dict[str, Any]]) -> list[str]:
    policies = []
    for row in index_rows:
        unit = str(row.get("unit") or "")
        quantity = str(row.get("quantity") or "")
        if unit in {"requests", "tokens", "bytes", "blocks", "ms"} or any(word in quantity for word in ("count", "mass", "tokens", "bytes", "blocks")):
            policies.append("log1p_standard")
        elif unit in {"ratio", "fraction"} or "ratio" in quantity or "progress" in quantity:
            policies.append("identity")
        else:
            policies.append("standard")
    return policies


def fit_normalization(array, policies: list[str]) -> dict[str, Any]:
    import numpy as np
    if array.shape[1] != len(policies):
        raise ValueError("normalization index does not match matrix dimension")
    means = np.zeros(array.shape[1], dtype=np.float64)
    scales = np.ones(array.shape[1], dtype=np.float64)
    for index, policy in enumerate(policies):
        values = np.asarray(array[:, index], dtype=np.float64)
        if policy == "log1p_standard":
            values = np.log1p(values)
        if policy != "identity":
            means[index] = float(values.mean())
            standard = float(values.std())
            scales[index] = standard if standard > 1e-12 else 1.0
    return {"policy": policies, "mean": means.tolist(), "scale": scales.tolist()}


def apply_normalization(array, spec: dict[str, Any]):
    import numpy as np
    output = np.asarray(array, dtype=np.float32).copy()
    for index, policy in enumerate(spec["policy"]):
        if policy == "log1p_standard":
            output[:, index] = np.log1p(output[:, index])
        if policy != "identity":
            output[:, index] = (output[:, index] - spec["mean"][index]) / spec["scale"][index]
    return output


def save_split(output_root: Path, split: str, arrays: dict[str, Any], normalization: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    target = output_root / split
    target.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, array in arrays.items():
        raw = np.asarray(array, dtype=np.float32)
        np.save(target / f"{name}.npy", raw)
        np.save(target / f"{name}_normalized.npy", apply_normalization(raw, normalization[name]))
        summary[name] = list(raw.shape)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--capacity-summary", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    sealed = [row for row in progress["runs"] if row.get("status") == "SEALED"]
    if len(sealed) != progress["planned_runs"]:
        raise RuntimeError(f"dataset build requires all runs SEALED: {len(sealed)}/{progress['planned_runs']}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_rows = []
    arrays_by_split: dict[str, dict[str, list[Any]]] = {}
    canonical_hashes = None
    for row in sealed:
        run_root = args.staging_root / "runs" / row["run_id"]
        snapshots = run_root / "derived" / "snapshots"
        hashes = {
            "state_index": sha256(snapshots / "state_index.json"),
            "bin_schema": sha256(snapshots / "bin_schema.yaml"),
        }
        canonical_hashes = canonical_hashes or hashes
        if hashes != canonical_hashes:
            raise RuntimeError(f"schema drift in {row['run_id']}: {hashes}")
        split = "test/transient" if row.get("transient_pattern") else row["split"]
        target = arrays_by_split.setdefault(split, {name: [] for name in ARRAY_FILES})
        window_count = None
        for name, filename in ARRAY_FILES.items():
            value = np.load(snapshots / filename, mmap_mode="r")
            if name == "MU":
                assert window_count is not None
                value = np.repeat(np.asarray(value)[None, :], window_count, axis=0)
            else:
                window_count = int(value.shape[0]) if window_count is None else window_count
            target[name].append(np.asarray(value, dtype=np.float32))
        run_rows.append(run_metrics(run_root, row))
    merged: dict[str, dict[str, Any]] = {
        split: {name: np.concatenate(parts, axis=0) for name, parts in values.items()}
        for split, values in arrays_by_split.items()
    }
    state_index_rows = json.loads((args.frozen_dir / "state_index.json").read_text(encoding="utf-8"))
    first_run = args.staging_root / "runs" / sealed[0]["run_id"] / "derived" / "snapshots"
    disturbance_rows = json.loads((first_run / "disturbance_index.json").read_text(encoding="utf-8"))
    output_rows = json.loads((first_run / "output_index.json").read_text(encoding="utf-8"))
    static_json = json.loads((first_run / "static_config.json").read_text(encoding="utf-8"))
    static_rows = [{"name": name, "unit": "scalar", "quantity": name} for name in static_json["numeric_index"]]
    train = merged["train"]
    normalization = {
        "schema_version": "servingrom.normalization.v1",
        "fit_split": "train",
        "X": fit_normalization(train["X"], quantity_policy(state_index_rows)),
        "X_next": fit_normalization(train["X_next"], quantity_policy(state_index_rows)),
        "D": fit_normalization(train["D"], quantity_policy(disturbance_rows)),
        "Y": fit_normalization(train["Y"], quantity_policy(output_rows)),
        "MU": fit_normalization(train["MU"], quantity_policy(static_rows)),
    }
    split_shapes = {split: save_split(args.output_root, split, arrays, normalization) for split, arrays in merged.items()}
    atomic_json(args.output_root / "normalization.json", normalization)
    shutil.copy2(args.frozen_dir / "state_index.json", args.output_root / "state_index.json")
    shutil.copy2(args.frozen_dir / "bin_schema.yaml", args.output_root / "bin_schema.yaml")
    shutil.copy2(args.capacity_summary, args.output_root / "capacity_summary.json")
    shutil.copy2(args.progress, args.output_root / "collection_progress.json")
    schema = {
        "dataset_id": progress["dataset_id"], "version": "1.0",
        "snapshot_period_ms": 200, "state_dimension": 1804,
        "disturbance_dimension": 31, "output_dimension": 19,
        "window_semantics": "[t_k,t_k+1)", "run_level_split": True,
        "canonical_hashes": canonical_hashes,
    }
    atomic_json(args.output_root / "snapshot_schema.json", schema)
    pq.write_table(pa.Table.from_pylist(run_rows), args.output_root / "run_manifest.parquet", compression="zstd")
    nonzero = np.count_nonzero(train["X"], axis=0) / max(train["X"].shape[0], 1)
    block_stats = {}
    for row in state_index_rows:
        block = row["block"]
        block_stats.setdefault(block, []).append(float(nonzero[int(row["index"])]))
    quality = {
        "planned_runs": progress["planned_runs"], "sealed_runs": len(sealed),
        "split_shapes": split_shapes,
        "state_nonzero_ratio": float(np.count_nonzero(train["X"]) / train["X"].size),
        "long_term_zero_dimensions": [int(index) for index in np.where(nonzero == 0)[0]],
        "block_nonzero_ratio": {name: float(statistics.mean(values)) for name, values in block_stats.items()},
    }
    atomic_json(args.output_root / "quality_summary.json", quality)
    lines = [
        "# ServingROM ROM Dataset v1 质量摘要", "",
        f"- SEALED runs：{len(sealed)}/{progress['planned_runs']}",
        f"- Train state nonzero ratio：{quality['state_nonzero_ratio']:.6f}",
        f"- 长期恒零维度：{len(quality['long_term_zero_dimensions'])}", "",
        "## Split Shapes", "",
    ]
    lines.extend(f"- `{split}`：`{json.dumps(shapes, ensure_ascii=False)}`" for split, shapes in split_shapes.items())
    (args.output_root / "quality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    capacity = json.loads(args.capacity_summary.read_text(encoding="utf-8"))
    capacity_lines = ["# ServingROM 容量标定摘要", ""]
    capacity_lines.extend(
        f"- `{name}`：`lambda_stable={rate}` requests/s"
        for name, rate in sorted(capacity["lambda_stable"].items())
    )
    (args.output_root / "capacity_summary.md").write_text("\n".join(capacity_lines) + "\n", encoding="utf-8")
    split_windows = {split: int(values["X"][0]) for split, values in split_shapes.items()}
    split_requests = Counter()
    for row in run_rows:
        split_name = "test/transient" if row.get("transient_pattern") else row["split"]
        split_requests[split_name] += int(row["request_count"])
    report_lines = [
        "# ServingROM ROM Dataset v1 正式采集报告", "",
        f"- 计划/SEALED：`{progress['planned_runs']}/{len(sealed)}`",
        f"- INVALID：`{sum(row.get('status') == 'INVALID' for row in progress['runs'])}`",
        f"- 数据集：`{progress['dataset_id']}`", "", "## 数据划分", "",
    ]
    report_lines.extend(
        f"- `{split}`：窗口 `{windows}`，请求 `{split_requests[split]}`"
        for split, windows in split_windows.items()
    )
    report_lines += ["", "## 矩阵", ""]
    report_lines.extend(f"- `{split}`：`{json.dumps(shapes, ensure_ascii=False)}`" for split, shapes in split_shapes.items())
    report_lines += ["", "## 状态覆盖", "", f"- 全状态非零率：`{quality['state_nonzero_ratio']:.6f}`", f"- 长期恒零维度：`{len(quality['long_term_zero_dimensions'])}`", ""]
    report_lines.extend(f"- `{name}`：`{ratio:.6f}`" for name, ratio in sorted(quality["block_nonzero_ratio"].items()))
    (args.output_root / "ROM_DATA_COLLECTION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (args.output_root / "invalid_runs").mkdir(exist_ok=True)
    manifest_path = args.output_root / "dataset_manifest.json"
    files = []
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path != manifest_path and not path.name.endswith(".tmp"):
            files.append({"path": path.relative_to(args.output_root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"dataset_id": progress["dataset_id"], "schema_version": "1.0", "file_count": len(files), "files": files}
    atomic_json(manifest_path, manifest)
    print(json.dumps({"runs": len(sealed), "split_shapes": split_shapes, "manifest_sha256": sha256(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
