"""Fail-closed validation and sealing for Full-order snapshot runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_DERIVED = (
    "full_state.npy", "disturbance.npy", "output.npy", "next_state.npy",
    "static_config.json", "snapshot_windows.parquet", "snapshot_quality.parquet",
    "request_index.json", "bin_schema.yaml",
)


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshots(run_root: Path) -> dict[str, Any]:
    root = Path(run_root)
    derived = root / "derived"
    violations: list[dict[str, Any]] = []
    for name in REQUIRED_DERIVED:
        if not (derived / name).exists():
            violations.append({"code": "derived_file_missing", "file": name})
    if violations:
        return {"valid": False, "violations": violations}
    import numpy as np
    state = np.load(derived / "full_state.npy")
    disturbance = np.load(derived / "disturbance.npy")
    output = np.load(derived / "output.npy")
    next_state = np.load(derived / "next_state.npy")
    windows = _read_parquet(derived / "snapshot_windows.parquet")
    quality = _read_parquet(derived / "snapshot_quality.parquet")
    if len(windows) != len(state) or len(state) != len(disturbance) or len(state) != len(output):
        violations.append({"code": "snapshot_row_count_mismatch", "windows": len(windows), "state": len(state), "disturbance": len(disturbance), "output": len(output)})
    if len(next_state) != max(len(state) - 1, 0):
        violations.append({"code": "next_state_row_count_mismatch", "next_state": len(next_state), "state": len(state)})
    for expected, row in enumerate(windows):
        if row.get("window_index") != expected:
            violations.append({"code": "window_index_gap", "expected": expected, "actual": row.get("window_index")})
        if row.get("end_wall_ns", 0) <= row.get("start_wall_ns", 0):
            violations.append({"code": "invalid_window_interval", "window_index": expected})
        if expected and row["start_wall_ns"] != windows[expected - 1]["end_wall_ns"]:
            violations.append({"code": "window_coverage_gap", "window_index": expected})
    quality_by_index = {row.get("window_index"): row for row in quality}
    if len(quality_by_index) != len(windows):
        violations.append({"code": "quality_window_cardinality_mismatch"})
    invalid = [row for row in quality if not row.get("valid")]
    # Snapshot Builder is fail-closed: an invalid window makes a run unsealable.
    if invalid:
        violations.append({"code": "invalid_snapshot_windows", "count": len(invalid), "reasons": sorted({row.get("invalid_reason") for row in invalid})})
    return {
        "schema_version": "servingrom.snapshot_quality.v1",
        "valid": not violations,
        "window_count": len(windows),
        "invalid_window_count": len(invalid),
        "violations": violations,
    }


def write_snapshot_quality(run_root: Path, report: dict[str, Any]) -> None:
    reports = Path(run_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "snapshot_data_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Full-order Snapshot 数据质量报告", "", f"- 通过：`{report['valid']}`", f"- 窗口数：{report.get('window_count', 0)}", f"- 无效窗口：{report.get('invalid_window_count', 0)}", "", "## 违规", ""]
    lines.extend((f"- `{row['code']}`：{json.dumps(row, ensure_ascii=False, sort_keys=True)}" for row in report["violations"]) or ["- 无"])
    (reports / "snapshot_data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def seal_run(run_root: Path) -> dict[str, Any]:
    """Seal only a complete, balanced, fully validated immutable run."""
    from servingrom_telemetry.run_metadata import RunLayout, build_sha256_manifest
    root = Path(run_root)
    report = validate_snapshots(root)
    writer_failures: list[str] = []
    for summary in (root / "raw").glob("**/*.summary.json"):
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("events_written") != payload.get("events_enqueued") or payload.get("events_dropped_queue_full", 0) or payload.get("events_dropped_writer_failed", 0):
            writer_failures.append(summary.relative_to(root).as_posix())
    if writer_failures:
        report["valid"] = False
        report.setdefault("violations", []).append({"code": "writer_not_balanced", "files": writer_failures})
    status = {"status": "SEALED" if report["valid"] else "INVALID", "snapshot_quality": report}
    (root / "metadata" / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["valid"]:
        layout = RunLayout(
            root=root,
            experiment_id=root.parent.name,
            run_id=root.name,
        )
        status["sha256_manifest"] = build_sha256_manifest(layout)
        (root / "metadata" / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status
