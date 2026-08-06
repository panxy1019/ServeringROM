"""Fail-closed conservation checks and sealing for ServingROM snapshots."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_SNAPSHOT_FILES = (
    "full_state.npy", "disturbance.npy", "output.npy", "static_config.npy",
    "next_state.npy", "window_table.parquet", "snapshot_quality.parquet",
    "request_state_inventory.parquet", "state_index.json",
    "disturbance_index.json", "output_index.json", "static_config.json",
    "bin_schema.yaml", "snapshot_manifest.json",
)
STATE_ORDER = {
    "ABSENT": 0, "ADMITTED": 1, "PREFILL_WAITING": 2, "PREFILL_RUNNING": 3,
    "HANDOFF_WAITING": 4, "KV_QUEUED": 5, "KV_TRANSFERRING": 6,
    "KV_READY": 7, "DECODE_WAITING": 8, "DECODE_RUNNING": 9,
    "COMPLETED": 10, "CANCELLED": 10, "FAILED": 10, "REJECTED": 10,
}


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(path: Path) -> dict[str, int]:
    return {row["name"]: int(row["index"]) for row in json.loads(path.read_text(encoding="utf-8"))}


def _raw_jsonl_quality(root: Path) -> tuple[int, list[dict[str, Any]]]:
    damaged = 0
    by_process: dict[str, list[int]] = defaultdict(list)
    for path in (root / "raw").glob("**/*.jsonl"):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                    if event.get("process_instance_id") and event.get("event_seq") is not None:
                        by_process[event["process_instance_id"]].append(int(event["event_seq"]))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    damaged += 1
    gaps = []
    for process, sequence in by_process.items():
        unique = sorted(set(sequence))
        expected = list(range(1, unique[-1] + 1)) if unique else []
        if unique != expected or len(sequence) != len(unique):
            gaps.append({"process_instance_id": process, "events": len(sequence), "unique": len(unique), "expected": len(expected)})
    return damaged, gaps


def validate_snapshots(run_root: Path) -> dict[str, Any]:
    import numpy as np
    root = Path(run_root)
    snapshot_dir = root / "derived" / "snapshots"
    violations: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    def fail(code: str, **detail: Any) -> None:
        violations.append({"code": code, **detail})

    for name in REQUIRED_SNAPSHOT_FILES:
        if not (snapshot_dir / name).exists():
            fail("derived_file_missing", file=name)
    if violations:
        return {"schema_version": "servingrom.snapshot_quality.v2", "valid": False, "eligible_for_training": False, "metrics": metrics, "violations": violations}

    state = np.load(snapshot_dir / "full_state.npy")
    next_state = np.load(snapshot_dir / "next_state.npy")
    disturbance = np.load(snapshot_dir / "disturbance.npy")
    output = np.load(snapshot_dir / "output.npy")
    static = np.load(snapshot_dir / "static_config.npy")
    windows = _read_parquet(snapshot_dir / "window_table.parquet")
    quality = _read_parquet(snapshot_dir / "snapshot_quality.parquet")
    inventory = _read_parquet(snapshot_dir / "request_state_inventory.parquet")
    state_index = _index(snapshot_dir / "state_index.json")
    disturbance_index = _index(snapshot_dir / "disturbance_index.json")
    output_index = _index(snapshot_dir / "output_index.json")
    metrics.update({
        "window_count": len(windows), "state_dimensions": state.shape[1] if state.ndim == 2 else 0,
        "disturbance_dimensions": disturbance.shape[1] if disturbance.ndim == 2 else 0,
        "output_dimensions": output.shape[1] if output.ndim == 2 else 0,
    })
    measurement_path = root / "metadata" / "measurement.json"
    if measurement_path.exists():
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        expected_windows = (
            int(measurement["measurement_end_wall_ns"])
            - int(measurement["measurement_start_wall_ns"])
        ) // (int(measurement["snapshot_period_ms"]) * 1_000_000)
        metrics["expected_measurement_windows"] = expected_windows
        if len(windows) != expected_windows:
            fail("measurement_window_count_mismatch", expected=expected_windows, actual=len(windows))
        if any(row.get("segment") != "measurement" for row in windows):
            fail("non_measurement_window_in_training_snapshot")
    if not (len(windows) == len(state) == len(next_state) == len(disturbance) == len(output) == len(quality)):
        fail("snapshot_row_count_mismatch", windows=len(windows), state=len(state), next_state=len(next_state), disturbance=len(disturbance), output=len(output), quality=len(quality))
    for expected, row in enumerate(windows):
        if int(row.get("window_id", -1)) != expected:
            fail("window_index_gap", expected=expected, actual=row.get("window_id"))
        if int(row.get("end_wall_ns", 0)) - int(row.get("start_wall_ns", 0)) != int(row.get("duration_ms", 0)) * 1_000_000:
            fail("window_period_mismatch", window_id=expected)
        if expected and int(row["start_wall_ns"]) != int(windows[expected - 1]["end_wall_ns"]):
            fail("window_coverage_gap", window_id=expected)
    invalid = [row for row in quality if not bool(row.get("valid"))]
    ratio = (len(quality) - len(invalid)) / len(quality) if quality else 0.0
    metrics["valid_window_count"] = len(quality) - len(invalid)
    metrics["invalid_window_count"] = len(invalid)
    metrics["valid_window_ratio"] = ratio
    metrics["snapshot_window_coverage"] = 1.0 if windows else 0.0
    if ratio <= 0.99:
        fail("valid_window_ratio_below_gate", ratio=ratio, invalid=len(invalid))

    for name, array in (("full_state", state), ("next_state", next_state), ("disturbance", disturbance), ("output", output), ("static_config", static)):
        if np.isnan(array).any():
            fail("numeric_nan", array=name, count=int(np.isnan(array).sum()))
        if np.isinf(array).any():
            fail("numeric_inf", array=name, count=int(np.isinf(array).sum()))
        if (array < 0).any():
            fail("numeric_negative", array=name, count=int((array < 0).sum()))
    if len(state) and not np.array_equal(state[1:], next_state[:-1]):
        fail("next_state_shift_mismatch")

    active_i = state_index["active_attempt_count"]
    accepted_i = disturbance_index["accepted_arrival_count"]
    complete_i = output_index["completed_request_count"]
    cancel_i = output_index["request_cancel_count"]
    error_i = output_index["request_error_count"]
    inventory_failures = []
    for index in range(len(windows)):
        expected = state[index, active_i] + disturbance[index, accepted_i] - output[index, complete_i] - output[index, cancel_i] - output[index, error_i]
        actual = next_state[index, active_i]
        if actual != expected:
            inventory_failures.append({"window_id": index, "expected": float(expected), "actual": float(actual)})
    metrics["request_inventory_conservation_ratio"] = 1.0 - len(inventory_failures) / len(windows) if windows else 0.0
    if inventory_failures:
        fail("request_inventory_conservation", count=len(inventory_failures), examples=inventory_failures[:10])

    phase_names = (
        "admitted_count", "prefill_waiting_count", "prefill_running_count",
        "handoff_waiting_count", "kv_queue_count", "kv_transfer_inflight_count",
        "kv_ready_count", "decode_d1_waiting_count", "decode_d1_running_count",
        "decode_d2_waiting_count", "decode_d2_running_count",
    )
    phase_sum = sum(state[:, state_index[name]] for name in phase_names)
    next_phase_sum = sum(next_state[:, state_index[name]] for name in phase_names)
    phase_bad = int((phase_sum != state[:, active_i]).sum() + (next_phase_sum != next_state[:, active_i]).sum())
    metrics["stage_inventory_conservation_ratio"] = 1.0 - phase_bad / max(2 * len(windows), 1)
    if phase_bad:
        fail("stage_inventory_conservation", count=phase_bad)
    transition_bad = []
    for row in inventory:
        before, after = row.get("state_start"), row.get("state_end")
        if before not in STATE_ORDER or after not in STATE_ORDER or STATE_ORDER[after] < STATE_ORDER[before]:
            transition_bad.append({"window_id": row.get("window_id"), "request_id": row.get("request_id"), "before": before, "after": after})
    if transition_bad:
        fail("request_state_transition_invalid", count=len(transition_bad), examples=transition_bad[:10])

    transfers = _read_parquet(root / "derived" / "kv_transfers.parquet")
    attempts = {row.get("request_id"): row for row in _read_parquet(root / "derived" / "attempt_lifecycle.parquet") if row.get("request_id")}
    kv_bad = []
    for transfer in transfers:
        enqueue, start, ready = transfer.get("enqueue_wall_ns"), transfer.get("first_start_wall_ns"), transfer.get("kv_ready_wall_ns")
        attempt = attempts.get(transfer.get("request_id"), {})
        first_decode = attempt.get("decode_first_byte_wall_ns")
        reasons = []
        if None in (enqueue, start, ready) or not (int(enqueue) <= int(start) <= int(ready)):
            reasons.append("time_order")
        if first_decode is not None and ready is not None and int(ready) > int(first_decode):
            reasons.append("ready_after_decode_first_byte")
        if int(transfer.get("completed_rank_count") or 0) != int(transfer.get("expected_rank_count") or 0):
            reasons.append("rank_count")
        if json.loads(transfer.get("missing_ranks_json") or "[]"):
            reasons.append("missing_ranks")
        if int(transfer.get("actual_total_bytes") or 0) <= 0:
            reasons.append("bytes")
        if not bool(transfer.get("success")):
            reasons.append("failed")
        if reasons:
            kv_bad.append({"request_id": transfer.get("request_id"), "reasons": reasons})
    metrics["kv_transfer_count"] = len(transfers)
    metrics["kv_lifecycle_violation_count"] = len(kv_bad)
    if kv_bad:
        fail("kv_lifecycle_violation", count=len(kv_bad), examples=kv_bad[:10])

    internal_report_path = root / "reports" / "internal_data_quality.json"
    proxy_report_path = root / "reports" / "proxy_lifecycle_quality.json"
    for name, path in (("internal", internal_report_path), ("proxy", proxy_report_path)):
        if not path.exists():
            fail("quality_report_missing", report=name)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics[f"{name}_violation_count"] = int(report.get("violation_count") or report.get("metrics", {}).get("violation_count") or 0)
        if metrics[f"{name}_violation_count"]:
            fail("upstream_quality_violation", report=name, count=metrics[f"{name}_violation_count"])

    writer_failures = []
    summary_count = 0
    for path in (root / "raw").glob("**/*.summary.json"):
        summary_count += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("events_written") != payload.get("events_enqueued") or any(int(payload.get(field, 0)) for field in ("events_dropped_queue_full", "events_dropped_writer_failed", "serialization_errors", "write_errors", "flush_errors")):
            writer_failures.append(path.relative_to(root).as_posix())
    metrics["writer_summary_count"] = summary_count
    metrics["writer_failure_count"] = len(writer_failures)
    if writer_failures:
        fail("writer_not_balanced", files=writer_failures)
    damaged, event_seq_gaps = _raw_jsonl_quality(root)
    metrics["jsonl_damaged_lines"] = damaged
    metrics["event_seq_gap_processes"] = len(event_seq_gaps)
    if damaged:
        fail("jsonl_damaged", count=damaged)
    if event_seq_gaps:
        fail("event_seq_gap", processes=event_seq_gaps)

    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_bad = []
    for relative, expected in {**manifest.get("inputs", {}), **manifest.get("outputs", {})}.items():
        path = root / relative
        if not path.exists() or _sha256(path) != expected:
            manifest_bad.append(relative)
    metrics["snapshot_manifest_mismatch_count"] = len(manifest_bad)
    if manifest_bad:
        fail("snapshot_manifest_mismatch", files=manifest_bad)
    frozen_path = root / "metadata" / "frozen_schema.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        actual_frozen = {
            "state_index_sha256": _sha256(snapshot_dir / "state_index.json"),
            "bin_schema_sha256": _sha256(snapshot_dir / "bin_schema.yaml"),
        }
        frozen_bad = {
            key: {"expected": frozen.get(key), "actual": value}
            for key, value in actual_frozen.items()
            if frozen.get(key) != value
        }
        metrics["frozen_schema_mismatch_count"] = len(frozen_bad)
        if frozen_bad:
            fail("frozen_schema_mismatch", fields=frozen_bad)
    return {
        "schema_version": "servingrom.snapshot_quality.v2",
        "valid": not violations,
        "eligible_for_training": False,
        "metrics": metrics,
        "violations": violations,
    }


def write_snapshot_quality(run_root: Path, report: dict[str, Any]) -> None:
    reports = Path(run_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "snapshot_data_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Full-order Snapshot 数据质量报告", "",
        f"- 验证通过：`{report['valid']}`",
        f"- 窗口数：{report.get('metrics', {}).get('window_count', 0)}",
        f"- 有效窗口比例：{report.get('metrics', {}).get('valid_window_ratio', 0):.6f}",
        f"- 请求库存守恒率：{report.get('metrics', {}).get('request_inventory_conservation_ratio', 0):.6f}",
        f"- 阶段库存守恒率：{report.get('metrics', {}).get('stage_inventory_conservation_ratio', 0):.6f}",
        "", "## 违规", "",
    ]
    lines.extend([f"- `{row['code']}`：{json.dumps(row, ensure_ascii=False, sort_keys=True)}" for row in report["violations"]] or ["- 无"])
    (reports / "snapshot_data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def seal_run(run_root: Path) -> dict[str, Any]:
    from servingrom_telemetry.run_metadata import RunLayout, build_sha256_manifest
    root = Path(run_root)
    report = validate_snapshots(root)
    status = {
        "status": "SEALED" if report["valid"] else "INVALID",
        "eligible_for_training": bool(report["valid"]),
        "reasons": report["violations"],
        "snapshot_quality": report,
        "sha256_manifest_path": "metadata/sha256_manifest.json" if report["valid"] else None,
    }
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["valid"]:
        layout = RunLayout(root=root, experiment_id=root.parent.name, run_id=root.name)
        manifest = build_sha256_manifest(layout)
        mismatches = [
            item["path"]
            for item in manifest["files"]
            if _sha256(root / item["path"]) != item["sha256"]
        ]
        if mismatches:
            status.update({
                "status": "INVALID", "eligible_for_training": False,
                "reasons": [{"code": "sha256_manifest_mismatch", "files": mismatches}],
            })
            (metadata / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return status
        return {**status, "sha256_manifest": manifest}
    return status
