"""Deterministic, event-replay based Full-order snapshot construction.

This module intentionally contains no control policy.  It replays immutable
Proxy/engine/Mooncake facts into fixed wall-clock windows and makes every
missing observation explicit in the accompanying quality table.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


STATE_ORDER = (
    "ARRIVED", "ADMITTED", "PREFILL_WAITING", "PREFILL_RUNNING",
    "HANDOFF_WAITING", "KV_QUEUED", "KV_TRANSFERRING", "KV_READY",
    "DECODE_WAITING", "DECODE_RUNNING", "COMPLETED", "CANCELLED",
    "FAILED", "REJECTED",
)
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "REJECTED"}
SNAPSHOT_SCHEMA_VERSION = "servingrom.full_order_snapshot.v1"


@dataclass(frozen=True)
class SnapshotConfig:
    period_ms: int = 200
    input_histogram_max_tokens: int = 32768
    input_histogram_bin_tokens: int = 256
    context_histogram_max_tokens: int = 32768
    context_histogram_bin_tokens: int = 256

    @property
    def period_ns(self) -> int:
        return self.period_ms * 1_000_000


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows) if rows else pa.table({"window_index": pa.array([], type=pa.int64())})
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _float(value: Any) -> float | None:
    return None if value is None else float(value)


def _histogram(values: Iterable[int | float | None], *, maximum: int, width: int) -> list[int]:
    bins = [0] * (math.ceil(maximum / width) + 1)
    for value in values:
        if value is None:
            continue
        index = min(max(int(value), 0) // width, len(bins) - 1)
        bins[index] += 1
    return bins


def _event_time(row: dict[str, Any]) -> int | None:
    for key in ("ts_wall_ns", "last_complete_wall_ns", "first_start_wall_ns", "added_wall_ns", "terminal_wall_ns"):
        if row.get(key) is not None:
            return int(row[key])
    return None


def _parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, list) else []
    except (TypeError, ValueError):
        return []


def _request_rows(run_root: Path) -> list[dict[str, Any]]:
    """Unify Proxy attempts and engine rows; engine facts remain authoritative."""
    attempts = _read_parquet(run_root / "derived" / "attempt_lifecycle.parquet")
    traces = _read_parquet(run_root / "derived" / "trace_lifecycle.parquet")
    engines = _read_parquet(run_root / "derived" / "engine_requests.parquet")
    transfers = _read_parquet(run_root / "derived" / "kv_transfers.parquet")
    by_id: dict[str, dict[str, Any]] = {}
    traces_by_id = {row.get("trace_id"): row for row in traces if row.get("trace_id")}
    for attempt in attempts:
        request_id = attempt.get("request_id")
        if request_id:
            trace = traces_by_id.get(attempt.get("trace_id"), {})
            by_id[request_id] = {
                "request_id": request_id, **trace, **attempt,
                "admission_accepted": trace.get("accepted"),
                "arrival_wall_ns": trace.get("arrival_wall_ns"),
                "terminal_wall_ns": trace.get("terminal_wall_ns"),
                "terminal_event": trace.get("terminal_event"),
                "input_tokens": trace.get("input_tokens"),
            }
    for engine in engines:
        request_id = engine.get("request_id")
        if not request_id:
            continue
        row = by_id.setdefault(request_id, {"request_id": request_id})
        component = engine.get("component")
        if component == "prefill":
            row.update({
                "prefill_added_wall_ns": engine.get("added_wall_ns"),
                "prefill_terminal_wall_ns": engine.get("terminal_wall_ns"),
                "input_tokens": engine.get("prompt_tokens", row.get("input_tokens")),
            })
        elif component in {"decode-0", "decode-1"}:
            row.update({
                "decode_component": component,
                "decode_added_wall_ns": engine.get("added_wall_ns"),
                "decode_terminal_wall_ns": engine.get("terminal_wall_ns"),
            })
    for transfer in transfers:
        request_id = transfer.get("request_id")
        if request_id and request_id in by_id:
            by_id[request_id].update({
                "kv_enqueue_wall_ns": transfer.get("enqueue_wall_ns"),
                "kv_first_start_wall_ns": transfer.get("first_start_wall_ns"),
                "kv_ready_wall_ns": transfer.get("kv_ready_wall_ns") or transfer.get("last_complete_wall_ns"),
                "kv_success": transfer.get("success"),
                "kv_missing_ranks": _parse_json_list(transfer.get("missing_ranks_json")),
            })
    return list(by_id.values())


def _state_at(row: dict[str, Any], t: int) -> str:
    arrival = row.get("arrival_wall_ns")
    if arrival is None or t < arrival:
        return "ABSENT"
    status = str(row.get("terminal_event") or "")
    terminal = row.get("terminal_wall_ns") or row.get("decode_terminal_wall_ns")
    if status == "request_rejected" and terminal is not None and t >= terminal:
        return "REJECTED"
    if terminal is not None and t >= terminal:
        return "CANCELLED" if "cancel" in status else "FAILED" if "error" in status else "COMPLETED"
    submit = row.get("prefill_submit_wall_ns")
    if submit is None or t < submit:
        return "ADMITTED"
    if row.get("prefill_added_wall_ns") is None or t < row["prefill_added_wall_ns"]:
        return "PREFILL_WAITING"
    if row.get("prefill_terminal_wall_ns") is None or t < row["prefill_terminal_wall_ns"]:
        return "PREFILL_RUNNING"
    enqueue = row.get("kv_enqueue_wall_ns")
    start = row.get("kv_first_start_wall_ns")
    ready = row.get("kv_ready_wall_ns")
    if enqueue is None or t < enqueue:
        return "HANDOFF_WAITING"
    if start is None or t < start:
        return "KV_QUEUED"
    if ready is None or t < ready:
        return "KV_TRANSFERRING"
    if row.get("decode_added_wall_ns") is None or t < row["decode_added_wall_ns"]:
        return "DECODE_WAITING"
    return "DECODE_RUNNING"


def _window_bounds(event_times: list[int], period_ns: int) -> tuple[int, int]:
    if not event_times:
        raise ValueError("no wall-clock events available for snapshot construction")
    start = min(event_times) // period_ns * period_ns
    end = ((max(event_times) // period_ns) + 1) * period_ns
    return start, end


def _load_static_config(run_root: Path) -> dict[str, Any]:
    metadata = run_root / "metadata"
    candidates = [metadata / "run.yaml", metadata / "effective_engine_config.json", metadata / "deployment.yaml"]
    values: dict[str, Any] = {"schema_version": SNAPSHOT_SCHEMA_VERSION}
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                values[path.name] = json.loads(text)
            else:
                values[path.name] = text
        except OSError:
            values[path.name] = None
    # Unknown static fields are explicit nulls rather than invented values.
    for field in ("prefill_token_budget", "decode_token_budget", "chunk_size", "max_num_seqs", "kv_block_size", "model_max_len"):
        values.setdefault(field, None)
    return values


def build_snapshots(run_root: Path, config: SnapshotConfig = SnapshotConfig()) -> dict[str, Any]:
    root = Path(run_root)
    requests = _request_rows(root)
    scheduler = _read_parquet(root / "derived" / "scheduler_iterations.parquet")
    membership = _read_parquet(root / "derived" / "scheduler_membership.parquet")
    token_events = _read_parquet(root / "derived" / "token_emissions.parquet")
    transfers = _read_parquet(root / "derived" / "kv_transfers.parquet")
    devices = _read_parquet(root / "derived" / "device_metrics.parquet")
    all_times = [time for row in (requests + scheduler + membership + token_events + transfers + devices) if (time := _event_time(row)) is not None]
    start, end = _window_bounds(all_times, config.period_ns)
    request_by_id = {row["request_id"]: row for row in requests if row.get("request_id")}
    scheduler_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scheduler:
        if row.get("ts_wall_ns") is not None:
            scheduler_by_window[(int(row["ts_wall_ns"]) - start) // config.period_ns].append(row)
    membership_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in membership:
        if row.get("ts_wall_ns") is not None:
            membership_by_window[(int(row["ts_wall_ns"]) - start) // config.period_ns].append(row)
    tokens_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in token_events:
        if row.get("ts_wall_ns") is not None:
            tokens_by_window[(int(row["ts_wall_ns"]) - start) // config.period_ns].append(row)
    device_by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in devices:
        if row.get("ts_wall_ns") is not None:
            device_by_window[(int(row["ts_wall_ns"]) - start) // config.period_ns].append(row)

    windows: list[dict[str, Any]] = []
    state_rows: list[list[float | int | None]] = []
    disturbance_rows: list[list[float | int | None]] = []
    output_rows: list[list[float | int | None]] = []
    quality: list[dict[str, Any]] = []
    request_index = {request_id: index for index, request_id in enumerate(sorted(request_by_id))}
    decode_components = ("decode-0", "decode-1")
    for window_index, t0 in enumerate(range(start, end, config.period_ns)):
        t1 = t0 + config.period_ns
        phases = Counter(_state_at(row, t0) for row in requests)
        scheduled = scheduler_by_window.get(window_index, [])
        members = membership_by_window.get(window_index, [])
        emissions = tokens_by_window.get(window_index, [])
        metrics = device_by_window.get(window_index, [])
        active = [row for row in requests if _state_at(row, t0) not in TERMINAL | {"ABSENT", "REJECTED"}]
        state_vector = [phases.get(name, 0) for name in STATE_ORDER]
        state_vector += [
            len(active),
            sum(1 for row in active if row.get("decode_component") == "decode-0"),
            sum(1 for row in active if row.get("decode_component") == "decode-1"),
            sum(1 for row in active if row.get("kv_ready_wall_ns") is not None),
            sum(1 for row in active if row.get("kv_missing_ranks")),
        ]
        input_hist = _histogram((row.get("input_tokens") for row in active), maximum=config.input_histogram_max_tokens, width=config.input_histogram_bin_tokens)
        context_hist = _histogram((row.get("context_tokens_before") for row in members), maximum=config.context_histogram_max_tokens, width=config.context_histogram_bin_tokens)
        state_rows.append(state_vector + input_hist + context_hist)
        arrivals = sum(1 for row in requests if row.get("arrival_wall_ns") is not None and t0 <= int(row["arrival_wall_ns"]) < t1)
        accepted = sum(1 for row in requests if row.get("admission_accepted") and row.get("arrival_wall_ns") is not None and t0 <= int(row["arrival_wall_ns"]) < t1)
        rejected = sum(1 for row in requests if row.get("terminal_event") == "request_rejected" and row.get("terminal_wall_ns") is not None and t0 <= int(row["terminal_wall_ns"]) < t1)
        scheduled_prefill = sum(int(row.get("scheduled_tokens") or 0) for row in members if row.get("component") == "prefill")
        disturbance_rows.append([arrivals, accepted, rejected, scheduled_prefill, len(scheduled), len(members)])
        output_tokens = sum(int(row.get("new_token_count") or 0) for row in emissions)
        output_rows.append([output_tokens, len(emissions), phases.get("COMPLETED", 0), phases.get("FAILED", 0), phases.get("CANCELLED", 0)])
        expected_components = {"prefill", *decode_components}
        covered = {row.get("component") for row in scheduled}
        missing_components = sorted(expected_components - covered)
        valid = not missing_components
        quality.append({
            "window_index": window_index, "start_wall_ns": t0, "end_wall_ns": t1,
            "valid": valid, "invalid_reason": None if valid else "component_coverage_gap",
            "coverage_components_json": _json(sorted(covered)),
            "missing_components_json": _json(missing_components),
            "scheduler_iteration_count": len(scheduled), "membership_count": len(members),
            "device_sample_count": len(metrics), "active_request_count": len(active),
        })
        windows.append({"window_index": window_index, "start_wall_ns": t0, "end_wall_ns": t1})
    derived = root / "derived"
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - deployment contract
        raise RuntimeError("numpy is required to write Full-order snapshots") from exc
    np.save(derived / "full_state.npy", np.asarray(state_rows, dtype=float))
    np.save(derived / "disturbance.npy", np.asarray(disturbance_rows, dtype=float))
    np.save(derived / "output.npy", np.asarray(output_rows, dtype=float))
    np.save(derived / "next_state.npy", np.asarray(state_rows[1:], dtype=float))
    (derived / "static_config.json").write_text(json.dumps(_load_static_config(root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_parquet(derived / "snapshot_windows.parquet", windows)
    _write_parquet(derived / "snapshot_quality.parquet", quality)
    (derived / "request_index.json").write_text(json.dumps(request_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (derived / "bin_schema.yaml").write_text(
        "schema_version: " + SNAPSHOT_SCHEMA_VERSION + "\n"
        + f"period_ms: {config.period_ms}\ninput_histogram:\n  width_tokens: {config.input_histogram_bin_tokens}\n  max_tokens: {config.input_histogram_max_tokens}\n"
        + f"context_histogram:\n  width_tokens: {config.context_histogram_bin_tokens}\n  max_tokens: {config.context_histogram_max_tokens}\n",
        encoding="utf-8",
    )
    return {"schema_version": SNAPSHOT_SCHEMA_VERSION, "window_count": len(windows), "valid_window_count": sum(row["valid"] for row in quality), "request_count": len(request_by_id), "period_ms": config.period_ms}
