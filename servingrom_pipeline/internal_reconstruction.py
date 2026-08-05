from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .internal_event_reader import InternalEventDataset, read_internal_events


DERIVED_TABLES = (
    "engine_requests",
    "scheduler_iterations",
    "scheduler_membership",
    "token_emissions",
    "kv_transfers",
    "model_execution_batches",
    "device_metrics",
)


def _json(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _attempt_index(run_root: Path) -> dict[str, dict[str, Any]]:
    path = Path(run_root) / "derived" / "attempt_lifecycle.parquet"
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    return {
        row["request_id"]: {
            "trace_id": row.get("trace_id"),
            "attempt_id": row.get("attempt_id"),
            "proxy_decoder_backend": row.get("decoder_backend"),
        }
        for row in pq.read_table(path).to_pylist()
        if row.get("request_id")
    }


def _base(event: dict[str, Any], attempts: dict[str, dict[str, Any]], request_id=None):
    payload = event.get("payload", {})
    request_id = event.get("request_id") if request_id is None else request_id
    association = attempts.get(request_id, {})
    return {
        "run_id": event["run_id"],
        "config_id": event["config_id"],
        "component": event["component"],
        "process_instance_id": event["process_instance_id"],
        "event_seq": event["event_seq"],
        "event_type": event["event_type"],
        "ts_wall_ns": event["ts_wall_ns"],
        "ts_mono_ns": event["ts_mono_ns"],
        "request_id": request_id,
        "trace_id": association.get("trace_id", event.get("trace_id")),
        "attempt_id": association.get("attempt_id", event.get("attempt_id")),
        "engine_role": payload.get("engine_role"),
        "engine_instance": payload.get("engine_instance"),
        "tp_rank": payload.get("tp_rank"),
        "is_driver_rank": payload.get("is_driver_rank"),
    }


def reconstruct_internal_tables(
    run_root: Path, dataset: InternalEventDataset | None = None
) -> dict[str, list[dict[str, Any]]]:
    root = Path(run_root)
    dataset = read_internal_events(root) if dataset is None else dataset
    attempts = _attempt_index(root)
    tables = {name: [] for name in DERIVED_TABLES}
    request_events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for event in dataset.events:
        event_type = event["event_type"]
        payload = event.get("payload", {})
        if event_type.startswith("engine_request_") and event.get("request_id"):
            request_events[(event["component"], event["request_id"])].append(event)
        elif event_type == "scheduler_iteration":
            row = _base(event, attempts)
            row.update({key: value for key, value in payload.items() if key != "members"})
            tables["scheduler_iterations"].append(row)
        elif event_type == "scheduler_membership":
            for member in payload.get("members", []):
                row = _base(event, attempts, member.get("request_id"))
                row.update({"iteration_id": payload.get("iteration_id"), **member})
                tables["scheduler_membership"].append(row)
        elif event_type == "engine_output_batch":
            for member in payload.get("members", []):
                row = _base(event, attempts, member.get("request_id"))
                row.update({"iteration_id": payload.get("iteration_id"), **member})
                tables["token_emissions"].append(row)
        elif event_type.startswith("kv_transfer_"):
            row = _base(event, attempts)
            row.update(payload)
            tables["kv_transfers"].append(row)
        elif event_type == "model_execution_batch":
            row = _base(event, attempts)
            row.update(payload)
            tables["model_execution_batches"].append(row)
        elif event_type == "device_metric":
            row = _base(event, attempts)
            row.update(
                {
                    "sample_interval_ms": payload.get("sample_interval_ms"),
                    "collector_duration_ns": payload.get("collector_duration_ns"),
                    "processes_json": _json(payload.get("processes")),
                    "network_json": _json(payload.get("network")),
                    "npu_metrics_json": _json(payload.get("npu_metrics")),
                    "npu_exporter_error": payload.get("npu_exporter_error"),
                }
            )
            tables["device_metrics"].append(row)

    for (component, request_id), events in sorted(request_events.items()):
        events.sort(key=lambda event: event["event_seq"])
        added = next((event for event in events if event["event_type"] == "engine_request_added"), None)
        terminal = next(
            (
                event
                for event in reversed(events)
                if event["event_type"] in {"engine_request_terminal", "engine_request_aborted"}
            ),
            None,
        )
        anchor = added or terminal
        if anchor is None:
            continue
        row = _base(anchor, attempts, request_id)
        row.update(
            {
                "component": component,
                "added_wall_ns": added.get("ts_wall_ns") if added else None,
                "terminal_wall_ns": terminal.get("ts_wall_ns") if terminal else None,
                "terminal_event": terminal.get("event_type") if terminal else None,
                "prompt_tokens": added.get("payload", {}).get("prompt_tokens") if added else None,
                "max_output_tokens": added.get("payload", {}).get("max_output_tokens") if added else None,
                "has_kv_transfer": added.get("payload", {}).get("has_kv_transfer") if added else None,
                "finish_reason": terminal.get("payload", {}).get("finish_reason") if terminal else None,
            }
        )
        tables["engine_requests"].append(row)
    return tables


def write_internal_parquet(run_root: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    derived = Path(run_root) / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    for name in DERIVED_TABLES:
        rows = tables[name]
        table = pa.Table.from_pylist(rows) if rows else pa.table({"run_id": pa.array([], type=pa.string())})
        path = derived / f"{name}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
