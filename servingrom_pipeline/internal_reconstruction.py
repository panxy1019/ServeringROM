from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .internal_event_reader import InternalEventDataset, read_internal_events


DERIVED_TABLES = (
    "engine_requests",
    "scheduler_iterations",
    "scheduler_membership",
    "token_emissions",
    "kv_transfer_ranks",
    "kv_transfers",
    "model_execution_batches",
    "device_metrics",
    "prefill_accounting",
)

_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
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
    engine_request_id = event.get("request_id") if request_id is None else request_id
    canonical_request_id = engine_request_id
    association = attempts.get(engine_request_id, {})
    if not association and engine_request_id:
        for candidate in _UUID_PATTERN.findall(engine_request_id):
            if candidate in attempts:
                canonical_request_id = candidate
                association = attempts[candidate]
                break
    return {
        "run_id": event["run_id"],
        "config_id": event["config_id"],
        "component": event["component"],
        "process_instance_id": event["process_instance_id"],
        "event_seq": event["event_seq"],
        "event_type": event["event_type"],
        "ts_wall_ns": event["ts_wall_ns"],
        "ts_mono_ns": event["ts_mono_ns"],
        "request_id": canonical_request_id,
        "engine_request_id": engine_request_id,
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
    kv_events: list[dict[str, Any]] = []

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
                row.update(
                    {
                        "iteration_id": payload.get("iteration_id"),
                        **{key: value for key, value in member.items() if key != "request_id"},
                    }
                )
                tables["scheduler_membership"].append(row)
        elif event_type == "engine_output_batch":
            for member in payload.get("members", []):
                row = _base(event, attempts, member.get("request_id"))
                row.update(
                    {
                        "iteration_id": payload.get("iteration_id"),
                        **{key: value for key, value in member.items() if key != "request_id"},
                    }
                )
                tables["token_emissions"].append(row)
        elif event_type in {
            "kv_transfer_enqueued",
            "kv_transfer_started",
            "kv_transfer_completed",
            "kv_transfer_failed",
        }:
            row = _base(event, attempts)
            row.update(payload)
            kv_events.append(row)
        elif event_type == "model_execution_batch":
            row = _base(event, attempts)
            row.update(payload)
            tables["model_execution_batches"].append(row)
        elif event_type == "prefill_accounting_probe":
            row = _base(event, attempts)
            row.update(payload)
            for field in (
                "probe_phase", "scheduled_tokens", "computed_tokens_before",
                "computed_tokens_after", "input_tokens", "output_tokens_after",
                "connector_computed_tokens", "connector_external_tokens",
                "final_computed_tokens", "handoff_token_count",
                "observation_source",
            ):
                row.setdefault(field, None)
            tables["prefill_accounting"].append(row)
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
                "initial_computed_tokens": added.get("payload", {}).get("initial_computed_tokens") if added else None,
                "max_output_tokens": added.get("payload", {}).get("max_output_tokens") if added else None,
                "has_kv_transfer": added.get("payload", {}).get("has_kv_transfer") if added else None,
                "finish_reason": terminal.get("payload", {}).get("finish_reason") if terminal else None,
            }
        )
        tables["engine_requests"].append(row)

    rank_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in kv_events:
        rank_groups[
            (
                row.get("request_id"),
                row.get("component"),
                row.get("target_engine"),
                row.get("tp_rank"),
            )
        ].append(row)

    for _, events in sorted(rank_groups.items(), key=lambda item: str(item[0])):
        events.sort(key=lambda row: (row["ts_mono_ns"], row["event_seq"]))
        anchor = events[0]
        starts = [row for row in events if row["event_type"] == "kv_transfer_started"]
        completions = [row for row in events if row["event_type"] == "kv_transfer_completed"]
        failures = [row for row in events if row["event_type"] == "kv_transfer_failed"]
        actual_bytes = sum(
            int(row["actual_bytes"])
            for row in completions
            if row.get("actual_bytes") is not None
        )
        rank_row = {
            **{key: anchor.get(key) for key in (
                "run_id", "config_id", "component", "process_instance_id",
                "request_id", "engine_request_id", "trace_id", "attempt_id",
                "engine_role", "engine_instance",
                "source_engine", "target_engine", "transfer_role", "tp_rank", "tp_size",
                "remote_request_id", "remote_handshake_port",
            )},
            "enqueue_wall_ns": min(
                (row.get("enqueue_wall_ns") for row in events if row.get("enqueue_wall_ns") is not None),
                default=None,
            ),
            "enqueue_mono_ns": min(
                (row.get("enqueue_mono_ns") for row in events if row.get("enqueue_mono_ns") is not None),
                default=None,
            ),
            "first_start_wall_ns": min(
                (row.get("start_wall_ns") for row in starts if row.get("start_wall_ns") is not None),
                default=None,
            ),
            "first_start_mono_ns": min(
                (row.get("start_mono_ns") for row in starts if row.get("start_mono_ns") is not None),
                default=None,
            ),
            "last_complete_wall_ns": max(
                (row.get("complete_wall_ns") for row in completions if row.get("complete_wall_ns") is not None),
                default=None,
            ),
            "last_complete_mono_ns": max(
                (row.get("complete_mono_ns") for row in completions if row.get("complete_mono_ns") is not None),
                default=None,
            ),
            "block_count": max(
                (int(row["block_count"]) for row in events if row.get("block_count") is not None),
                default=None,
            ),
            "actual_bytes": actual_bytes if completions else None,
            "descriptor_count": sum(
                int(row["descriptor_count"])
                for row in completions
                if row.get("descriptor_count") is not None
            ) if completions else None,
            "enqueue_count": sum(row["event_type"] == "kv_transfer_enqueued" for row in events),
            "start_count": len(starts),
            "complete_count": len(completions),
            "failure_count": len(failures),
            "success": bool(completions) and not failures and len(starts) == len(completions),
            "error_codes_json": _json([row.get("error_code") for row in failures]),
        }
        first_start = rank_row["first_start_mono_ns"]
        last_complete = rank_row["last_complete_mono_ns"]
        rank_row["transfer_wall_ns"] = (
            last_complete - first_start
            if first_start is not None and last_complete is not None
            else None
        )
        tables["kv_transfer_ranks"].append(rank_row)

    request_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in tables["kv_transfer_ranks"]:
        request_groups[(row.get("request_id"), row.get("target_engine"))].append(row)
    for _, ranks in sorted(request_groups.items(), key=lambda item: str(item[0])):
        anchor = ranks[0]
        tp_size = max((int(row.get("tp_size") or 0) for row in ranks), default=0)
        completed_ranks = sorted(
            int(row["tp_rank"])
            for row in ranks
            if row.get("tp_rank") is not None and row.get("success")
        )
        observed_ranks = sorted(
            {int(row["tp_rank"]) for row in ranks if row.get("tp_rank") is not None}
        )
        expected_ranks = list(range(tp_size))
        missing_ranks = sorted(set(expected_ranks) - set(completed_ranks))
        first_start = min(
            (row["first_start_mono_ns"] for row in ranks if row.get("first_start_mono_ns") is not None),
            default=None,
        )
        last_complete = max(
            (row["last_complete_mono_ns"] for row in ranks if row.get("last_complete_mono_ns") is not None),
            default=None,
        )
        first_start_wall = min(
            (row["first_start_wall_ns"] for row in ranks if row.get("first_start_wall_ns") is not None),
            default=None,
        )
        last_complete_wall = max(
            (row["last_complete_wall_ns"] for row in ranks if row.get("last_complete_wall_ns") is not None),
            default=None,
        )
        tables["kv_transfers"].append(
            {
                **{key: anchor.get(key) for key in (
                    "run_id", "config_id", "request_id", "trace_id", "attempt_id",
                    "source_engine", "target_engine", "remote_request_id",
                )},
                "proxy_decoder_backend": attempts.get(anchor.get("request_id"), {}).get(
                    "proxy_decoder_backend"
                ),
                "expected_rank_count": tp_size,
                "observed_rank_count": len(observed_ranks),
                "completed_rank_count": len(completed_ranks),
                "observed_ranks_json": _json(observed_ranks),
                "completed_ranks_json": _json(completed_ranks),
                "missing_ranks_json": _json(missing_ranks),
                "enqueue_wall_ns": min(
                    (row["enqueue_wall_ns"] for row in ranks if row.get("enqueue_wall_ns") is not None),
                    default=None,
                ),
                "enqueue_mono_ns": min(
                    (row["enqueue_mono_ns"] for row in ranks if row.get("enqueue_mono_ns") is not None),
                    default=None,
                ),
                "first_start_mono_ns": first_start,
                "last_complete_mono_ns": last_complete,
                "kv_ready_mono_ns": last_complete if not missing_ranks else None,
                "first_start_wall_ns": first_start_wall,
                "last_complete_wall_ns": last_complete_wall,
                "kv_ready_wall_ns": last_complete_wall if not missing_ranks else None,
                "transfer_wall_ns": (
                    last_complete - first_start
                    if first_start is not None and last_complete is not None
                    else None
                ),
                "actual_total_bytes": sum(
                    int(row["actual_bytes"])
                    for row in ranks
                    if row.get("actual_bytes") is not None
                ),
                "block_count": sum(
                    int(row["block_count"])
                    for row in ranks
                    if row.get("block_count") is not None
                ),
                "success": (
                    bool(ranks)
                    and not missing_ranks
                    and all(bool(row.get("success")) for row in ranks)
                ),
            }
        )
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
