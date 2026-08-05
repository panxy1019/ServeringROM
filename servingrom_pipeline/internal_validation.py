from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .internal_event_reader import InternalEventDataset


def validate_internal_data(
    run_root: Path,
    tables: dict[str, list[dict[str, Any]]],
    dataset: InternalEventDataset,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []

    def fail(code: str, **detail: Any) -> None:
        violations.append({"code": code, **detail})

    by_process: dict[str, list[int]] = defaultdict(list)
    for event in dataset.events:
        by_process[event["process_instance_id"]].append(event["event_seq"])
    for process, sequence in by_process.items():
        expected = set(range(1, max(sequence) + 1)) if sequence else set()
        missing = expected - set(sequence)
        duplicate_count = len(sequence) - len(set(sequence))
        if missing or duplicate_count:
            fail("event_seq_gap", process=process, missing=len(missing), duplicates=duplicate_count)

    writer_totals = Counter()
    for summary in dataset.summaries:
        for name in (
            "events_enqueued",
            "events_written",
            "events_dropped_queue_full",
            "events_dropped_writer_failed",
        ):
            writer_totals[name] += int(summary.get(name, 0))
        if summary.get("events_written", 0) != summary.get("events_enqueued", 0):
            fail("writer_counter_mismatch", process=summary.get("process_instance_id"))
    for damaged in dataset.damaged_lines:
        fail("jsonl_damaged_line", **damaged)

    attempts = {}
    attempt_path = Path(run_root) / "derived" / "attempt_lifecycle.parquet"
    if attempt_path.exists():
        import pyarrow.parquet as pq

        attempts = {
            row["request_id"]: row
            for row in pq.read_table(attempt_path).to_pylist()
            if row.get("request_id")
        }
    engine_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tables["engine_requests"]:
        engine_by_request[row["request_id"]].append(row)

    association = {
        "accepted_attempts": 0,
        "prefill_engine_associated": 0,
        "routed_attempts": 0,
        "unique_decode_associated": 0,
        "route_worker_match": 0,
    }
    for request_id, attempt in attempts.items():
        rows = engine_by_request.get(request_id, [])
        if attempt.get("prefill_submit_mono_ns") is not None:
            association["accepted_attempts"] += 1
            prefill = [row for row in rows if row["component"] == "prefill"]
            if prefill:
                association["prefill_engine_associated"] += 1
            else:
                fail("accepted_attempt_missing_prefill_engine", request_id=request_id)
        decoder = attempt.get("decoder_backend")
        if decoder:
            association["routed_attempts"] += 1
            decode_rows = [row for row in rows if row["component"] in {"decode-0", "decode-1"}]
            if len(decode_rows) == 1:
                association["unique_decode_associated"] += 1
                expected_component = "decode-0" if str(decoder).endswith(":13701") else "decode-1"
                if decode_rows[0]["component"] == expected_component:
                    association["route_worker_match"] += 1
                else:
                    fail(
                        "proxy_route_worker_mismatch",
                        request_id=request_id,
                        expected=expected_component,
                        actual=decode_rows[0]["component"],
                    )
            else:
                fail("routed_attempt_decode_engine_cardinality", request_id=request_id, count=len(decode_rows))

    iteration_totals = {
        (row["process_instance_id"], row["iteration_id"]): row
        for row in tables["scheduler_iterations"]
    }
    memberships: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in tables["scheduler_membership"]:
        memberships[(row["process_instance_id"], row["iteration_id"])].append(row)
    for key, iteration in iteration_totals.items():
        members = memberships.get(key, [])
        if sum(row["scheduled_tokens"] for row in members) != iteration["scheduled_tokens_total"]:
            fail("membership_token_sum_mismatch", process=key[0], iteration_id=key[1])
        if len(members) != iteration["scheduled_request_count"]:
            fail("membership_count_mismatch", process=key[0], iteration_id=key[1])

    kv_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tables["kv_transfers"]:
        kv_by_request[row["request_id"]].append(row)
    for request_id, rows in kv_by_request.items():
        event_types = {row["event_type"] for row in rows}
        if "kv_transfer_completed" in event_types and "kv_transfer_started" not in event_types:
            fail("kv_complete_without_start", request_id=request_id)
        if "kv_transfer_started" in event_types and not (
            {"kv_transfer_completed", "kv_transfer_failed"} & event_types
        ):
            fail("kv_transfer_missing_terminal", request_id=request_id)

    prefill_reconciliation = []
    decode_reconciliation = []
    membership_by_request: dict[tuple[str, str], int] = Counter()
    for row in tables["scheduler_membership"]:
        membership_by_request[(row["component"], row["request_id"])] += int(row["scheduled_tokens"])
    output_by_request: dict[tuple[str, str], int] = Counter()
    for row in tables["token_emissions"]:
        output_by_request[(row["component"], row["request_id"])] += int(row["new_token_count"])
    for request_id, rows in engine_by_request.items():
        for row in rows:
            if row["component"] == "prefill":
                scheduled = membership_by_request[("prefill", request_id)]
                prompt = row.get("prompt_tokens")
                delta = scheduled - prompt if prompt is not None else None
                classification = "exact" if delta == 0 else "requires_runtime_semantics_confirmation"
                prefill_reconciliation.append(
                    {"request_id": request_id, "prompt_tokens": prompt, "scheduled_tokens": scheduled, "delta": delta, "classification": classification}
                )
            elif row["component"] in {"decode-0", "decode-1"}:
                engine_tokens = output_by_request[(row["component"], request_id)]
                proxy_tokens = None
                if request_id in attempts:
                    trace_path = Path(run_root) / "derived" / "trace_lifecycle.parquet"
                    if trace_path.exists():
                        import pyarrow.parquet as pq
                        trace_by_id = {item["trace_id"]: item for item in pq.read_table(trace_path).to_pylist()}
                        proxy_tokens = trace_by_id.get(attempts[request_id].get("trace_id"), {}).get("output_tokens")
                decode_reconciliation.append(
                    {"request_id": request_id, "engine_tokens": engine_tokens, "proxy_tokens": proxy_tokens, "delta": engine_tokens - proxy_tokens if proxy_tokens is not None else None}
                )

    violation_counts = Counter(item["code"] for item in violations)
    return {
        "schema_version": "servingrom.internal_quality.v1",
        "event_count": len(dataset.events),
        "damaged_line_count": len(dataset.damaged_lines),
        "writer_totals": dict(writer_totals),
        "association": association,
        "table_counts": {name: len(rows) for name, rows in tables.items()},
        "prefill_token_reconciliation": prefill_reconciliation,
        "decode_token_reconciliation": decode_reconciliation,
        "violations": violations,
        "violation_count": len(violations),
        "violation_counts": dict(violation_counts),
    }


def write_internal_quality_report(run_root: Path, report: dict[str, Any]) -> None:
    reports = Path(run_root) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "internal_data_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# ServingROM 内部遥测数据质量报告",
        "",
        f"- 原始事件：{report['event_count']}",
        f"- 损坏行：{report['damaged_line_count']}",
        f"- 违规：{report['violation_count']}",
        "",
        "## 表规模",
        "",
    ]
    lines.extend(f"- `{name}`：{count}" for name, count in report["table_counts"].items())
    lines.extend(["", "## 违规分布", ""])
    lines.extend(
        [f"- `{name}`：{count}" for name, count in report["violation_counts"].items()]
        or ["- 无"]
    )
    (reports / "internal_data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
