"""Fixed-period, event-replayed ServingROM Full-order snapshots."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SNAPSHOT_SCHEMA_VERSION = "servingrom.full_order_snapshot.v2"
TERMINAL_STATES = {"COMPLETED", "CANCELLED", "FAILED", "REJECTED"}
ACTIVE_STATES = {
    "ADMITTED", "PREFILL_WAITING", "PREFILL_RUNNING", "HANDOFF_WAITING",
    "KV_QUEUED", "KV_TRANSFERRING", "KV_READY", "DECODE_WAITING",
    "DECODE_RUNNING",
}

LENGTH_EDGES = (0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, math.inf)
PREFILL_AGE_EDGES_MS = (0, 25, 50, 100, 200, 400, 800, 1200, 1600, 2000, 3000, math.inf)
TTFT_SLACK_EDGES_MS = (-math.inf, -1000, -500, -200, -100, -50, 0, 50, 100, 200, 400, 800, 1600, math.inf)
PROGRESS_EDGES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0, math.inf)
DECODE_CONTEXT_EDGES = (0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, math.inf)
GENERATION_PROGRESS_EDGES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0, math.inf)
TPOT_SLACK_EDGES_MS = (-math.inf, -100, -50, -20, -10, 0, 10, 20, 40, 60, 100, 200, math.inf)
WORKERS = ("decode-0", "decode-1")


@dataclass(frozen=True)
class SnapshotConfig:
    period_ms: int = 200
    default_ttft_slo_ms: float = 2000.0
    default_tpot_slo_ms: float = 100.0

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
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(rows) if rows else pa.table({"window_id": pa.array([], type=pa.int64())})
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _bin(value: float, edges: tuple[float, ...]) -> int:
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return len(edges) - 2


def _histogram(values: Iterable[int | float | None], *, maximum: int, width: int) -> list[int]:
    """Compatibility helper used by the narrow unit test suite."""
    bins = [0] * (math.ceil(maximum / width) + 1)
    for value in values:
        if value is not None:
            bins[min(max(int(value), 0) // width, len(bins) - 1)] += 1
    return bins


def _bound(value: float) -> float | None:
    return None if math.isinf(value) else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Vector:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.index: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        value: float | int,
        *,
        block: str,
        quantity: str,
        unit: str,
        description: str,
        worker: str | None = None,
        bins: dict[str, Any] | None = None,
    ) -> None:
        self.index.append({
            "index": len(self.values), "name": name, "block": block,
            "worker": worker, "quantity": quantity, "unit": unit,
            "description": description, **(bins or {}),
        })
        self.values.append(float(value))


def _target_worker(decoder_backend: Any) -> str | None:
    value = str(decoder_backend or "")
    if value.endswith(":13701"):
        return "decode-0"
    if value.endswith(":13702"):
        return "decode-1"
    return None


def _request_rows(root: Path) -> list[dict[str, Any]]:
    attempts = _read_parquet(root / "derived" / "attempt_lifecycle.parquet")
    traces = _read_parquet(root / "derived" / "trace_lifecycle.parquet")
    engines = _read_parquet(root / "derived" / "engine_requests.parquet")
    memberships = _read_parquet(root / "derived" / "scheduler_membership.parquet")
    emissions = _read_parquet(root / "derived" / "token_emissions.parquet")
    transfers = _read_parquet(root / "derived" / "kv_transfers.parquet")
    trace_by_id = {row["trace_id"]: row for row in traces if row.get("trace_id")}
    rows: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        request_id = attempt.get("request_id")
        if not request_id:
            continue
        trace = trace_by_id.get(attempt.get("trace_id"), {})
        rows[request_id] = {
            **trace, **attempt, "request_id": request_id,
            "arrival_wall_ns": trace.get("arrival_wall_ns"),
            "terminal_wall_ns": trace.get("terminal_wall_ns"),
            "terminal_event": trace.get("terminal_event"),
            "admission_accepted": bool(trace.get("accepted")),
            "input_tokens": trace.get("input_tokens"),
            "expected_output_tokens": trace.get("expected_output_tokens"),
            "output_tokens": trace.get("output_tokens"),
            "stream": trace.get("stream"),
            "decode_component": _target_worker(attempt.get("decoder_backend")),
        }
    for engine in engines:
        row = rows.get(engine.get("request_id"))
        if row is None:
            continue
        component = engine.get("component")
        if component == "prefill":
            row["prefill_added_wall_ns"] = engine.get("added_wall_ns")
            row["prefill_terminal_wall_ns"] = engine.get("terminal_wall_ns")
            row["input_tokens"] = row.get("input_tokens") or engine.get("prompt_tokens")
        elif component in WORKERS:
            row["decode_component"] = component
            row["decode_added_wall_ns"] = engine.get("added_wall_ns")
            row["decode_terminal_wall_ns"] = engine.get("terminal_wall_ns")
    membership_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in memberships:
        if member.get("request_id") in rows:
            membership_by_request[member["request_id"]].append(member)
    emission_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in emissions:
        if event.get("request_id") in rows:
            emission_by_request[event["request_id"]].append(event)
    for request_id, row in rows.items():
        members = sorted(membership_by_request.get(request_id, []), key=lambda item: int(item.get("ts_wall_ns") or 0))
        row["memberships"] = members
        prefill_members = [item for item in members if item.get("component") == "prefill"]
        decode_members = [item for item in members if item.get("component") in WORKERS]
        row["prefill_first_schedule_wall_ns"] = prefill_members[0].get("ts_wall_ns") if prefill_members else None
        row["decode_first_schedule_wall_ns"] = decode_members[0].get("ts_wall_ns") if decode_members else None
        row["token_events"] = sorted(emission_by_request.get(request_id, []), key=lambda item: int(item.get("ts_wall_ns") or 0))
    for transfer in transfers:
        row = rows.get(transfer.get("request_id"))
        if row is None:
            continue
        row.update({
            "kv_enqueue_wall_ns": transfer.get("enqueue_wall_ns"),
            "kv_first_start_wall_ns": transfer.get("first_start_wall_ns"),
            "kv_ready_wall_ns": transfer.get("kv_ready_wall_ns"),
            "kv_actual_bytes": transfer.get("actual_total_bytes"),
            "kv_success": transfer.get("success"),
            "kv_missing_ranks": _parse_json(transfer.get("missing_ranks_json"), []),
        })
    return list(rows.values())


def _state_with_quality(row: dict[str, Any], wall_ns: int) -> tuple[str, list[str]]:
    arrival = row.get("arrival_wall_ns")
    if arrival is None or wall_ns < int(arrival):
        return "ABSENT", []
    terminal = row.get("terminal_wall_ns")
    terminal_event = str(row.get("terminal_event") or "")
    if terminal_event == "request_rejected":
        if terminal is not None and wall_ns >= int(terminal):
            return "REJECTED", []
        return "ADMITTED", []
    if terminal is not None and wall_ns >= int(terminal):
        if "cancel" in terminal_event:
            return "CANCELLED", []
        if "error" in terminal_event:
            return "FAILED", []
        return "COMPLETED", []
    if not row.get("admission_accepted"):
        return "ADMITTED", ["accepted_state_missing"]
    submit = row.get("prefill_submit_wall_ns")
    if submit is None or wall_ns < int(submit):
        return "ADMITTED", []
    first_prefill = row.get("prefill_first_schedule_wall_ns") or row.get("prefill_added_wall_ns")
    if first_prefill is None or wall_ns < int(first_prefill):
        return "PREFILL_WAITING", []
    prefill_done = row.get("prefill_terminal_wall_ns") or row.get("prefill_complete_wall_ns")
    if prefill_done is None or wall_ns < int(prefill_done):
        return "PREFILL_RUNNING", []
    enqueue = row.get("kv_enqueue_wall_ns")
    if enqueue is None or wall_ns < int(enqueue):
        return "HANDOFF_WAITING", [] if enqueue is not None else ["kv_enqueue_missing"]
    start = row.get("kv_first_start_wall_ns")
    if start is None or wall_ns < int(start):
        return "KV_QUEUED", [] if start is not None else ["kv_start_missing"]
    ready = row.get("kv_ready_wall_ns")
    if ready is None or wall_ns < int(ready):
        return "KV_TRANSFERRING", [] if ready is not None else ["kv_ready_missing"]
    first_decode = row.get("decode_first_schedule_wall_ns") or row.get("decode_added_wall_ns")
    if first_decode is None or wall_ns < int(first_decode):
        return "DECODE_WAITING", [] if first_decode is not None else ["decode_schedule_missing"]
    return "DECODE_RUNNING", []


def _state_at(row: dict[str, Any], wall_ns: int) -> str:
    return _state_with_quality(row, wall_ns)[0]


def _generated_at(row: dict[str, Any], wall_ns: int) -> tuple[int, int | None]:
    generated = 0
    previous = None
    for event in row.get("token_events", []):
        timestamp = event.get("ts_wall_ns")
        if timestamp is None or int(timestamp) >= wall_ns:
            break
        generated += int(event.get("new_token_count") or 0)
        previous = int(timestamp)
    return generated, previous


def _latest_membership(row: dict[str, Any], wall_ns: int, component: str) -> dict[str, Any] | None:
    latest = None
    for member in row.get("memberships", []):
        timestamp = member.get("ts_wall_ns")
        if timestamp is not None and int(timestamp) < wall_ns and member.get("component") == component:
            latest = member
    return latest


def _matrix_add_2d(
    vector: _Vector,
    *,
    prefix: str,
    block: str,
    quantity: str,
    unit: str,
    description: str,
    values: list[list[float]],
    x_edges: tuple[float, ...],
    y_edges: tuple[float, ...],
    x_name: str,
    y_name: str,
    worker: str | None = None,
) -> None:
    for x in range(len(x_edges) - 1):
        for y in range(len(y_edges) - 1):
            vector.add(
                f"{prefix}.{x_name}_{x}.{y_name}_{y}", values[x][y], block=block,
                worker=worker, quantity=quantity, unit=unit, description=description,
                bins={
                    f"{x_name}_lower": _bound(x_edges[x]), f"{x_name}_upper": _bound(x_edges[x + 1]),
                    f"{y_name}_lower": _bound(y_edges[y]), f"{y_name}_upper": _bound(y_edges[y + 1]),
                },
            )


def _state_vector(requests: list[dict[str, Any]], wall_ns: int, config: SnapshotConfig) -> tuple[_Vector, dict[str, str], list[str]]:
    vector = _Vector()
    states: dict[str, str] = {}
    state_errors: list[str] = []
    active: list[dict[str, Any]] = []
    for row in requests:
        state, errors = _state_with_quality(row, wall_ns)
        states[row["request_id"]] = state
        state_errors.extend(f"{row['request_id']}:{error}" for error in errors)
        if state in ACTIVE_STATES:
            active.append(row)
    counts = Counter(states.values())
    expected_remaining = {worker: 0 for worker in WORKERS}
    for row in active:
        worker = row.get("decode_component")
        if worker in WORKERS:
            generated, _ = _generated_at(row, wall_ns)
            expected_remaining[worker] += max(int(row.get("expected_output_tokens") or 0) - generated, 0)
    scalar_values = {
        "active_trace_count": len({row.get("trace_id") for row in active}),
        "active_attempt_count": len(active),
        "accepted_not_terminal_count": len(active),
        "admitted_count": counts["ADMITTED"],
        "prefill_waiting_count": counts["PREFILL_WAITING"],
        "prefill_running_count": counts["PREFILL_RUNNING"],
        "prefill_inflight_tokens": sum(int(row.get("input_tokens") or 0) for row in active if states[row["request_id"]] in {"PREFILL_WAITING", "PREFILL_RUNNING"}),
        "handoff_pending_count": sum(counts[name] for name in ("HANDOFF_WAITING", "KV_QUEUED", "KV_TRANSFERRING", "KV_READY")),
        "handoff_waiting_count": counts["HANDOFF_WAITING"],
        "kv_queue_count": counts["KV_QUEUED"],
        "kv_transfer_inflight_count": counts["KV_TRANSFERRING"],
        "kv_transfer_inflight_bytes": sum(int(row.get("kv_actual_bytes") or 0) for row in active if states[row["request_id"]] == "KV_TRANSFERRING"),
        "kv_ready_wait_count": counts["KV_READY"] + counts["DECODE_WAITING"],
        "kv_ready_count": counts["KV_READY"],
        "decode_d1_waiting_count": sum(states[row["request_id"]] == "DECODE_WAITING" for row in active if row.get("decode_component") == "decode-0"),
        "decode_d1_running_count": sum(states[row["request_id"]] == "DECODE_RUNNING" for row in active if row.get("decode_component") == "decode-0"),
        "decode_d2_waiting_count": sum(states[row["request_id"]] == "DECODE_WAITING" for row in active if row.get("decode_component") == "decode-1"),
        "decode_d2_running_count": sum(states[row["request_id"]] == "DECODE_RUNNING" for row in active if row.get("decode_component") == "decode-1"),
        "decode_d1_expected_remaining_tokens": expected_remaining["decode-0"],
        "decode_d2_expected_remaining_tokens": expected_remaining["decode-1"],
        "decode_route_imbalance_requests": abs(sum(row.get("decode_component") == "decode-0" for row in active) - sum(row.get("decode_component") == "decode-1" for row in active)),
        "decode_route_imbalance_tokens": abs(expected_remaining["decode-0"] - expected_remaining["decode-1"]),
    }
    for name, value in scalar_values.items():
        unit = "bytes" if name.endswith("bytes") else "tokens" if name.endswith("tokens") else "requests"
        vector.add(name, value, block="scalar", quantity=name, unit=unit, description="Event-replayed system inventory at window start.")

    length_bins, age_bins = len(LENGTH_EDGES) - 1, len(PREFILL_AGE_EDGES_MS) - 1
    wait_count = [[0.0] * age_bins for _ in range(length_bins)]
    wait_mass = [[0.0] * age_bins for _ in range(length_bins)]
    slack_bins = len(TTFT_SLACK_EDGES_MS) - 1
    slack_count = [[0.0] * slack_bins for _ in range(length_bins)]
    slack_mass = [[0.0] * slack_bins for _ in range(length_bins)]
    progress_bins = len(PROGRESS_EDGES) - 1
    running_count = [[0.0] * progress_bins for _ in range(length_bins)]
    remaining_mass = [[0.0] * progress_bins for _ in range(length_bins)]
    for row in active:
        state = states[row["request_id"]]
        input_tokens = int(row.get("input_tokens") or 0)
        length_index = _bin(input_tokens, LENGTH_EDGES)
        if state == "PREFILL_WAITING":
            age_ms = max(0.0, (wall_ns - int(row.get("prefill_submit_wall_ns") or wall_ns)) / 1e6)
            age_index = _bin(age_ms, PREFILL_AGE_EDGES_MS)
            wait_count[length_index][age_index] += 1
            wait_mass[length_index][age_index] += input_tokens
        if state in {"PREFILL_WAITING", "PREFILL_RUNNING"}:
            elapsed_ms = max(0.0, (wall_ns - int(row.get("arrival_wall_ns") or wall_ns)) / 1e6)
            slack_index = _bin(config.default_ttft_slo_ms - elapsed_ms, TTFT_SLACK_EDGES_MS)
            slack_count[length_index][slack_index] += 1
            slack_mass[length_index][slack_index] += input_tokens
        if state == "PREFILL_RUNNING":
            member = _latest_membership(row, wall_ns, "prefill")
            computed = int(member.get("computed_tokens_before") or 0) if member else 0
            progress = computed / input_tokens if input_tokens else 0.0
            progress_index = _bin(progress, PROGRESS_EDGES)
            running_count[length_index][progress_index] += 1
            remaining_mass[length_index][progress_index] += max(input_tokens - computed, 0)
    _matrix_add_2d(vector, prefix="prefill.wait_count", block="prefill_wait", quantity="request_count", unit="requests", description="Waiting Prefill requests by input length and age.", values=wait_count, x_edges=LENGTH_EDGES, y_edges=PREFILL_AGE_EDGES_MS, x_name="length", y_name="age_ms")
    _matrix_add_2d(vector, prefix="prefill.wait_token_mass", block="prefill_wait", quantity="token_mass", unit="tokens", description="Waiting Prefill token mass by input length and age.", values=wait_mass, x_edges=LENGTH_EDGES, y_edges=PREFILL_AGE_EDGES_MS, x_name="length", y_name="age_ms")
    _matrix_add_2d(vector, prefix="prefill.ttft_slack_count", block="prefill_slack", quantity="request_count", unit="requests", description="Accepted requests without first token by length and TTFT slack.", values=slack_count, x_edges=LENGTH_EDGES, y_edges=TTFT_SLACK_EDGES_MS, x_name="length", y_name="slack_ms")
    _matrix_add_2d(vector, prefix="prefill.ttft_slack_token_mass", block="prefill_slack", quantity="token_mass", unit="tokens", description="Token mass without first token by length and TTFT slack.", values=slack_mass, x_edges=LENGTH_EDGES, y_edges=TTFT_SLACK_EDGES_MS, x_name="length", y_name="slack_ms")
    _matrix_add_2d(vector, prefix="prefill.running_count", block="prefill_running", quantity="request_count", unit="requests", description="Running Prefill requests by length and progress.", values=running_count, x_edges=LENGTH_EDGES, y_edges=PROGRESS_EDGES, x_name="length", y_name="progress")
    _matrix_add_2d(vector, prefix="prefill.remaining_token_mass", block="prefill_running", quantity="token_mass", unit="tokens", description="Remaining Prefill token mass by length and progress.", values=remaining_mass, x_edges=LENGTH_EDGES, y_edges=PROGRESS_EDGES, x_name="length", y_name="progress")

    for worker in WORKERS:
        for phase, state_name in (("handoff_wait", "HANDOFF_WAITING"), ("kv_queue", "KV_QUEUED"), ("kv_inflight", "KV_TRANSFERRING"), ("kv_ready_wait", "DECODE_WAITING")):
            count_bins = [0.0] * age_bins
            byte_bins = [0.0] * age_bins
            for row in active:
                if row.get("decode_component") != worker or states[row["request_id"]] != state_name:
                    continue
                anchor = row.get("prefill_terminal_wall_ns") if state_name == "HANDOFF_WAITING" else row.get("kv_enqueue_wall_ns") if state_name == "KV_QUEUED" else row.get("kv_first_start_wall_ns") if state_name == "KV_TRANSFERRING" else row.get("kv_ready_wall_ns")
                age_index = _bin(max(0.0, (wall_ns - int(anchor or wall_ns)) / 1e6), PREFILL_AGE_EDGES_MS)
                count_bins[age_index] += 1
                byte_bins[age_index] += int(row.get("kv_actual_bytes") or 0)
            for index, value in enumerate(count_bins):
                bins = {"age_ms_lower": _bound(PREFILL_AGE_EDGES_MS[index]), "age_ms_upper": _bound(PREFILL_AGE_EDGES_MS[index + 1])}
                vector.add(f"{worker}.{phase}.count.age_{index}", value, block="handoff", worker=worker, quantity="request_count", unit="requests", description="Mooncake request inventory by age.", bins=bins)
                vector.add(f"{worker}.{phase}.bytes.age_{index}", byte_bins[index], block="handoff", worker=worker, quantity="byte_mass", unit="bytes", description="Mooncake byte inventory by age.", bins=bins)

    context_bins = len(DECODE_CONTEXT_EDGES) - 1
    tpot_bins = len(TPOT_SLACK_EDGES_MS) - 1
    generation_bins = len(GENERATION_PROGRESS_EDGES) - 1
    for worker in WORKERS:
        waiting = [[0.0] * tpot_bins for _ in range(context_bins)]
        running = [[0.0] * generation_bins for _ in range(context_bins)]
        remaining = [[0.0] * generation_bins for _ in range(context_bins)]
        context_mass = [[0.0] * generation_bins for _ in range(context_bins)]
        first_token_pending = [0.0] * context_bins
        for row in active:
            if row.get("decode_component") != worker:
                continue
            state = states[row["request_id"]]
            generated, previous_token = _generated_at(row, wall_ns)
            member = _latest_membership(row, wall_ns, worker)
            context = int(member.get("context_tokens_before") or row.get("input_tokens") or 0) if member else int(row.get("input_tokens") or 0)
            context_index = _bin(context, DECODE_CONTEXT_EDGES)
            if state == "DECODE_WAITING":
                if previous_token is None:
                    first_token_pending[context_index] += 1
                else:
                    slack = config.default_tpot_slo_ms - (wall_ns - previous_token) / 1e6
                    waiting[context_index][_bin(slack, TPOT_SLACK_EDGES_MS)] += 1
            if state == "DECODE_RUNNING":
                expected = max(int(row.get("expected_output_tokens") or 0), 1)
                progress_index = _bin(generated / expected, GENERATION_PROGRESS_EDGES)
                running[context_index][progress_index] += 1
                remaining[context_index][progress_index] += max(expected - generated, 0)
                context_mass[context_index][progress_index] += context
        _matrix_add_2d(vector, prefix=f"{worker}.wait_count", block="decode_wait", worker=worker, quantity="request_count", unit="requests", description="Decode waiting requests by context and TPOT slack.", values=waiting, x_edges=DECODE_CONTEXT_EDGES, y_edges=TPOT_SLACK_EDGES_MS, x_name="context", y_name="tpot_slack_ms")
        _matrix_add_2d(vector, prefix=f"{worker}.running_count", block="decode_running", worker=worker, quantity="request_count", unit="requests", description="Decode running requests by context and generation progress.", values=running, x_edges=DECODE_CONTEXT_EDGES, y_edges=GENERATION_PROGRESS_EDGES, x_name="context", y_name="progress")
        _matrix_add_2d(vector, prefix=f"{worker}.remaining_output_mass", block="decode_running", worker=worker, quantity="token_mass", unit="tokens", description="Remaining requested output tokens.", values=remaining, x_edges=DECODE_CONTEXT_EDGES, y_edges=GENERATION_PROGRESS_EDGES, x_name="context", y_name="progress")
        _matrix_add_2d(vector, prefix=f"{worker}.context_token_mass", block="decode_running", worker=worker, quantity="token_mass", unit="tokens", description="Active Decode context token mass.", values=context_mass, x_edges=DECODE_CONTEXT_EDGES, y_edges=GENERATION_PROGRESS_EDGES, x_name="context", y_name="progress")
        for index, value in enumerate(first_token_pending):
            vector.add(f"{worker}.first_token_pending.context_{index}", value, block="decode_wait", worker=worker, quantity="request_count", unit="requests", description="Decode requests without an engine token timestamp.", bins={"context_lower": _bound(DECODE_CONTEXT_EDGES[index]), "context_upper": _bound(DECODE_CONTEXT_EDGES[index + 1])})
    return vector, states, state_errors


def _disturbance_vector(requests: list[dict[str, Any]], t0: int, t1: int) -> _Vector:
    vector = _Vector()
    arrivals = [row for row in requests if row.get("arrival_wall_ns") is not None and t0 <= int(row["arrival_wall_ns"]) < t1]
    accepted = [row for row in arrivals if row.get("admission_accepted")]
    rejected = [row for row in arrivals if row.get("terminal_event") == "request_rejected"]
    scalar = {
        "arrival_request_count": len(arrivals), "accepted_arrival_count": len(accepted),
        "rejected_arrival_count": len(rejected),
        "arrival_prompt_token_mass": sum(int(row.get("input_tokens") or 0) for row in arrivals),
        "arrival_requested_output_token_mass": sum(int(row.get("expected_output_tokens") or 0) for row in arrivals),
        "stream_request_count": sum(bool(row.get("stream")) for row in arrivals),
        "nonstream_request_count": sum(not bool(row.get("stream")) for row in arrivals),
    }
    for name, value in scalar.items():
        vector.add(name, value, block="arrival", quantity=name, unit="tokens" if name.endswith("mass") else "requests", description="External trace arrivals in the half-open window.")
    for prefix, field, edges in (("arrival_input_length", "input_tokens", LENGTH_EDGES), ("arrival_output_length", "expected_output_tokens", LENGTH_EDGES)):
        counts = [0] * (len(edges) - 1)
        for row in arrivals:
            counts[_bin(float(row.get(field) or 0), edges)] += 1
        for index, value in enumerate(counts):
            vector.add(f"{prefix}.bin_{index}", value, block="arrival_histogram", quantity="request_count", unit="requests", description="Arrival request length histogram.", bins={"lower": _bound(edges[index]), "upper": _bound(edges[index + 1])})
    return vector


def _request_tpot_ms(row: dict[str, Any]) -> float | None:
    timestamps = [int(event["ts_wall_ns"]) for event in row.get("token_events", []) if event.get("ts_wall_ns") is not None]
    if len(timestamps) < 2:
        return None
    return sum(b - a for a, b in zip(timestamps, timestamps[1:])) / (len(timestamps) - 1) / 1e6


def _output_vector(
    requests: list[dict[str, Any]], memberships: list[dict[str, Any]],
    emissions: list[dict[str, Any]], transfers: list[dict[str, Any]],
    t0: int, t1: int, config: SnapshotConfig,
) -> _Vector:
    vector = _Vector()
    terminals = [row for row in requests if row.get("terminal_wall_ns") is not None and t0 <= int(row["terminal_wall_ns"]) < t1]
    completed = [row for row in terminals if row.get("terminal_event") == "request_complete"]
    cancelled = [row for row in terminals if "cancel" in str(row.get("terminal_event") or "")]
    failed = [row for row in terminals if "error" in str(row.get("terminal_event") or "")]
    rejected = [row for row in terminals if row.get("terminal_event") == "request_rejected"]
    good = []
    for row in completed:
        ttft_ms = float(row.get("ttft_proxy_ns") or math.inf) / 1e6
        tpot_ms = _request_tpot_ms(row)
        if ttft_ms <= config.default_ttft_slo_ms and (tpot_ms is None or tpot_ms <= config.default_tpot_slo_ms):
            good.append(row)
    window_emissions = [row for row in emissions if row.get("ts_wall_ns") is not None and t0 <= int(row["ts_wall_ns"]) < t1]
    window_members = [row for row in memberships if row.get("ts_wall_ns") is not None and t0 <= int(row["ts_wall_ns"]) < t1]
    window_transfers = [row for row in transfers if row.get("kv_ready_wall_ns") is not None and t0 <= int(row["kv_ready_wall_ns"]) < t1]
    scalar = {
        "completed_request_count": len(completed),
        "completed_output_tokens": sum(int(row.get("output_tokens") or 0) for row in completed),
        "completed_prompt_tokens": sum(int(row.get("input_tokens") or 0) for row in completed),
        "goodput_request_count": len(good),
        "goodput_output_tokens": sum(int(row.get("output_tokens") or 0) for row in good),
        "ttft_slo_violation_count": sum(float(row.get("ttft_proxy_ns") or math.inf) / 1e6 > config.default_ttft_slo_ms for row in completed),
        "tpot_slo_violation_count": sum((_request_tpot_ms(row) or 0) > config.default_tpot_slo_ms for row in completed),
        "ttft_sum_ms": sum(float(row.get("ttft_proxy_ns") or 0) / 1e6 for row in completed),
        "tpot_sum_ms": sum(_request_tpot_ms(row) or 0 for row in completed),
        "request_rejected_count": len(rejected), "request_cancel_count": len(cancelled),
        "request_error_count": len(failed),
        "decode_d1_emitted_tokens": sum(int(row.get("new_token_count") or 0) for row in window_emissions if row.get("component") == "decode-0"),
        "decode_d2_emitted_tokens": sum(int(row.get("new_token_count") or 0) for row in window_emissions if row.get("component") == "decode-1"),
        "kv_transfer_completed_count": len(window_transfers),
        "kv_transfer_completed_bytes": sum(int(row.get("actual_total_bytes") or 0) for row in window_transfers),
        "kv_transfer_failed_count": sum(not bool(row.get("success")) for row in window_transfers),
        "prefill_scheduled_tokens": sum(int(row.get("scheduled_tokens") or 0) for row in window_members if row.get("component") == "prefill"),
        "decode_scheduled_tokens": sum(int(row.get("scheduled_tokens") or 0) for row in window_members if row.get("component") in WORKERS),
    }
    for name, value in scalar.items():
        unit = "ms" if name.endswith("_ms") else "bytes" if name.endswith("bytes") else "tokens" if "tokens" in name else "requests"
        vector.add(name, value, block="window_output", quantity=name, unit=unit, description="Observed result in the half-open window.")
    return vector


def _static_config(root: Path, config: SnapshotConfig) -> dict[str, Any]:
    deployment = (root / "metadata" / "deployment.yaml").read_text(encoding="utf-8") if (root / "metadata" / "deployment.yaml").exists() else ""
    def arg(name: str) -> int | None:
        match = re.search(rf"--{re.escape(name)}(?:=|\s+)(\d+)", deployment)
        return int(match.group(1)) if match else None
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "config_id": "qwen36-1p2d-d2-full-decode-only-async-v1",
        "prefill_max_num_batched_tokens": 8192,
        "prefill_max_num_seqs": 16,
        "prefill_chunked_prefill_enabled": None,
        "prefill_long_prefill_threshold": None,
        "decode_max_num_batched_tokens": 4096,
        "decode_max_num_seqs": 64,
        "graph_mode": "FULL_DECODE_ONLY",
        "async_scheduling": True,
        "prefill_tp": 2,
        "decode_tp": 2,
        "kv_connector": "MooncakeConnectorV1",
        "proxy_max_prefill_inflight_tokens": arg("max-prefill-inflight-tokens") or 8192,
        "model_revision": None,
        "model_max_len": 32768,
        "precision": "W8A8",
        "snapshot_period_ms": config.period_ms,
        "ttft_slo_ms": config.default_ttft_slo_ms,
        "tpot_slo_ms": config.default_tpot_slo_ms,
        "slo_source": "run_default",
        "unavailable_fields": ["physical_dma_time", "per_request_kv_blocks_free", "npu_hardware_utilization"],
    }


def _writer_health(root: Path) -> tuple[bool, list[str], dict[str, int]]:
    errors: list[str] = []
    components: Counter[str] = Counter()
    summaries = list((root / "raw").glob("**/*.summary.json"))
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        component = str(payload.get("component") or path.parent.name)
        components[component] += 1
        if payload.get("events_written") != payload.get("events_enqueued"):
            errors.append(f"writer_mismatch:{path.relative_to(root)}")
        if any(int(payload.get(name, 0)) for name in ("events_dropped_queue_full", "events_dropped_writer_failed", "serialization_errors", "write_errors", "flush_errors")):
            errors.append(f"writer_error:{path.relative_to(root)}")
    for expected in ("proxy", "prefill", "decode-0", "decode-1", "mooncake", "device"):
        if components[expected] == 0:
            errors.append(f"writer_checkpoint_missing:{expected}")
    return not errors, errors, dict(components)


def _snapshot_manifest(root: Path, snapshot_dir: Path, input_paths: list[Path]) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "inputs": {path.relative_to(root).as_posix(): _sha256(path) for path in input_paths if path.exists()},
        "outputs": {path.relative_to(root).as_posix(): _sha256(path) for path in sorted(snapshot_dir.iterdir()) if path.is_file() and path.name != "snapshot_manifest.json"},
    }


def build_snapshots(run_root: Path, config: SnapshotConfig = SnapshotConfig()) -> dict[str, Any]:
    import numpy as np
    root = Path(run_root)
    derived = root / "derived"
    snapshot_dir = derived / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    requests = _request_rows(root)
    scheduler = _read_parquet(derived / "scheduler_iterations.parquet")
    memberships = _read_parquet(derived / "scheduler_membership.parquet")
    emissions = _read_parquet(derived / "token_emissions.parquet")
    transfers = _read_parquet(derived / "kv_transfers.parquet")
    devices = _read_parquet(derived / "device_metrics.parquet")
    arrivals = [int(row["arrival_wall_ns"]) for row in requests if row.get("arrival_wall_ns") is not None]
    terminals = [int(row["terminal_wall_ns"]) for row in requests if row.get("terminal_wall_ns") is not None]
    if not arrivals or not terminals:
        raise ValueError("snapshot measurement requires request arrival and terminal events")
    start = min(arrivals) // config.period_ns * config.period_ns
    end = ((max(terminals) // config.period_ns) + 1) * config.period_ns
    boundaries = list(range(start, end + config.period_ns, config.period_ns))
    state_vectors: list[list[float]] = []
    state_maps: list[dict[str, str]] = []
    state_errors: list[list[str]] = []
    state_index: list[dict[str, Any]] | None = None
    for boundary in boundaries:
        vector, states, errors = _state_vector(requests, boundary, config)
        state_vectors.append(vector.values)
        state_maps.append(states)
        state_errors.append(errors)
        state_index = state_index or vector.index
        if vector.index != state_index:
            raise RuntimeError("state index changed across windows")
    writer_ok, writer_errors, writer_components = _writer_health(root)
    device_times = sorted(int(row["ts_wall_ns"]) for row in devices if row.get("ts_wall_ns") is not None)
    run_id = requests[0].get("run_id") or root.name
    config_id = requests[0].get("config_id") or "qwen36-1p2d-d2-full-decode-only-async-v1"
    window_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    request_state_rows: list[dict[str, Any]] = []
    disturbances: list[list[float]] = []
    outputs: list[list[float]] = []
    disturbance_index = output_index = None
    for index, (t0, t1) in enumerate(zip(boundaries, boundaries[1:])):
        disturbance = _disturbance_vector(requests, t0, t1)
        output = _output_vector(requests, memberships, emissions, transfers, t0, t1, config)
        disturbances.append(disturbance.values)
        outputs.append(output.values)
        disturbance_index = disturbance_index or disturbance.index
        output_index = output_index or output.index
        if disturbance.index != disturbance_index or output.index != output_index:
            raise RuntimeError("window vector index changed")
        active_start = sum(state in ACTIVE_STATES for state in state_maps[index].values())
        active_end = sum(state in ACTIVE_STATES for state in state_maps[index + 1].values())
        nearest_device_gap = min((abs(timestamp - (t0 + t1) // 2) for timestamp in device_times), default=10**30)
        reasons = list(state_errors[index])
        if not writer_ok:
            reasons.extend(writer_errors)
        if nearest_device_gap > config.period_ns:
            reasons.append("device_sample_coverage_gap")
        component_coverage = {
            "proxy": writer_components.get("proxy", 0) > 0,
            "prefill": writer_components.get("prefill", 0) > 0,
            "decode-0": writer_components.get("decode-0", 0) > 0,
            "decode-1": writer_components.get("decode-1", 0) > 0,
            "mooncake": writer_components.get("mooncake", 0) > 0,
            "device": nearest_device_gap <= config.period_ns,
        }
        arrivals_count = int(disturbance.values[disturbance_index[0]["index"]])
        terminal_count = sum(1 for row in requests if row.get("terminal_wall_ns") is not None and t0 <= int(row["terminal_wall_ns"]) < t1)
        row = {
            "window_id": index, "start_wall_ns": t0, "end_wall_ns": t1,
            "duration_ms": config.period_ms, "valid": not reasons,
            "invalid_reasons": _json(sorted(set(reasons))),
            "active_requests_start": active_start, "active_requests_end": active_end,
            "arrivals": arrivals_count, "terminals": terminal_count,
            "component_coverage": _json(component_coverage), "run_id": run_id,
            "config_id": config_id,
        }
        window_rows.append(row)
        for request_id in sorted(state_maps[index]):
            request_state_rows.append({
                "window_id": index, "request_id": request_id,
                "state_start": state_maps[index][request_id],
                "state_end": state_maps[index + 1][request_id],
                "run_id": run_id, "config_id": config_id,
            })
        quality_rows.append({
            "window_id": index, "valid": not reasons,
            "invalid_reasons": row["invalid_reasons"],
            "device_nearest_sample_gap_ns": nearest_device_gap,
            "writer_complete": writer_ok,
        })
    state_array = np.asarray(state_vectors, dtype=np.float64)
    full_state = state_array[:-1]
    next_state = state_array[1:]
    disturbance_array = np.asarray(disturbances, dtype=np.float64)
    output_array = np.asarray(outputs, dtype=np.float64)
    static = _static_config(root, config)
    static_numeric_names = [name for name, value in static.items() if isinstance(value, (int, float, bool)) and value is not None]
    static_numeric = np.asarray([float(static[name]) for name in static_numeric_names], dtype=np.float64)
    np.save(snapshot_dir / "full_state.npy", full_state)
    np.save(snapshot_dir / "next_state.npy", next_state)
    np.save(snapshot_dir / "disturbance.npy", disturbance_array)
    np.save(snapshot_dir / "output.npy", output_array)
    np.save(snapshot_dir / "static_config.npy", static_numeric)
    (snapshot_dir / "static_config.json").write_text(json.dumps({**static, "numeric_index": static_numeric_names}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (snapshot_dir / "state_index.json").write_text(json.dumps(state_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (snapshot_dir / "disturbance_index.json").write_text(json.dumps(disturbance_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (snapshot_dir / "output_index.json").write_text(json.dumps(output_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (snapshot_dir / "bin_schema.yaml").write_text(
        "schema_version: " + SNAPSHOT_SCHEMA_VERSION + "\n"
        + f"snapshot_period_ms: {config.period_ms}\nwindow_semantics: '[t_k,t_k+1)'\n"
        + "prefill_length_edges: " + _json([_bound(value) for value in LENGTH_EDGES]) + "\n"
        + "prefill_age_ms_edges: " + _json([_bound(value) for value in PREFILL_AGE_EDGES_MS]) + "\n"
        + "ttft_slack_ms_edges: " + _json([_bound(value) for value in TTFT_SLACK_EDGES_MS]) + "\n"
        + "prefill_progress_edges: " + _json([_bound(value) for value in PROGRESS_EDGES]) + "\n"
        + "decode_context_edges: " + _json([_bound(value) for value in DECODE_CONTEXT_EDGES]) + "\n"
        + "generation_progress_edges: " + _json([_bound(value) for value in GENERATION_PROGRESS_EDGES]) + "\n"
        + "tpot_slack_ms_edges: " + _json([_bound(value) for value in TPOT_SLACK_EDGES_MS]) + "\n",
        encoding="utf-8",
    )
    _write_parquet(snapshot_dir / "window_table.parquet", window_rows)
    _write_parquet(snapshot_dir / "snapshot_quality.parquet", quality_rows)
    _write_parquet(snapshot_dir / "request_state_inventory.parquet", request_state_rows)
    inputs = [derived / f"{name}.parquet" for name in (
        "trace_lifecycle", "attempt_lifecycle", "engine_requests", "scheduler_iterations",
        "scheduler_membership", "token_emissions", "kv_transfer_ranks", "kv_transfers",
        "model_execution_batches", "device_metrics", "prefill_accounting",
    )]
    manifest = _snapshot_manifest(root, snapshot_dir, inputs)
    (snapshot_dir / "snapshot_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION, "window_count": len(window_rows),
        "valid_window_count": sum(bool(row["valid"]) for row in window_rows),
        "valid_window_ratio": sum(bool(row["valid"]) for row in window_rows) / len(window_rows),
        "request_count": len(requests), "period_ms": config.period_ms,
        "state_dimensions": full_state.shape[1], "disturbance_dimensions": disturbance_array.shape[1],
        "output_dimensions": output_array.shape[1],
    }
