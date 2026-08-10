"""Read-only Control-v1 inputs aligned to frozen 200 ms snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _proxy_events(root: Path) -> list[dict[str, Any]]:
    events = []
    for path in sorted((root / "raw" / "proxy").glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            events.append(json.loads(line))
    return sorted(events, key=lambda row: (int(row["ts_wall_ns"]), int(row["event_seq"])))


def build_control_snapshots(run_root: Path) -> dict[str, Any]:
    import numpy as np

    root = Path(run_root)
    snapshots = root / "derived" / "snapshots"
    output_dir = root / "derived" / "control"
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = _read_parquet(snapshots / "window_table.parquet")
    events = _proxy_events(root)
    command_events: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] != "actuator_applied":
            continue
        payload = event["payload"]
        command_id = str(payload["control_command_id"])
        previous = command_events.get(command_id)
        if previous and previous["payload"] != payload:
            raise ValueError(f"conflicting actuator_applied event: {command_id}")
        command_events[command_id] = event
    applied = sorted(command_events.values(), key=lambda row: int(row["payload"]["applied_wall_ns"]))
    if not applied:
        raise ValueError("no actuator_applied events found")

    routes = [event for event in events if event["event_type"] == "p_to_d_route"]
    control_rows = []
    cursor = -1
    current = None
    previous_u = None
    for window in windows:
        t0, t1 = int(window["start_wall_ns"]), int(window["end_wall_ns"])
        while cursor + 1 < len(applied) and int(applied[cursor + 1]["payload"]["applied_wall_ns"]) <= t0:
            cursor += 1
            current = applied[cursor]
        if current is None:
            raise ValueError(f"window {window['window_id']} begins before first actuator_applied")
        payload = current["payload"]
        value = payload["effective_value"]
        if not isinstance(value, (int, float)):
            raise ValueError(f"window {window['window_id']} has non-numeric U: {value}")
        selected = [event for event in routes if t0 <= int(event["ts_wall_ns"]) < t1]
        a_routes = [event for event in selected if str(event["payload"].get("selected_decoder", "")).endswith(":13701")]
        token_total = sum(float(event["payload"].get("expected_output_tokens") or 0) for event in selected)
        token_a = sum(float(event["payload"].get("expected_output_tokens") or 0) for event in a_routes)
        before = [event["payload"].get("expected_remaining_tokens_before") or {} for event in selected]
        a_remaining = [float(row.get("0.0.0.0:13701", 0)) for row in before]
        b_remaining = [float(row.get("0.0.0.0:13702", 0)) for row in before]
        row = {
            "window_id": int(window["window_id"]),
            "start_wall_ns": t0,
            "end_wall_ns": t1,
            "u_rho_A": float(value),
            "u_prev": float(value if previous_u is None else previous_u),
            "delta_u": float(0.0 if previous_u is None else float(value) - previous_u),
            "time_since_control_change_seconds": (t0 - int(payload["applied_wall_ns"])) / 1e9,
            "control_command_id": payload["control_command_id"],
            "control_generation": int(payload["control_generation"]),
            "control_applied_wall_ns": int(payload["applied_wall_ns"]),
            "u_source_event_type": "actuator_applied",
            "routed_request_count": len(selected),
            "routed_A_request_count": len(a_routes),
            "actual_request_ratio": len(a_routes) / len(selected) if selected else None,
            "routed_expected_token_mass": token_total,
            "routed_A_expected_token_mass": token_a,
            "actual_token_ratio": token_a / token_total if token_total else None,
            "decode_A_expected_remaining_mean": sum(a_remaining) / len(a_remaining) if a_remaining else None,
            "decode_B_expected_remaining_mean": sum(b_remaining) / len(b_remaining) if b_remaining else None,
        }
        control_rows.append(row)
        previous_u = float(value)

    u = np.asarray([[row["u_rho_A"]] for row in control_rows], dtype=np.float64)
    auxiliary = np.asarray(
        [[row["u_prev"], row["delta_u"], row["time_since_control_change_seconds"], row["control_generation"]]
         for row in control_rows], dtype=np.float64,
    )
    np.save(output_dir / "control_input.npy", u)
    np.save(output_dir / "control_auxiliary.npy", auxiliary)
    _write_parquet(output_dir / "control_windows.parquet", control_rows)
    (output_dir / "control_index.json").write_text(json.dumps([
        {"index": 0, "name": "rho_A", "source": "actuator_applied.effective_value", "unit": "ratio"}
    ], indent=2) + "\n", encoding="utf-8")
    (output_dir / "control_auxiliary_index.json").write_text(json.dumps([
        {"index": 0, "name": "U_prev"}, {"index": 1, "name": "delta_U"},
        {"index": 2, "name": "time_since_control_change_seconds"},
        {"index": 3, "name": "control_generation"},
    ], indent=2) + "\n", encoding="utf-8")

    output = np.load(snapshots / "output.npy")
    output_index = {row["name"]: int(row["index"]) for row in json.loads((snapshots / "output_index.json").read_text())}
    slow_rows = []
    for start in range(0, len(control_rows), 25):
        stop = min(start + 25, len(control_rows))
        if stop - start != 25:
            raise ValueError("fast window count is not divisible by 25")
        block = output[start:stop]
        controls = control_rows[start:stop]
        def total(name: str) -> float:
            return float(block[:, output_index[name]].sum())
        completed = total("completed_request_count")
        output_tokens = total("completed_output_tokens")
        good_tokens = total("goodput_output_tokens")
        ttft_sum = total("ttft_sum_ms")
        tpot_sum = total("tpot_sum_ms")
        remaining_a = [row["decode_A_expected_remaining_mean"] for row in controls if row["decode_A_expected_remaining_mean"] is not None]
        remaining_b = [row["decode_B_expected_remaining_mean"] for row in controls if row["decode_B_expected_remaining_mean"] is not None]
        route_count = sum(row["routed_request_count"] for row in controls)
        route_a = sum(row["routed_A_request_count"] for row in controls)
        token_mass = sum(row["routed_expected_token_mass"] for row in controls)
        token_a = sum(row["routed_A_expected_token_mass"] for row in controls)
        slow_rows.append({
            "slow_window_id": len(slow_rows), "fast_window_start": start, "fast_window_end": stop,
            "start_wall_ns": controls[0]["start_wall_ns"], "end_wall_ns": controls[-1]["end_wall_ns"],
            "target_rho_A_mean": sum(row["u_rho_A"] for row in controls) / 25,
            "actual_request_ratio": route_a / route_count if route_count else None,
            "actual_token_ratio": token_a / token_mass if token_mass else None,
            "completed_requests": completed, "completed_output_tokens": output_tokens,
            "throughput_output_tokens_per_second": output_tokens / 5.0,
            "goodput_output_tokens_per_second": good_tokens / 5.0,
            "ttft_mean_ms": ttft_sum / completed if completed else None,
            "tpot_mean_ms": tpot_sum / completed if completed else None,
            "decode_A_expected_remaining_mean": sum(remaining_a) / len(remaining_a) if remaining_a else None,
            "decode_B_expected_remaining_mean": sum(remaining_b) / len(remaining_b) if remaining_b else None,
            "decode_expected_remaining_imbalance": (
                sum(remaining_a) / len(remaining_a) - sum(remaining_b) / len(remaining_b)
                if remaining_a and remaining_b else None
            ),
            "kv_transfer_completed_bytes": total("kv_transfer_completed_bytes"),
            "prefill_scheduled_tokens": total("prefill_scheduled_tokens"),
            "decode_scheduled_tokens": total("decode_scheduled_tokens"),
        })
    _write_parquet(output_dir / "slow_control_kpi_windows.parquet", slow_rows)
    manifest = {
        "schema_version": "servingrom.control_snapshot.v1",
        "u_semantics": "actuator_applied.effective_value forward-held at fast-window start",
        "fast_windows": len(control_rows), "slow_windows": len(slow_rows),
        "inputs": {
            "window_table": _sha256(snapshots / "window_table.parquet"),
            "output": _sha256(snapshots / "output.npy"),
            "proxy_jsonl": {path.name: _sha256(path) for path in sorted((root / "raw" / "proxy").glob("*.jsonl"))},
        },
    }
    (output_dir / "control_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
