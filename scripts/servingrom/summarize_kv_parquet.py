#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "p50": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    requests = pq.read_table(args.run_root / "derived" / "kv_transfers.parquet").to_pylist()
    ranks = pq.read_table(args.run_root / "derived" / "kv_transfer_ranks.parquet").to_pylist()
    bytes_values = [float(row["actual_total_bytes"]) for row in requests]
    wall_ms = [float(row["transfer_wall_ns"]) / 1_000_000 for row in requests]
    result = {
        "request_transfer_count": len(requests),
        "unique_request_count": len({row["request_id"] for row in requests}),
        "successful_request_transfers": sum(bool(row["success"]) for row in requests),
        "request_transfers_with_kv_ready": sum(row["kv_ready_mono_ns"] is not None for row in requests),
        "request_transfers_with_positive_bytes": sum(row["actual_total_bytes"] > 0 for row in requests),
        "route_backend_counts": dict(Counter(row["proxy_decoder_backend"] for row in requests)),
        "missing_rank_rows": sum(row["missing_ranks_json"] != "[]" for row in requests),
        "expected_rank_count_values": sorted({row["expected_rank_count"] for row in requests}),
        "completed_rank_count_values": sorted({row["completed_rank_count"] for row in requests}),
        "rank_transfer_count": len(ranks),
        "successful_rank_transfers": sum(bool(row["success"]) for row in ranks),
        "rank_failure_count": sum(int(row["failure_count"]) for row in ranks),
        "actual_total_bytes_sum": int(sum(bytes_values)),
        "actual_total_bytes_distribution": distribution(bytes_values),
        "transfer_wall_ms_distribution": distribution(wall_ms),
    }
    output = args.output or args.run_root / "reports" / "kv_transfer_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
