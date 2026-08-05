#!/usr/bin/env python3
"""Build per-case benchmark and observability summaries from a PD experiment run."""

import argparse
import csv
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


def percentile(values, q):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def stats(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {
            "avg": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "avg": round(sum(values) / len(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p90": round(percentile(values, 0.90), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "max": round(max(values), 3),
    }


def parse_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_observations(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
                record["_time"] = parse_timestamp(record["timestamp"])
                records.append(record)
            except Exception as exc:
                records.append({"_parse_error": f"line {line_number}: {exc!r}"})
    return records


def benchmark_window(data):
    ended = datetime.strptime(data["date"], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    return ended - timedelta(seconds=float(data["duration"])), ended


def nearest_record(records, target, before):
    candidates = [record for record in records if "_time" in record]
    if before:
        eligible = [record for record in candidates if record["_time"] <= target]
        return max(eligible, key=lambda record: record["_time"], default=None)
    eligible = [record for record in candidates if record["_time"] >= target]
    return min(eligible, key=lambda record: record["_time"], default=None)


def engine_counter_delta(before, after, engine, metric):
    try:
        return round(after["engines"][engine][metric] - before["engines"][engine][metric], 3)
    except (KeyError, TypeError):
        return None


def summarize_case(path, observations):
    data = json.loads(path.read_text(encoding="utf-8"))
    started, ended = benchmark_window(data)
    samples = [
        record for record in observations
        if "_time" in record and started <= record["_time"] <= ended
    ]
    before = nearest_record(observations, started, before=True)
    after = nearest_record(observations, ended, before=False)

    result = {
        "case": path.stem,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "sample_count": len(samples),
        "benchmark": data,
        "engines": {},
        "roles": {},
        "npus": {},
        "counter_deltas": {},
    }

    for engine in ("prefill", "decode_a", "decode_b"):
        engine_samples = [record.get("engines", {}).get(engine, {}) for record in samples]
        result["engines"][engine] = {
            "running": stats([sample.get("num_requests_running") for sample in engine_samples]),
            "waiting": stats([sample.get("num_requests_waiting") for sample in engine_samples]),
            "kv_usage": stats([sample.get("kv_cache_usage_perc") for sample in engine_samples]),
            "waiting_nonzero_fraction": round(
                sum((sample.get("num_requests_waiting") or 0) > 0 for sample in engine_samples)
                / len(engine_samples), 4
            ) if engine_samples else None,
        }
        result["counter_deltas"][engine] = {
            metric: engine_counter_delta(before, after, engine, metric)
            for metric in (
                "prompt_tokens_total",
                "generation_tokens_total",
                "request_success_total",
                "external_prefix_cache_hits_total",
                "external_prefix_cache_queries_total",
            )
        }

    for role in ("prefill", "decode_a", "decode_b", "proxy"):
        process_samples = [record.get("processes", {}).get(role, {}) for record in samples]
        result["roles"][role] = {
            "cpu_percent": stats([sample.get("cpu_percent") for sample in process_samples]),
            "rss_mib": stats([sample.get("rss_mib") for sample in process_samples]),
        }

    for phy_id in range(2, 8):
        npu_samples = [record.get("npus", {}).get(str(phy_id), {}) for record in samples]
        cores = [sample.get("aicore_percent") for sample in npu_samples]
        result["npus"][str(phy_id)] = {
            "aicore_percent": stats(cores),
            "aicore_active_fraction": round(
                sum((value or 0) >= 5 for value in cores) / len(cores), 4
            ) if cores else None,
            "aicore_high_fraction": round(
                sum((value or 0) >= 80 for value in cores) / len(cores), 4
            ) if cores else None,
            "hbm_used_mib": stats([sample.get("hbm_used_mib") for sample in npu_samples]),
        }

    proxy_counts = [record.get("proxy", {}).get("request_num") for record in samples]
    result["proxy_request_num"] = stats(proxy_counts)
    result["sample_errors"] = sum(bool(record.get("errors")) for record in samples)
    return result


def parse_transfer_logs(log_dir):
    pattern = re.compile(
        r"(?P<request>\S+) took (?P<ms>[0-9.]+) ms.*local_device_id (?P<rank>\d+)"
    )
    result = {}
    for name in ("decode-a", "decode-b"):
        path = log_dir / f"{name}.log"
        entries = []
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "KV cache transfer for request" not in line:
                    continue
                match = pattern.search(line)
                if match:
                    entries.append({
                        "request": match.group("request"),
                        "duration_ms": float(match.group("ms")),
                        "rank": int(match.group("rank")),
                    })
        by_request = {}
        for entry in entries:
            by_request.setdefault(entry["request"], []).append(entry)
        critical = [max(item["duration_ms"] for item in items) for items in by_request.values()]
        result[name.replace("-", "_")] = {
            "rank_events": len(entries),
            "requests": len(by_request),
            "critical_duration_ms": stats(critical),
            "incomplete_rank_requests": sum(len(items) != 2 for items in by_request.values()),
        }
    return result


def flatten_rows(cases):
    rows = []
    for case in cases:
        bench = case["benchmark"]
        row = {
            "case": case["case"],
            "completed": bench.get("completed"),
            "failed": bench.get("failed"),
            "duration_s": bench.get("duration"),
            "max_concurrency": bench.get("max_concurrency"),
            "request_rate": bench.get("request_rate"),
            "request_throughput": bench.get("request_throughput"),
            "request_goodput": bench.get("request_goodput"),
            "output_tok_s": bench.get("output_throughput"),
            "total_tok_s": bench.get("total_token_throughput"),
            "ttft_p50_ms": bench.get("p50_ttft_ms"),
            "ttft_p95_ms": bench.get("p95_ttft_ms"),
            "tpot_p50_ms": bench.get("p50_tpot_ms"),
            "tpot_p95_ms": bench.get("p95_tpot_ms"),
            "e2el_p95_ms": bench.get("p95_e2el_ms"),
            "prefill_wait_max": case["engines"]["prefill"]["waiting"]["max"],
            "decode_a_wait_max": case["engines"]["decode_a"]["waiting"]["max"],
            "decode_b_wait_max": case["engines"]["decode_b"]["waiting"]["max"],
            "decode_a_generation_delta": case["counter_deltas"]["decode_a"]["generation_tokens_total"],
            "decode_b_generation_delta": case["counter_deltas"]["decode_b"]["generation_tokens_total"],
        }
        for phy_id in range(2, 8):
            row[f"npu{phy_id}_aicore_avg"] = case["npus"][str(phy_id)]["aicore_percent"]["avg"]
            row[f"npu{phy_id}_aicore_p90"] = case["npus"][str(phy_id)]["aicore_percent"]["p90"]
            row[f"npu{phy_id}_hbm_max_mib"] = case["npus"][str(phy_id)]["hbm_used_mib"]["max"]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root
    observations = load_observations(root / "observations.jsonl")
    cases = [
        summarize_case(path, observations)
        for path in sorted((root / "benchmarks").glob("*.json"))
    ]
    summary = {
        "result_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_records": len(observations),
        "observation_parse_errors": sum("_parse_error" in record for record in observations),
        "cases": cases,
        "transfer": parse_transfer_logs(root / "logs"),
    }
    (root / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = flatten_rows(cases)
    if rows:
        with (root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "cases": len(cases),
        "observations": len(observations),
        "analysis": str(root / "analysis.json"),
        "csv": str(root / "benchmark_summary.csv"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
