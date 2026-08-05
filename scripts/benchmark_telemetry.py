#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servingrom_telemetry.config import TelemetryConfig
from servingrom_telemetry.emitter import AsyncTelemetryEmitter


def percentile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        return 0
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress ServingROM async telemetry")
    parser.add_argument("--events", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--producers", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int, default=131_072)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--flush-interval-ms", type=int, default=100)
    parser.add_argument("--max-file-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--pace-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.events <= 0 or args.producers <= 0:
        raise SystemExit("events and producers must be positive")
    if not 0 < args.pace_threshold < 1:
        raise SystemExit("pace-threshold must be between zero and one")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TelemetryConfig(
        enabled=True,
        experiment_id="servingrom-telemetry-step1-stress",
        run_id=f"stress-{args.events}",
        config_id="telemetry-library-v1",
        component="synthetic-benchmark",
        host_id="local-benchmark",
        output_dir=args.output_dir,
        queue_capacity=args.queue_capacity,
        batch_size=args.batch_size,
        flush_interval_ms=args.flush_interval_ms,
        max_file_bytes=args.max_file_bytes,
    )
    emitter = AsyncTelemetryEmitter(config)
    barrier = threading.Barrier(args.producers + 1)
    latency_arrays = [array("Q") for _ in range(args.producers)]
    emit_failures = [0] * args.producers
    counts = [args.events // args.producers] * args.producers
    for index in range(args.events % args.producers):
        counts[index] += 1

    def produce(worker: int) -> None:
        latencies = latency_arrays[worker]
        barrier.wait()
        for index in range(counts[worker]):
            if index % 256 == 0:
                while (
                    emitter.health_snapshot()["queue_depth_current"]
                    >= args.queue_capacity * args.pace_threshold
                ):
                    time.sleep(0.0005)
            started = time.perf_counter_ns()
            accepted = emitter.emit(
                "synthetic_event",
                {"worker": worker, "index": index, "value": index % 97},
                trace_id=f"trace-{worker}",
                attempt_id=0,
                request_id=f"request-{worker}",
            )
            latencies.append(time.perf_counter_ns() - started)
            if not accepted:
                emit_failures[worker] += 1

    threads = [threading.Thread(target=produce, args=(index,)) for index in range(args.producers)]
    for thread in threads:
        thread.start()
    barrier.wait()
    emit_started = time.perf_counter()
    for thread in threads:
        thread.join()
    emit_finished = time.perf_counter()
    close_ok = emitter.close(timeout_s=300)
    finished = time.perf_counter()

    health = emitter.health_snapshot()
    latencies = sorted(value for values in latency_arrays for value in values)
    output_size = sum(path.stat().st_size for path in args.output_dir.glob("*.jsonl"))
    emit_seconds = emit_finished - emit_started
    total_seconds = finished - emit_started
    report = {
        "events_requested": args.events,
        "producers": args.producers,
        "queue_capacity": args.queue_capacity,
        "batch_size": args.batch_size,
        "emit_latency_ns": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies, default=0),
        },
        "emit_seconds": emit_seconds,
        "end_to_end_seconds": total_seconds,
        "events_per_second": args.events / emit_seconds,
        "writer_events_per_second": health["events_written"] / total_seconds,
        "writer_mib_per_second": output_size / (1024 * 1024) / total_seconds,
        "maximum_queue_depth": health["queue_depth_high_watermark"],
        "output_size_bytes": output_size,
        "emit_returned_false": sum(emit_failures),
        "close_ok": close_ok,
        "health": health,
    }
    report_path = args.output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    valid = (
        close_ok
        and sum(emit_failures) == 0
        and health["events_dropped_queue_full"] == 0
        and health["events_dropped_writer_failed"] == 0
        and health["serialization_errors"] == 0
        and health["write_errors"] == 0
        and health["events_written"] == health["events_enqueued"] == args.events
    )
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
