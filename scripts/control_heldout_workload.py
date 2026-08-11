#!/usr/bin/env python3
"""Run one isolated Round 14.3 held-out actuator trajectory."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import aiohttp

from control_dataset_workload import apply_formal_value, execute_schedule
from control_pilot_workload import rollback_baseline
from rom_workload import (
    arrival_rate_function,
    build_prompt_bank,
    health_monitor,
    schedule_segment,
    summarize,
    wait_for_drain,
)


def derive_trajectory_seed(benchmark_id: str, plan_id: str, arrival_seed: int) -> int:
    raw = f"{benchmark_id}\0{plan_id}\0trajectory\0{arrival_seed}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _append(rows: list[dict[str, Any]], offset: float, value: float, phase: str) -> None:
    if rows and abs(float(rows[-1]["rho_A"]) - value) < 1e-12:
        return
    if rows and abs(float(rows[-1]["rho_A"]) - value) > 0.2000000001:
        raise ValueError(f"illegal held-out step: {rows[-1]['rho_A']} -> {value}")
    rows.append({"offset_seconds": float(offset), "rho_A": float(value), "phase": phase})


def build_heldout_schedule(family: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"offset_seconds": 0.0, "rho_A": 0.5, "phase": "initial"}]
    rng = random.Random(seed)
    if family == "interpolation":
        pattern = (0.6, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5)
        offset, index = 60.0, 0
        while offset <= 510:
            _append(rows, offset, pattern[index % len(pattern)], family)
            offset += float(rng.choice((15, 20, 25, 30)))
            index += 1
    elif family == "unseen-composite":
        pattern = (0.6, 0.5, 0.4, 0.5, 0.7, 0.5, 0.3, 0.5)
        offset, index = 60.0, 0
        while offset <= 510:
            _append(rows, offset, pattern[index % len(pattern)], family)
            offset += float(rng.choice((20, 25, 30)))
            index += 1
    elif family == "slow-ramp":
        pattern = (0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5)
        for index, offset in enumerate(range(60, 511, 25)):
            _append(rows, float(offset), pattern[index % len(pattern)], family)
    elif family == "boundary-near":
        pattern = (0.4, 0.2, 0.4, 0.5, 0.6, 0.8, 0.6, 0.5)
        for index, offset in enumerate(range(60, 511, 25)):
            _append(rows, float(offset), pattern[index % len(pattern)], family)
    else:
        raise ValueError(f"unknown trajectory family: {family}")
    _append(rows, 540.0, 0.5, "recovery")
    for previous, current in zip(rows, rows[1:]):
        dwell = current["offset_seconds"] - previous["offset_seconds"]
        if dwell < 15 or abs(current["rho_A"] - previous["rho_A"]) > 0.2000000001:
            raise AssertionError(f"invalid held-out schedule: {previous} -> {current}")
        previous["planned_dwell_seconds"] = dwell
    rows[-1]["planned_dwell_seconds"] = 60.0
    return rows


async def run(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds, connect=30, sock_read=args.request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit=4096, limit_per_host=4096)
    schedule = build_heldout_schedule(args.trajectory_family, args.trajectory_seed)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as client:
        prompt_bank = await build_prompt_bank(client, args.tokenize_endpoint, workload)
        health: list[dict[str, Any]] = []
        stop_health = asyncio.Event()
        monitor = asyncio.create_task(health_monitor(client, args.endpoint, stop_health, health))
        tasks: list[asyncio.Task] = []
        arrival_rng = random.Random(args.arrival_seed)
        rate_fn = arrival_rate_function(
            "poisson", args.arrival_rate, on_seconds=10, off_seconds=15,
            on_multiplier=2.5, duration=args.measurement_seconds,
        )
        warmup_control = await apply_formal_value(client, args.endpoint, 0.5, f"{args.run_id}-warmup")
        index = await schedule_segment(
            client=client, endpoint=args.endpoint, prompt_bank=prompt_bank, workload=workload,
            rate_function=rate_fn, duration=args.warmup_seconds, rng=arrival_rng,
            seed=args.arrival_seed, run_id=args.run_id, phase="warmup", tasks=tasks, start_index=0,
        )
        period_ns = args.snapshot_period_ms * 1_000_000
        measurement_start = ((time.time_ns() + 1_000_000_000 + period_ns - 1) // period_ns) * period_ns
        await asyncio.sleep(max(0.0, (measurement_start - time.time_ns()) / 1e9))
        measurement_end = measurement_start + int(args.measurement_seconds * 1e9)
        control_task = asyncio.create_task(
            execute_schedule(client, args.endpoint, args.run_id, measurement_start, schedule[1:])
        )
        await schedule_segment(
            client=client, endpoint=args.endpoint, prompt_bank=prompt_bank, workload=workload,
            rate_function=rate_fn, duration=args.measurement_seconds, rng=arrival_rng,
            seed=args.arrival_seed, run_id=args.run_id, phase="measurement", tasks=tasks, start_index=index,
        )
        control_results = [{**schedule[0], **warmup_control}] + await control_task
        await asyncio.gather(*tasks)
        drain = await wait_for_drain(client, args.endpoint, args.drain_timeout_seconds)
        rollback = await rollback_baseline(client, args.endpoint, args.run_id)
        stop_health.set()
        await monitor
    results = [task.result() for task in tasks]
    return {
        "schema_version": "servingrom.control_heldout_workload.v1",
        "benchmark_id": args.benchmark_id, "run_id": args.run_id, "plan_id": args.plan_id,
        "split": "test/control-heldout", "workload": workload["name"],
        "trajectory_family": args.trajectory_family, "arrival_process": "poisson",
        "target_arrival_rate": args.arrival_rate, "load_fraction": args.load_fraction,
        "arrival_seed": args.arrival_seed, "trajectory_seed": args.trajectory_seed,
        "trajectory_program_version": args.trajectory_program_version,
        "warmup_seconds": args.warmup_seconds, "measurement_seconds": args.measurement_seconds,
        "measurement_start_wall_ns": measurement_start, "measurement_end_wall_ns": measurement_end,
        "snapshot_period_ms": args.snapshot_period_ms, "control_schedule": control_results,
        "rollback": rollback, "summary": summarize(results, health, measurement_start, measurement_end),
        "drain": drain, "health_samples": health,
        "prompt_bank": {str(key): {k: v for k, v in value.items() if k != "prompt"} for key, value in prompt_bank.items()},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--workload-config", type=Path, required=True)
    p.add_argument("--endpoint", required=True); p.add_argument("--tokenize-endpoint", required=True)
    p.add_argument("--benchmark-id", required=True); p.add_argument("--run-id", required=True); p.add_argument("--plan-id", required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--trajectory-family", required=True)
    p.add_argument("--arrival-seed", type=int, required=True); p.add_argument("--trajectory-seed", type=int, required=True)
    p.add_argument("--arrival-rate", type=float, required=True); p.add_argument("--load-fraction", type=float, required=True)
    p.add_argument("--trajectory-program-version", required=True)
    p.add_argument("--warmup-seconds", type=float, default=120); p.add_argument("--measurement-seconds", type=float, default=600)
    p.add_argument("--drain-timeout-seconds", type=float, default=1200)
    p.add_argument("--request-timeout-seconds", type=float, default=1200); p.add_argument("--snapshot-period-ms", type=int, default=200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args, json.loads(args.workload_config.read_text())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps({"run_id": args.run_id, "summary": result["summary"], "drain": result["drain"]}))
    return 0 if result["drain"]["drained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
