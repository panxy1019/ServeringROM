#!/usr/bin/env python3
"""Formal Round 14.2 workload with an independent composite control program."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp

from control_pilot_workload import ACTUATOR, request_json, rollback_baseline
from rom_workload import (
    arrival_rate_function,
    build_prompt_bank,
    health_monitor,
    schedule_segment,
    summarize,
    wait_for_drain,
)


LEVELS = (0.3, 0.5, 0.7)


def derive_control_seed(dataset_id: str, plan_id: str, split_seed: int) -> int:
    material = f"{dataset_id}\0{plan_id}\0control\0{split_seed}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def append_change(rows: list[dict[str, Any]], offset: float, value: float, phase: str) -> None:
    if rows and abs(float(rows[-1]["rho_A"]) - value) < 1e-12:
        return
    if rows and abs(float(rows[-1]["rho_A"]) - value) > 0.2000000001:
        raise ValueError(f"illegal control step at {offset}: {rows[-1]['rho_A']} -> {value}")
    rows.append({"offset_seconds": float(offset), "rho_A": value, "phase": phase})


def build_composite_schedule(control_seed: int) -> list[dict[str, Any]]:
    """Build a legal 600 s schedule; adjacent planned changes are >=10 s apart."""
    rng = random.Random(control_seed)
    rows: list[dict[str, Any]] = [{"offset_seconds": 0.0, "rho_A": 0.5, "phase": "initial"}]

    # 60-240 s: rate-limited PRBS. Extremes are always separated by 0.5.
    extreme = rng.choice((0.3, 0.7))
    for offset in range(60, 240, 15):
        value = extreme if (offset // 15) % 2 == 0 else 0.5
        append_change(rows, float(offset), value, "rate_limited_prbs")
        if value == 0.5:
            extreme = 0.7 if extreme == 0.3 else 0.3

    # 240-420 s: independent multilevel random dwell, using 10/15/20 s.
    offset = 240.0
    current = float(rows[-1]["rho_A"])
    while offset <= 410.0:
        candidates = [value for value in LEVELS if value != current and abs(value - current) <= 0.2000000001]
        target = rng.choice(candidates)
        append_change(rows, offset, target, "multilevel_random_dwell")
        current = target
        offset += float(rng.choice((10, 15, 20)))

    append_change(rows, 420.0, 0.5, "step_response")
    for offset, value in ((440, 0.7), (460, 0.5), (480, 0.3), (500, 0.5), (520, 0.7), (540, 0.5)):
        append_change(rows, float(offset), value, "step_response" if offset < 540 else "recovery")

    for previous, current in zip(rows, rows[1:]):
        dwell = current["offset_seconds"] - previous["offset_seconds"]
        if dwell < 10.0 or abs(current["rho_A"] - previous["rho_A"]) > 0.2000000001:
            raise AssertionError(f"invalid formal control program: {previous} -> {current}")
        previous["planned_dwell_seconds"] = dwell
    rows[-1]["planned_dwell_seconds"] = 60.0
    return rows


async def apply_formal_value(client: aiohttp.ClientSession, endpoint: str, value: float, label: str) -> dict[str, Any]:
    status, state = await request_json(client, "GET", f"{endpoint}/servingrom/control/state")
    if status != 200:
        raise RuntimeError(f"control state failed: {status} {state}")
    expected = state["effective_rho_A"] if state["control_mode"] == "controlled" else "baseline"
    command = {
        "control_command_id": f"{label}-{uuid.uuid4().hex[:12]}",
        "control_generation": int(state["control_generation"]) + 1,
        "actuator_name": ACTUATOR,
        "requested_value": value,
        "expected_current_value": expected,
        "requested_wall_ns": time.time_ns(),
    }
    status, prepared = await request_json(client, "POST", f"{endpoint}/servingrom/control/prepare", command)
    if status != 200 or not prepared.get("accepted"):
        raise RuntimeError(f"formal control prepare rejected: {status} {prepared}")
    status, committed = await request_json(client, "POST", f"{endpoint}/servingrom/control/commit", command)
    if status != 200 or not committed.get("accepted"):
        raise RuntimeError(f"formal control commit rejected: {status} {committed}")
    return {"command": command, "prepare": prepared, "commit": committed, "observed_wall_ns": time.time_ns()}


async def execute_schedule(client, endpoint, run_id, start_wall_ns, rows):
    results = []
    for index, row in enumerate(rows):
        target = start_wall_ns + int(row["offset_seconds"] * 1e9)
        await asyncio.sleep(max(0.0, (target - time.time_ns()) / 1e9))
        result = await apply_formal_value(client, endpoint, float(row["rho_A"]), f"{run_id}-u{index:03d}")
        results.append({**row, **result})
    return results


async def run(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds, connect=30, sock_read=args.request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit=4096, limit_per_host=4096)
    schedule = build_composite_schedule(args.control_seed)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as client:
        prompt_bank = await build_prompt_bank(client, args.tokenize_endpoint, workload)
        health: list[dict[str, Any]] = []
        stop_health = asyncio.Event()
        monitor = asyncio.create_task(health_monitor(client, args.endpoint, stop_health, health))
        tasks: list[asyncio.Task] = []
        arrival_rng = random.Random(args.arrival_seed)
        rate_fn = arrival_rate_function(
            args.arrival_process, args.arrival_rate,
            on_seconds=args.on_seconds, off_seconds=args.off_seconds,
            on_multiplier=args.on_multiplier, duration=args.measurement_seconds,
        )
        # Control is defined for all warmup and held at 0.5 for well over the final 30 s.
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
        # The 0 s row is descriptive: warmup_control already established the same value.
        control_task = asyncio.create_task(execute_schedule(client, args.endpoint, args.run_id, measurement_start, schedule[1:]))
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
        "schema_version": "servingrom.control_dataset_workload.v1",
        "dataset_id": args.dataset_id, "run_id": args.run_id, "plan_id": args.plan_id,
        "split": args.split, "workload": workload["name"], "arrival_process": args.arrival_process,
        "target_arrival_rate": args.arrival_rate, "load_fraction": args.load_fraction,
        "arrival_seed": args.arrival_seed, "control_seed": args.control_seed,
        "control_program_version": args.control_program_version,
        "warmup_seconds": args.warmup_seconds, "measurement_seconds": args.measurement_seconds,
        "measurement_start_wall_ns": measurement_start, "measurement_end_wall_ns": measurement_end,
        "snapshot_period_ms": args.snapshot_period_ms, "control_schedule": control_results,
        "rollback": rollback, "summary": summarize(results, health, measurement_start, measurement_end),
        "drain": drain, "health_samples": health,
        "prompt_bank": {str(key): {k: v for k, v in value.items() if k != "prompt"} for key, value in prompt_bank.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True); parser.add_argument("--tokenize-endpoint", required=True)
    parser.add_argument("--dataset-id", required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-id", required=True); parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrival-seed", type=int, required=True); parser.add_argument("--control-seed", type=int, required=True)
    parser.add_argument("--arrival-rate", type=float, required=True); parser.add_argument("--load-fraction", type=float, required=True)
    parser.add_argument("--arrival-process", choices=("poisson", "on_off_burst"), required=True)
    parser.add_argument("--control-program-version", required=True)
    parser.add_argument("--on-seconds", type=float, default=10); parser.add_argument("--off-seconds", type=float, default=15)
    parser.add_argument("--on-multiplier", type=float, default=2.5)
    parser.add_argument("--warmup-seconds", type=float, default=120); parser.add_argument("--measurement-seconds", type=float, default=600)
    parser.add_argument("--drain-timeout-seconds", type=float, default=1200)
    parser.add_argument("--request-timeout-seconds", type=float, default=1200); parser.add_argument("--snapshot-period-ms", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workload = json.loads(args.workload_config.read_text(encoding="utf-8"))
    result = asyncio.run(run(args, workload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"run_id": args.run_id, "summary": result["summary"], "drain": result["drain"]}))
    return 0 if result["drain"]["drained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
