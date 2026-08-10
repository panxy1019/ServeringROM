#!/usr/bin/env python3
"""Open-loop workload plus guarded Control-v1 excitation on one time base."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp

from rom_workload import (
    arrival_rate_function,
    build_prompt_bank,
    health_monitor,
    schedule_segment,
    summarize,
    wait_for_drain,
)


ACTUATOR = "decode_routing_ratio"


def build_schedule(kind: str, duration: float, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if kind == "prbs":
        # Extreme dwell is 24 s. Every extreme transition uses a legal 0.5
        # bridge for 6 s, so no command exceeds max delta 0.2.
        offset, value = 0.0, 0.3
        rows.append({"offset_seconds": offset, "rho_A": value, "planned_dwell_seconds": 30.0})
        while offset < duration:
            offset += 30.0
            if offset >= duration:
                break
            rows.append({"offset_seconds": offset, "rho_A": 0.5, "planned_dwell_seconds": 6.0})
            offset += 6.0
            if offset >= duration:
                break
            value = 0.7 if value == 0.3 else 0.3
            rows.append({"offset_seconds": offset, "rho_A": value, "planned_dwell_seconds": 24.0})
        return rows
    if kind == "step":
        pattern = (0.5, 0.7, 0.5, 0.3, 0.5)
        offset = 0.0
        while offset < duration:
            for value in pattern:
                if offset >= duration:
                    break
                rows.append({"offset_seconds": offset, "rho_A": value, "planned_dwell_seconds": 30.0})
                offset += 30.0
        return rows
    if kind == "random-dwell":
        rng = random.Random(seed)
        value, offset = 0.5, 0.0
        while offset < duration:
            candidates = [candidate for candidate in (0.3, 0.5, 0.7) if candidate != value]
            target = rng.choice(candidates)
            if abs(target - value) > 0.2:
                target = 0.5
            dwell = float(rng.choice((5, 10, 15)))
            rows.append({"offset_seconds": offset, "rho_A": target, "planned_dwell_seconds": dwell})
            value = target
            offset += dwell
        return rows
    raise ValueError(f"unsupported excitation: {kind}")


async def request_json(client: aiohttp.ClientSession, method: str, url: str, body=None):
    async with client.request(method, url, json=body) as response:
        value = await response.json()
        return response.status, value


async def apply_value(client: aiohttp.ClientSession, endpoint: str, value: float, label: str):
    # An absolute 5 s schedule can reach PREPARE a few milliseconds before the
    # previous command's applied timestamp + minimum dwell. Retry only that
    # explicit safety rejection; every other rejection remains fail-closed.
    dwell_rejections = []
    for retry in range(20):
        status, state = await request_json(client, "GET", f"{endpoint}/servingrom/control/state")
        if status != 200:
            raise RuntimeError(f"control state failed: {status} {state}")
        expected = state["effective_rho_A"] if state["control_mode"] == "controlled" else "baseline"
        command = {
            "control_command_id": f"{label}-r{retry:02d}-{uuid.uuid4().hex[:12]}",
            "control_generation": int(state["control_generation"]) + 1,
            "actuator_name": ACTUATOR,
            "requested_value": value,
            "expected_current_value": expected,
            "requested_wall_ns": time.time_ns(),
        }
        status, prepared = await request_json(client, "POST", f"{endpoint}/servingrom/control/prepare", command)
        if status == 409 and prepared.get("reason") == "minimum_dwell_time_not_met":
            dwell_rejections.append(prepared)
            await asyncio.sleep(0.1)
            continue
        if status != 200 or not prepared.get("accepted"):
            raise RuntimeError(f"control prepare failed: {status} {prepared}")
        status, committed = await request_json(client, "POST", f"{endpoint}/servingrom/control/commit", command)
        if status != 200 or not committed.get("accepted"):
            raise RuntimeError(f"control commit failed: {status} {committed}")
        return {
            "command": command, "prepare": prepared, "commit": committed,
            "observed_wall_ns": time.time_ns(),
            "minimum_dwell_retry_count": len(dwell_rejections),
        }
    raise RuntimeError(f"minimum dwell retry budget exhausted: {dwell_rejections[-1]}")


async def rollback_baseline(client: aiohttp.ClientSession, endpoint: str, label: str):
    _, state = await request_json(client, "GET", f"{endpoint}/servingrom/control/state")
    if state["control_mode"] == "baseline":
        return {"already_baseline": True, "state": state}
    command = {
        "control_command_id": f"{label}-rollback-{uuid.uuid4().hex[:12]}",
        "control_generation": int(state["control_generation"]) + 1,
        "actuator_name": ACTUATOR,
        "requested_value": "baseline",
        "expected_current_value": state["effective_rho_A"],
        "requested_wall_ns": time.time_ns(),
    }
    status, value = await request_json(client, "POST", f"{endpoint}/servingrom/control/rollback", command)
    if status != 200 or not value.get("accepted"):
        raise RuntimeError(f"control rollback failed: {status} {value}")
    return value


async def execute_schedule(
    client: aiohttp.ClientSession,
    endpoint: str,
    run_id: str,
    measurement_start_wall_ns: int,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows):
        target_wall_ns = measurement_start_wall_ns + int(row["offset_seconds"] * 1e9)
        await asyncio.sleep(max(0.0, (target_wall_ns - time.time_ns()) / 1e9))
        result = await apply_value(client, endpoint, float(row["rho_A"]), f"{run_id}-u{index:03d}")
        results.append({**row, **result})
    return results


async def run(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds, connect=30, sock_read=args.request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit=4096, limit_per_host=4096)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as client:
        prompt_bank = await build_prompt_bank(client, args.tokenize_endpoint, workload)
        health: list[dict[str, Any]] = []
        stop_health = asyncio.Event()
        monitor = asyncio.create_task(health_monitor(client, args.endpoint, stop_health, health))
        tasks: list[asyncio.Task] = []
        rng = random.Random(args.seed)
        rate_fn = arrival_rate_function(
            "poisson", args.arrival_rate, on_seconds=10, off_seconds=10,
            on_multiplier=2, duration=args.measurement_seconds,
        )
        index = await schedule_segment(
            client=client, endpoint=args.endpoint, prompt_bank=prompt_bank,
            workload=workload, rate_function=rate_fn, duration=args.warmup_seconds,
            rng=rng, seed=args.seed, run_id=args.run_id, phase="warmup",
            tasks=tasks, start_index=0,
        )
        schedule = build_schedule(args.excitation, args.measurement_seconds, args.seed)
        # Apply the first value before the measured interval and leave one full
        # second for command visibility before the aligned 200 ms boundary.
        first = await apply_value(client, args.endpoint, float(schedule[0]["rho_A"]), f"{args.run_id}-u000")
        schedule = schedule[1:]
        period_ns = args.snapshot_period_ms * 1_000_000
        measurement_start = ((time.time_ns() + 1_000_000_000 + period_ns - 1) // period_ns) * period_ns
        await asyncio.sleep(max(0.0, (measurement_start - time.time_ns()) / 1e9))
        measurement_end = measurement_start + int(args.measurement_seconds * 1e9)
        control_task = asyncio.create_task(
            execute_schedule(client, args.endpoint, args.run_id, measurement_start, schedule)
        )
        await schedule_segment(
            client=client, endpoint=args.endpoint, prompt_bank=prompt_bank,
            workload=workload, rate_function=rate_fn, duration=args.measurement_seconds,
            rng=rng, seed=args.seed, run_id=args.run_id, phase="measurement",
            tasks=tasks, start_index=index,
        )
        control_results = [
            {**build_schedule(args.excitation, args.measurement_seconds, args.seed)[0], **first}
        ] + await control_task
        await asyncio.gather(*tasks)
        drain = await wait_for_drain(client, args.endpoint, args.drain_timeout_seconds)
        rollback = await rollback_baseline(client, args.endpoint, args.run_id)
        stop_health.set()
        await monitor
    results = [task.result() for task in tasks]
    return {
        "schema_version": "servingrom.control_pilot_workload.v1",
        "run_id": args.run_id,
        "workload": workload["name"],
        "excitation": args.excitation,
        "target_arrival_rate": args.arrival_rate,
        "load_fraction": args.load_fraction,
        "seed": args.seed,
        "warmup_seconds": args.warmup_seconds,
        "measurement_seconds": args.measurement_seconds,
        "measurement_start_wall_ns": measurement_start,
        "measurement_end_wall_ns": measurement_end,
        "snapshot_period_ms": args.snapshot_period_ms,
        "control_schedule": control_results,
        "rollback": rollback,
        "summary": summarize(results, health, measurement_start, measurement_end),
        "drain": drain,
        "health_samples": health,
        "prompt_bank": {str(key): {k: v for k, v in value.items() if k != "prompt"} for key, value in prompt_bank.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenize-endpoint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arrival-rate", type=float, required=True)
    parser.add_argument("--load-fraction", type=float, required=True)
    parser.add_argument("--excitation", choices=("prbs", "random-dwell", "step"), required=True)
    parser.add_argument("--warmup-seconds", type=float, default=120)
    parser.add_argument("--measurement-seconds", type=float, default=600)
    parser.add_argument("--drain-timeout-seconds", type=float, default=1200)
    parser.add_argument("--request-timeout-seconds", type=float, default=1200)
    parser.add_argument("--snapshot-period-ms", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workload = json.loads(args.workload_config.read_text(encoding="utf-8"))
    result = asyncio.run(run(args, workload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"run_id": args.run_id, "summary": result["summary"], "drain": result["drain"]}, ensure_ascii=False))
    return 0 if result["drain"]["drained"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
