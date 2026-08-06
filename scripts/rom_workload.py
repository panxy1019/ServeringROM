#!/usr/bin/env python3
"""Open-loop workload generator for the immutable ServingROM ROM dataset."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import aiohttp


MODEL = "qwen36-27b-w8a8"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def choose_weighted(rng: random.Random, values: list[dict[str, Any]]) -> int:
    needle = rng.random()
    total = 0.0
    for item in values:
        total += float(item["weight"])
        if needle <= total:
            return int(item["tokens"])
    return int(values[-1]["tokens"])


async def tokenizer_count(client: aiohttp.ClientSession, endpoint: str, prompt: str) -> int:
    async with client.post(
        f"{endpoint.rstrip('/')}/tokenize",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
        },
    ) as response:
        body = await response.json()
        if response.status != 200:
            raise RuntimeError(f"tokenizer returned {response.status}: {body}")
        return int(body["count"])


async def build_prompt_bank(
    client: aiohttp.ClientSession,
    tokenize_endpoint: str,
    workload: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    bank: dict[int, dict[str, Any]] = {}
    template = str(workload["prompt_template"])
    targets = sorted({int(item["tokens"]) for item in workload["input_token_distribution"]})
    for target in targets:
        low, high = 1, max(2, target)
        best: tuple[int, str, int] | None = None
        while low <= high:
            repeat = (low + high) // 2
            prompt = (template + "\n") * repeat
            count = await tokenizer_count(client, tokenize_endpoint, prompt)
            candidate = (abs(count - target), prompt, count)
            if best is None or candidate[0] < best[0]:
                best = candidate
            if count < target:
                low = repeat + 1
            elif count > target:
                high = repeat - 1
            else:
                break
        assert best is not None
        if best[2] + max(int(item["tokens"]) for item in workload["output_token_distribution"]) > int(workload["max_context_tokens"]):
            raise ValueError(f"workload target exceeds legal context: {target} -> {best[2]}")
        bank[target] = {"prompt": best[1], "actual_tokens": best[2], "target_tokens": target}
    return bank


def parse_usage(body: bytes, stream: bool) -> int | None:
    try:
        if not stream:
            return int(json.loads(body).get("usage", {}).get("completion_tokens"))
        completion = None
        for line in body.splitlines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            payload = json.loads(line[6:])
            usage = payload.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion = int(usage["completion_tokens"])
        return completion
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


async def send_request(
    client: aiohttp.ClientSession,
    endpoint: str,
    request_id: str,
    prompt_entry: dict[str, Any],
    output_tokens: int,
    stream: bool,
    seed: int,
    phase: str,
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt_entry["prompt"]}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": seed,
        "ignore_eos": True,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    start_wall = time.time_ns()
    start = time.perf_counter()
    first_byte = None
    status = 0
    body_parts: list[bytes] = []
    error = None
    try:
        async with client.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            headers={"X-Request-Id": request_id},
            json=payload,
        ) as response:
            status = response.status
            async for chunk in response.content.iter_any():
                if first_byte is None:
                    first_byte = time.perf_counter()
                body_parts.append(chunk)
    except Exception as exc:  # recorded and handled by campaign quality gates
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    body = b"".join(body_parts)
    return {
        "request_id": request_id,
        "phase": phase,
        "status": status,
        "error": error,
        "arrival_wall_ns": start_wall,
        "input_tokens": int(prompt_entry["actual_tokens"]),
        "requested_output_tokens": output_tokens,
        "completion_tokens": parse_usage(body, stream),
        "stream": stream,
        "ttft_seconds": first_byte - start if first_byte is not None else None,
        "e2e_seconds": end - start,
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "response_bytes": len(body),
    }


async def health_monitor(
    client: aiohttp.ClientSession,
    endpoint: str,
    stop: asyncio.Event,
    output: list[dict[str, Any]],
) -> None:
    while not stop.is_set():
        started = time.time_ns()
        try:
            async with client.get(f"{endpoint.rstrip('/')}/healthcheck") as response:
                payload = await response.json()
                output.append({"ts_wall_ns": started, "status": response.status, **payload})
        except Exception as exc:
            output.append({"ts_wall_ns": started, "status": 0, "error": f"{type(exc).__name__}: {exc}"})
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


def arrival_rate_function(
    kind: str,
    average_rate: float,
    *,
    on_seconds: float,
    off_seconds: float,
    on_multiplier: float,
    duration: float,
) -> Callable[[float], float]:
    if kind == "poisson":
        return lambda _: average_rate
    if kind == "on_off_burst":
        cycle = on_seconds + off_seconds
        expected_multiplier = on_multiplier * on_seconds / cycle
        if not math.isclose(expected_multiplier, 1.0, rel_tol=1e-9):
            raise ValueError("on/off parameters must preserve the requested long-term average")
        return lambda elapsed: average_rate * on_multiplier if elapsed % cycle < on_seconds else 0.0
    if kind == "step-up":
        return lambda elapsed: average_rate * (0.30 if elapsed < duration / 2 else 0.85)
    if kind == "step-down":
        return lambda elapsed: average_rate * (0.85 if elapsed < duration / 2 else 0.40)
    if kind == "ramp-up":
        return lambda elapsed: average_rate * (0.30 + 0.65 * min(max(elapsed / duration, 0.0), 1.0))
    if kind == "held-out-composite":
        third = duration / 3
        return lambda elapsed: average_rate * (0.90 if elapsed < third else 0.50 if elapsed < 2 * third else 0.80)
    raise ValueError(f"unsupported arrival process: {kind}")


async def schedule_segment(
    *,
    client: aiohttp.ClientSession,
    endpoint: str,
    prompt_bank: dict[int, dict[str, Any]],
    workload: dict[str, Any],
    rate_function: Callable[[float], float],
    duration: float,
    rng: random.Random,
    seed: int,
    run_id: str,
    phase: str,
    tasks: list[asyncio.Task],
    start_index: int,
) -> int:
    started = time.perf_counter()
    index = start_index
    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= duration:
            break
        rate = rate_function(elapsed)
        if rate <= 0:
            await asyncio.sleep(min(0.05, duration - elapsed))
            continue
        interval = rng.expovariate(rate)
        await asyncio.sleep(min(interval, max(duration - elapsed, 0)))
        if time.perf_counter() - started >= duration:
            break
        target = choose_weighted(rng, workload["input_token_distribution"])
        output_tokens = choose_weighted(rng, workload["output_token_distribution"])
        stream = rng.random() < float(workload["stream_ratio"])
        request_id = f"{run_id}-{phase}-{index:07d}"
        tasks.append(asyncio.create_task(send_request(
            client, endpoint, request_id, prompt_bank[target], output_tokens,
            stream, seed + index, phase,
        )))
        index += 1
    return index


async def wait_for_drain(
    client: aiohttp.ClientSession,
    endpoint: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    last = None
    while time.perf_counter() - started < timeout_seconds:
        async with client.get(f"{endpoint.rstrip('/')}/healthcheck") as response:
            last = await response.json()
        active_requests = sum(int(value) for value in last.get("decode_active_requests", {}).values())
        active_tokens = sum(int(value) for value in last.get("decode_expected_remaining_tokens", {}).values())
        if int(last.get("request_num", -1)) == 0 and int(last.get("prefill_inflight_tokens", -1)) == 0 and active_requests == 0 and active_tokens == 0:
            return {"drained": True, "seconds": time.perf_counter() - started, "health": last}
        await asyncio.sleep(1)
    return {"drained": False, "seconds": time.perf_counter() - started, "health": last}


def summarize(
    results: list[dict[str, Any]],
    health: list[dict[str, Any]],
    measurement_start: int | None,
    measurement_end: int | None,
) -> dict[str, Any]:
    selected = [
        row for row in results
        if measurement_start is None or measurement_start <= row["arrival_wall_ns"] < int(measurement_end)
    ]
    statuses = Counter(str(row["status"]) for row in selected)
    successes = [row for row in selected if row["status"] == 200]
    ttft = [row["ttft_seconds"] for row in successes if row["ttft_seconds"] is not None]
    e2e = [row["e2e_seconds"] for row in successes]
    duration = (int(measurement_end) - int(measurement_start)) / 1e9 if measurement_start is not None else None
    return {
        "arrival_count": len(selected),
        "actual_arrival_rate": len(selected) / duration if duration else None,
        "status_counts": dict(statuses),
        "success_count": len(successes),
        "rejected_count": statuses.get("429", 0),
        "error_count": sum(bool(row["error"]) or row["status"] not in (200, 429) for row in selected),
        "input_tokens": sum(row["input_tokens"] for row in selected),
        "requested_output_tokens": sum(row["requested_output_tokens"] for row in selected),
        "completion_tokens": sum(row["completion_tokens"] or 0 for row in successes),
        "ttft_p50_seconds": percentile(ttft, 0.50),
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "ttft_p99_seconds": percentile(ttft, 0.99),
        "e2e_p50_seconds": percentile(e2e, 0.50),
        "e2e_p95_seconds": percentile(e2e, 0.95),
        "e2e_p99_seconds": percentile(e2e, 0.99),
        "health_sample_count": len(health),
    }


def backlog_trend(health: list[dict[str, Any]]) -> tuple[float, float]:
    clean = [row for row in health if row.get("request_num") is not None]
    clean = clean[len(clean) // 2:]
    if len(clean) < 2:
        return math.inf, math.inf
    x = [(row["ts_wall_ns"] - clean[0]["ts_wall_ns"]) / 1e9 for row in clean]
    y = [float(row["request_num"]) for row in clean]
    x_mean, y_mean = statistics.mean(x), statistics.mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator if denominator else 0.0
    edge = max(2, len(clean) // 5)
    backlog_growth = statistics.mean(y[-edge:]) - statistics.mean(y[:edge])
    return slope, backlog_growth


async def run_formal(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds, connect=30, sock_read=args.request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit=4096, limit_per_host=4096)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as client:
        prompt_bank = await build_prompt_bank(client, args.tokenize_endpoint, workload)
        health: list[dict[str, Any]] = []
        stop_health = asyncio.Event()
        monitor = asyncio.create_task(health_monitor(client, args.endpoint, stop_health, health))
        tasks: list[asyncio.Task] = []
        rng = random.Random(args.seed)
        index = 0
        transient_initial = {
            "step-up": 0.30, "step-down": 0.85,
            "ramp-up": 0.30, "held-out-composite": 0.90,
        }
        if args.arrival_process in transient_initial:
            base_rate = lambda _elapsed: args.arrival_rate * transient_initial[args.arrival_process]
        else:
            base_rate = arrival_rate_function(
                args.arrival_process, args.arrival_rate,
                on_seconds=args.on_seconds, off_seconds=args.off_seconds,
                on_multiplier=args.on_multiplier, duration=args.measurement_seconds,
            )
        index = await schedule_segment(
            client=client, endpoint=args.endpoint, prompt_bank=prompt_bank,
            workload=workload, rate_function=base_rate,
            duration=args.warmup_seconds, rng=rng, seed=args.seed,
            run_id=args.run_id, phase="warmup", tasks=tasks, start_index=index,
        )
        period_ns = args.snapshot_period_ms * 1_000_000
        measurement_start = ((time.time_ns() + period_ns - 1) // period_ns) * period_ns
        await asyncio.sleep(max(0.0, (measurement_start - time.time_ns()) / 1e9))
        measurement_end = measurement_start + int(args.measurement_seconds * 1e9)
        measurement_rate = arrival_rate_function(
            args.arrival_process, args.arrival_rate,
            on_seconds=args.on_seconds, off_seconds=args.off_seconds,
            on_multiplier=args.on_multiplier, duration=args.measurement_seconds,
        )
        await schedule_segment(
            client=client, endpoint=args.endpoint, prompt_bank=prompt_bank,
            workload=workload, rate_function=measurement_rate,
            duration=args.measurement_seconds, rng=rng, seed=args.seed,
            run_id=args.run_id, phase="measurement", tasks=tasks, start_index=index,
        )
        await asyncio.gather(*tasks)
        drain = await wait_for_drain(client, args.endpoint, args.drain_timeout_seconds)
        stop_health.set()
        await monitor
    results = [task.result() for task in tasks]
    return {
        "schema_version": "servingrom.workload_run.v1",
        "run_id": args.run_id,
        "workload": workload["name"],
        "arrival_process": args.arrival_process,
        "target_arrival_rate": args.arrival_rate,
        "seed": args.seed,
        "prompt_bank": {str(key): {name: value for name, value in entry.items() if name != "prompt"} for key, entry in prompt_bank.items()},
        "warmup_seconds": args.warmup_seconds,
        "measurement_seconds": args.measurement_seconds,
        "measurement_start_wall_ns": measurement_start,
        "measurement_end_wall_ns": measurement_end,
        "snapshot_period_ms": args.snapshot_period_ms,
        "summary": summarize(results, health, measurement_start, measurement_end),
        "drain": drain,
        "health_samples": health,
    }


async def run_calibration(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=args.request_timeout_seconds, connect=30, sock_read=args.request_timeout_seconds)
    connector = aiohttp.TCPConnector(limit=4096, limit_per_host=4096)
    candidates = [float(value) for value in args.candidate_rates.split(",")]
    results = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as client:
        prompt_bank = await build_prompt_bank(client, args.tokenize_endpoint, workload)
        for candidate_index, rate in enumerate(candidates):
            health: list[dict[str, Any]] = []
            stop_health = asyncio.Event()
            monitor = asyncio.create_task(health_monitor(client, args.endpoint, stop_health, health))
            tasks: list[asyncio.Task] = []
            rng = random.Random(args.seed + candidate_index * 100000)
            rate_function = arrival_rate_function("poisson", rate, on_seconds=1, off_seconds=1, on_multiplier=2, duration=args.measurement_seconds)
            await schedule_segment(
                client=client, endpoint=args.endpoint, prompt_bank=prompt_bank,
                workload=workload, rate_function=rate_function,
                duration=args.measurement_seconds, rng=rng, seed=args.seed,
                run_id=args.run_id, phase=f"cal-{candidate_index}", tasks=tasks, start_index=0,
            )
            await asyncio.gather(*tasks)
            drain = await wait_for_drain(client, args.endpoint, args.drain_timeout_seconds)
            stop_health.set()
            await monitor
            request_results = [task.result() for task in tasks]
            summary = summarize(request_results, health, None, None)
            slope, growth = backlog_trend(health)
            arrivals = max(summary["arrival_count"], 1)
            accepted = summary["arrival_count"] - summary["rejected_count"]
            completion_gap = abs(accepted - summary["success_count"])
            stable = (
                drain["drained"]
                and summary["rejected_count"] / arrivals <= args.reject_rate_max
                and summary["error_count"] / arrivals <= args.error_rate_max
                and completion_gap <= max(1, math.ceil(accepted * 0.01))
                and (slope <= args.backlog_slope_max or growth <= 1.0)
            )
            results.append({
                "candidate_rate": rate, "stable": stable,
                "backlog_slope_per_second": slope,
                "backlog_edge_mean_growth": growth,
                "accepted_completion_gap": completion_gap,
                "summary": summary, "drain": drain,
            })
            if not stable:
                break
    stable_rates = [row["candidate_rate"] for row in results if row["stable"]]
    if not stable_rates:
        raise RuntimeError(f"no stable calibration point for {workload['name']}")
    return {
        "schema_version": "servingrom.capacity_calibration.v1",
        "workload": workload["name"],
        "lambda_stable": max(stable_rates),
        "candidates": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("formal", "calibration"), required=True)
    parser.add_argument("--workload-config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenize-endpoint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arrival-rate", type=float, default=0.0)
    parser.add_argument("--arrival-process", default="poisson")
    parser.add_argument("--candidate-rates", default="")
    parser.add_argument("--warmup-seconds", type=float, default=180)
    parser.add_argument("--measurement-seconds", type=float, default=480)
    parser.add_argument("--drain-timeout-seconds", type=float, default=1200)
    parser.add_argument("--request-timeout-seconds", type=float, default=1200)
    parser.add_argument("--snapshot-period-ms", type=int, default=200)
    parser.add_argument("--on-seconds", type=float, default=10)
    parser.add_argument("--off-seconds", type=float, default=15)
    parser.add_argument("--on-multiplier", type=float, default=2.5)
    parser.add_argument("--reject-rate-max", type=float, default=0.01)
    parser.add_argument("--error-rate-max", type=float, default=0.0)
    parser.add_argument("--backlog-slope-max", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workload = json.loads(args.workload_config.read_text(encoding="utf-8"))
    if args.mode == "formal":
        result = asyncio.run(run_formal(args, workload))
    else:
        result = asyncio.run(run_calibration(args, workload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: result.get(key) for key in ("run_id", "workload", "lambda_stable", "summary", "drain")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
