#!/usr/bin/env python3
"""Open-loop mixed-shape load generator for the local 1P2D proxy."""

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import defaultdict

import httpx


SHAPES = {
    "short": (128, 16),
    "balanced": (512, 128),
    "prefill_heavy": (4096, 16),
    "decode_heavy": (512, 512),
}


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percent / 100))
    return ordered[index]


def prompt_for_tokens(count: int) -> str:
    # The exact token count is intentionally left to the proxy tokenizer. This
    # produces a stable, low-entropy prompt close to the requested scale.
    return " ".join(f"token{i % 997}" for i in range(count))


async def one_request(client, shape, model, timeout):
    input_tokens, output_tokens = SHAPES[shape]
    payload = {
        "model": model,
        "prompt": prompt_for_tokens(input_tokens),
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    started = time.perf_counter()
    response = await client.post("/v1/completions", json=payload, timeout=timeout)
    elapsed = (time.perf_counter() - started) * 1000
    if response.status_code == 429:
        return shape, "rejected", elapsed, response.json()
    response.raise_for_status()
    usage = response.json().get("usage", {})
    return shape, "success", elapsed, usage


async def run(args):
    randomizer = random.Random(args.seed)
    sequence = [name for name in SHAPES for _ in range(args.per_shape)]
    randomizer.shuffle(sequence)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    results = []

    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        async def scheduled(index, shape):
            await asyncio.sleep(index / args.request_rate)
            async with semaphore:
                try:
                    return await one_request(client, shape, args.model, args.timeout)
                except Exception as exc:
                    return shape, "failed", None, repr(exc)

        results = await asyncio.gather(*(scheduled(i, shape) for i, shape in enumerate(sequence)))

    by_shape = defaultdict(list)
    states = defaultdict(int)
    states_by_shape = defaultdict(lambda: defaultdict(int))
    for shape, state, elapsed, _ in results:
        states[state] += 1
        states_by_shape[shape][state] += 1
        if elapsed is not None:
            by_shape[shape].append(elapsed)
    summary = {
        "request_rate": args.request_rate,
        "max_concurrency": args.max_concurrency,
        "per_shape": args.per_shape,
        "states": dict(states),
        "states_by_shape": {
            shape: dict(shape_states)
            for shape, shape_states in sorted(states_by_shape.items())
        },
        "shapes": {
            shape: {
                "count": len(values),
                "e2e_ms_p50": round(percentile(values, 50), 3) if values else None,
                "e2e_ms_p95": round(percentile(values, 95), 3) if values else None,
                "e2e_ms_p99": round(percentile(values, 99), 3) if values else None,
                "e2e_ms_mean": round(statistics.mean(values), 3) if values else None,
            }
            for shape, values in by_shape.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if states["failed"] == 0 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen36-27b-w8a8")
    parser.add_argument("--per-shape", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
