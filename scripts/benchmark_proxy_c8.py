#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import httpx


def proxy_cpu_seconds() -> float | None:
    try:
        pid = int(Path("/var/run/qwen36-pd/proxy.pid").read_text().strip())
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def prompt(words: int) -> str:
    return " ".join(f"token{index % 997}" for index in range(words))


async def one_request(client: httpx.AsyncClient, index: int, args) -> dict:
    payload = {
        "model": args.model,
        "prompt": prompt(args.input_words),
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "seed": args.seed + index,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter_ns()
    first_content_ns = None
    text = ""
    usage = {}
    status_code = None
    async with client.stream("POST", "/v1/completions", json=payload) as response:
        status_code = response.status_code
        if response.status_code != 200:
            await response.aread()
            finished = time.perf_counter_ns()
            return {
                "index": index,
                "status_code": status_code,
                "ttft_ms": None,
                "e2e_ms": (finished - started) / 1e6,
                "output_tokens": 0,
                "output_sha256": None,
            }
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                continue
            item = json.loads(body)
            usage = item.get("usage") or usage
            choices = item.get("choices") or []
            if choices:
                content = choices[0].get("text") or ""
                if content and first_content_ns is None:
                    first_content_ns = time.perf_counter_ns()
                text += content
    finished = time.perf_counter_ns()
    return {
        "index": index,
        "status_code": status_code,
        "ttft_ms": (first_content_ns - started) / 1e6 if first_content_ns else None,
        "e2e_ms": (finished - started) / 1e6,
        "output_tokens": usage.get("completion_tokens"),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


async def run(args) -> dict:
    semaphore = asyncio.Semaphore(args.concurrency)
    cpu_before = proxy_cpu_seconds()
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        async def limited(index: int) -> dict:
            async with semaphore:
                return await one_request(client, index, args)

        results = await asyncio.gather(*(limited(index) for index in range(args.requests)))
    wall = time.perf_counter() - started
    cpu_after = proxy_cpu_seconds()
    ttft = [row["ttft_ms"] for row in results if row["ttft_ms"] is not None]
    e2e = [row["e2e_ms"] for row in results]
    output_tokens = sum(int(row["output_tokens"] or 0) for row in results)
    return {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "input_words": args.input_words,
        "requested_output_tokens": args.output_tokens,
        "wall_seconds": wall,
        "requests_per_second": args.requests / wall,
        "output_tokens_per_second": output_tokens / wall,
        "proxy_cpu_seconds": (
            cpu_after - cpu_before if cpu_before is not None and cpu_after is not None else None
        ),
        "proxy_cpu_cores_average": (
            (cpu_after - cpu_before) / wall
            if cpu_before is not None and cpu_after is not None
            else None
        ),
        "ttft_ms": {
            "p50": percentile(ttft, 0.50),
            "p95": percentile(ttft, 0.95),
            "p99": percentile(ttft, 0.99),
            "mean": statistics.mean(ttft) if ttft else None,
        },
        "e2e_ms": {
            "p50": percentile(e2e, 0.50),
            "p95": percentile(e2e, 0.95),
            "p99": percentile(e2e, 0.99),
            "mean": statistics.mean(e2e),
        },
        "status_counts": {
            str(code): sum(row["status_code"] == code for row in results)
            for code in sorted({row["status_code"] for row in results})
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen36-27b-w8a8")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--input-words", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
