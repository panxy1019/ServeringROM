#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


async def request(
    client: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
    *,
    cancel_after_first_chunk: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": "qwen36-27b-w8a8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1024,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_chunk = None
    chunks: list[bytes] = []
    status = None
    async with client.post(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        headers={"X-Request-Id": request_id},
        json=payload,
    ) as response:
        status = response.status
        async for chunk in response.content.iter_any():
            if first_chunk is None:
                first_chunk = time.perf_counter()
            chunks.append(chunk)
            if cancel_after_first_chunk:
                break
    ended = time.perf_counter()
    body = b"".join(chunks)
    return {
        "request_id": request_id,
        "status": status,
        "cancelled": cancel_after_first_chunk,
        "ttft_seconds": first_chunk - started if first_chunk else None,
        "e2e_seconds": ended - started,
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "response_bytes": len(body),
    }


async def run(args) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=600, connect=10, sock_read=600)
    connector = aiohttp.TCPConnector(limit=32, limit_per_host=32)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as client:
        cases = []
        cases.append(await request(client, args.endpoint, "请用一句话解释圆的半径。", 8, "phase-a-short"))
        cases.append(
            await request(
                client,
                args.endpoint,
                ("这是用于长输入 Prefill 验证的教材段落。" * 260),
                8,
                "phase-a-long",
            )
        )
        c8_started = time.perf_counter()
        c8 = await asyncio.gather(
            *[
                request(
                    client,
                    args.endpoint,
                    "请分步骤说明分数加法的基本方法。" * 25,
                    32,
                    f"phase-a-c8-{index}",
                )
                for index in range(8)
            ]
        )
        c8_wall = time.perf_counter() - c8_started
        cases.extend(c8)
        cases.append(
            await request(
                client,
                args.endpoint,
                "请简短回答：1+1等于几？",
                64,
                "phase-a-cancel",
                cancel_after_first_chunk=True,
            )
        )
        cases.append(
            await request(
                client,
                args.endpoint,
                "长上下文 Mooncake 传输验证。" * 500,
                8,
                "phase-a-mooncake-long",
            )
        )
        rejected = await request(
            client,
            args.endpoint,
            "超限输入。" * 12000,
            8,
            "phase-a-reject-429",
        )
        cases.append(rejected)
    successful_c8 = [item for item in c8 if item["status"] == 200]
    return {
        "endpoint": args.endpoint,
        "started_at": args.started_at,
        "cases": cases,
        "c8": {
            "wall_seconds": c8_wall,
            "requests_per_second": len(successful_c8) / c8_wall,
            "ttft_p50_seconds": percentile([x["ttft_seconds"] for x in successful_c8], 0.5),
            "ttft_p95_seconds": percentile([x["ttft_seconds"] for x in successful_c8], 0.95),
            "e2e_p50_seconds": percentile([x["e2e_seconds"] for x in successful_c8], 0.5),
            "e2e_p95_seconds": percentile([x["e2e_seconds"] for x in successful_c8], 0.95),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.started_at = time.time_ns()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["c8"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
