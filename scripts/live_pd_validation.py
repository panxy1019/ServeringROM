#!/usr/bin/env python3
"""Live checks for fair Decode routing and token-budget admission."""

import argparse
import asyncio
import json
import time

import httpx


def prompt(token_count: int) -> str:
    return " ".join(f"token{i % 997}" for i in range(token_count))


async def request(
    client: httpx.AsyncClient,
    *,
    input_tokens: int,
    output_tokens: int,
    use_token_ids: bool = False,
):
    started = time.perf_counter()
    response = await client.post(
        "/v1/completions",
        json={
            "model": "qwen36-27b-w8a8",
            # Token IDs make the admission experiment exact. The routing test
            # deliberately keeps text input to exercise chat/completion tokenization.
            "prompt": [42] * input_tokens if use_token_ids else prompt(input_tokens),
            "max_tokens": output_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
    )
    return {
        "status_code": response.status_code,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
    }


async def main_async(args):
    async with httpx.AsyncClient(base_url=args.base_url, timeout=args.timeout) as client:
        if args.mode == "sequential-route":
            results = []
            for _ in range(args.count):
                results.append(await request(client, input_tokens=128, output_tokens=16))
        else:
            results = await asyncio.gather(
                *(
                    request(
                        client,
                        input_tokens=4096,
                        output_tokens=16,
                        use_token_ids=True,
                    )
                    for _ in range(args.count)
                )
            )
    summary = {
        "mode": args.mode,
        "count": args.count,
        "status_counts": {
            str(code): sum(result["status_code"] == code for result in results)
            for code in sorted({result["status_code"] for result in results})
        },
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sequential-route", "prefill-admission"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
