#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    args = parser.parse_args()
    payload = {
        "model": "qwen36-27b-w8a8",
        "messages": [{"role": "user", "content": "请用三句话解释勾股定理。"}],
        "max_tokens": 64,
        "temperature": 0,
        "seed": 1024,
        "stream": False,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=600, trust_env=False) as client:
        response = client.post(
            f"{args.endpoint.rstrip('/')}/v1/chat/completions",
            headers={"X-Request-Id": args.request_id},
            json=payload,
        )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    result = {
        "status_code": response.status_code,
        "request_id": args.request_id,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "output_tokens": body.get("usage", {}).get("completion_tokens"),
        "elapsed_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
