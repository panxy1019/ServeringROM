#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


def request_json(method: str, url: str, value=None):
    data = None if value is None else json.dumps(value).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("state", "prepare", "commit", "rollback"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--command-id")
    parser.add_argument("--generation", type=int)
    parser.add_argument("--value")
    parser.add_argument("--expected")
    args = parser.parse_args()
    if args.action == "state":
        status, value = request_json("GET", f"{args.base_url}/servingrom/control/state")
    else:
        if not all((args.command_id, args.generation is not None, args.value is not None, args.expected is not None)):
            parser.error("control actions require --command-id, --generation, --value and --expected")
        def scalar(value: str):
            try:
                return float(value)
            except ValueError:
                return value
        command = {
            "control_command_id": args.command_id,
            "control_generation": args.generation,
            "actuator_name": "decode_routing_ratio",
            "requested_value": scalar(args.value),
            "expected_current_value": scalar(args.expected),
            "requested_wall_ns": time.time_ns(),
        }
        status, value = request_json(
            "POST", f"{args.base_url}/servingrom/control/{args.action}", command
        )
    print(json.dumps({"http_status": status, "response": value}, ensure_ascii=False, indent=2))
    return 0 if status < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
