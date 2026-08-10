#!/usr/bin/env python3
"""Idempotently return Control-v1 routing to its frozen baseline."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid


ACTUATOR = "decode_routing_ratio"


def request_json(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    payload = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=payload, method=method)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--label", default="campaign-safety")
    args = parser.parse_args()

    status, state = request_json("GET", f"{args.endpoint}/servingrom/control/state")
    if status != 200:
        raise RuntimeError(f"control state failed: {status} {state}")
    if state["control_mode"] == "baseline":
        print(json.dumps({"accepted": True, "already_baseline": True, "state": state}))
        return 0

    command = {
        "control_command_id": f"{args.label}-rollback-{uuid.uuid4().hex[:12]}",
        "control_generation": int(state["control_generation"]) + 1,
        "actuator_name": ACTUATOR,
        "requested_value": "baseline",
        "expected_current_value": state["effective_rho_A"],
        "requested_wall_ns": time.time_ns(),
    }
    status, result = request_json(
        "POST", f"{args.endpoint}/servingrom/control/rollback", command
    )
    if status != 200 or not result.get("accepted"):
        raise RuntimeError(f"control rollback failed: {status} {result}")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
