#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


ACTUATOR = "decode_routing_ratio"


def http_json(method: str, url: str, body: Any = None, timeout_s: float = 180.0):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else None
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        return exc.code, json.loads(payload) if payload else None


def command(generation: int, value: float | str, expected: float | str, command_id: str):
    return {
        "control_command_id": command_id,
        "control_generation": generation,
        "actuator_name": ACTUATOR,
        "requested_value": value,
        "expected_current_value": expected,
        "requested_wall_ns": time.time_ns(),
    }


def apply_control(base_url: str, generation: int, value: float, expected: float | str):
    value_text = str(value).replace(".", "p")
    payload = command(generation, value, expected, f"smoke-g{generation}-rho-{value_text}-{uuid.uuid4().hex[:8]}")
    prepare_status, prepared = http_json("POST", f"{base_url}/servingrom/control/prepare", payload)
    if prepare_status != 200 or not prepared.get("accepted"):
        raise RuntimeError(f"prepare failed: {prepare_status} {prepared}")
    commit_status, committed = http_json("POST", f"{base_url}/servingrom/control/commit", payload)
    if commit_status != 200 or not committed.get("accepted"):
        raise RuntimeError(f"commit failed: {commit_status} {committed}")
    replay_status, replay = http_json("POST", f"{base_url}/servingrom/control/commit", payload)
    if replay_status != 200 or not replay.get("idempotent_replay"):
        raise RuntimeError(f"idempotent replay failed: {replay_status} {replay}")
    return {"command": payload, "prepare": prepared, "commit": committed, "commit_replay": replay}


def run_request(base_url: str, model: str, prompt: str, seed: int):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": 32,
        "stream": False,
    }
    started = time.monotonic()
    status, response = http_json("POST", f"{base_url}/v1/chat/completions", body)
    elapsed = time.monotonic() - started
    if status != 200:
        raise RuntimeError(f"completion failed: {status} {response}")
    text = response["choices"][0]["message"]["content"]
    return {
        "status": status,
        "elapsed_seconds": elapsed,
        "output_tokens": response.get("usage", {}).get("completion_tokens"),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def exercise_stage(base_url: str, model: str, name: str, prompt: str, seed: int, requests: int, hold_s: float):
    started = time.monotonic()
    samples = [run_request(base_url, model, prompt, seed) for _ in range(requests)]
    remaining = hold_s - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)
    _, state = http_json("GET", f"{base_url}/servingrom/control/state")
    hashes = sorted({sample["output_sha256"] for sample in samples})
    return {
        "name": name,
        "requests": requests,
        "wall_seconds": time.monotonic() - started,
        "latency_p50_seconds": statistics.median(sample["elapsed_seconds"] for sample in samples),
        "latency_max_seconds": max(sample["elapsed_seconds"] for sample in samples),
        "output_sha256_values": hashes,
        "state": state,
    }


def rejected_prepare(base_url: str, payload: dict[str, Any], expected_reason: str):
    status, response = http_json("POST", f"{base_url}/servingrom/control/prepare", payload)
    if status != 409 or response.get("reason") != expected_reason:
        raise RuntimeError(f"expected {expected_reason}, got {status} {response}")
    return {"http_status": status, "response": response}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen36-27b-w8a8")
    parser.add_argument("--requests-per-stage", type=int, default=20)
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompt = "用一句中文说明为什么运行时控制命令需要幂等。"
    seed = 1024
    report: dict[str, Any] = {
        "started_wall_ns": time.time_ns(),
        "base_url": args.base_url,
        "model": args.model,
        "seed": seed,
        "stages": [],
        "commands": [],
        "negative_tests": {},
    }

    _, initial = http_json("GET", f"{args.base_url}/servingrom/control/state")
    if initial["control_mode"] != "baseline":
        raise RuntimeError(f"smoke must start from baseline: {initial}")
    report["initial_state"] = initial
    report["stages"].append(exercise_stage(args.base_url, args.model, "baseline", prompt, seed, args.requests_per_stage, args.hold_seconds))

    generation = initial["control_generation"] + 1
    report["commands"].append(apply_control(args.base_url, generation, 0.5, "baseline"))
    report["negative_tests"]["duplicate_prepare"] = rejected_prepare(
        args.base_url, report["commands"][-1]["command"], "duplicate_command_id"
    )
    report["negative_tests"]["dwell"] = rejected_prepare(
        args.base_url,
        command(generation + 1, 0.3, 0.5, f"smoke-dwell-{uuid.uuid4().hex[:8]}"),
        "minimum_dwell_time_not_met",
    )
    report["stages"].append(exercise_stage(args.base_url, args.model, "rho_0.5", prompt, seed, args.requests_per_stage, args.hold_seconds))

    generation += 1
    report["commands"].append(apply_control(args.base_url, generation, 0.3, 0.5))
    report["stages"].append(exercise_stage(args.base_url, args.model, "rho_0.3", prompt, seed, args.requests_per_stage, args.hold_seconds))

    report["negative_tests"]["maximum_step"] = rejected_prepare(
        args.base_url,
        command(generation + 1, 0.7, 0.3, f"smoke-step-{uuid.uuid4().hex[:8]}"),
        "maximum_step_exceeded",
    )
    report["negative_tests"]["out_of_range"] = rejected_prepare(
        args.base_url,
        command(generation + 1, 0.9, 0.3, f"smoke-range-{uuid.uuid4().hex[:8]}"),
        "requested_value_out_of_range",
    )
    report["negative_tests"]["stale_generation"] = rejected_prepare(
        args.base_url,
        command(generation, 0.3, 0.3, f"smoke-stale-{uuid.uuid4().hex[:8]}"),
        "stale_or_nonmonotonic_generation",
    )

    generation += 1
    report["commands"].append(apply_control(args.base_url, generation, 0.5, 0.3))
    report["stages"].append(exercise_stage(args.base_url, args.model, "rho_0.5_bridge", prompt, seed, args.requests_per_stage, args.hold_seconds))
    generation += 1
    report["commands"].append(apply_control(args.base_url, generation, 0.7, 0.5))
    report["stages"].append(exercise_stage(args.base_url, args.model, "rho_0.7", prompt, seed, args.requests_per_stage, args.hold_seconds))
    generation += 1
    report["commands"].append(apply_control(args.base_url, generation, 0.5, 0.7))
    report["stages"].append(exercise_stage(args.base_url, args.model, "rho_0.5_final", prompt, seed, args.requests_per_stage, args.hold_seconds))

    status, fallback = http_json(
        "POST", f"{args.base_url}/servingrom/control/test/safety-fallback", {"reason": "decode_unhealthy_smoke"}
    )
    if status != 200 or fallback.get("effective_value") != "baseline":
        raise RuntimeError(f"safety fallback failed: {status} {fallback}")
    report["safety_fallback"] = fallback
    generation = fallback["control_generation"]
    report["stages"].append(exercise_stage(args.base_url, args.model, "safe_baseline", prompt, seed, args.requests_per_stage, args.hold_seconds))

    # Re-enter control once to prove explicit rollback, rather than treating the
    # safety fallback itself as the rollback test.
    generation += 1
    report["commands"].append(apply_control(args.base_url, generation, 0.5, "baseline"))
    report["stages"].append(exercise_stage(args.base_url, args.model, "rho_0.5_before_rollback", prompt, seed, args.requests_per_stage, args.hold_seconds))
    generation += 1
    rollback_command = command(generation, "baseline", 0.5, f"smoke-rollback-{uuid.uuid4().hex[:8]}")
    status, rollback = http_json("POST", f"{args.base_url}/servingrom/control/rollback", rollback_command)
    if status != 200 or rollback.get("effective_value") != "baseline":
        raise RuntimeError(f"rollback failed: {status} {rollback}")
    report["rollback"] = rollback
    report["stages"].append(exercise_stage(args.base_url, args.model, "rollback_baseline", prompt, seed, args.requests_per_stage, args.hold_seconds))

    hashes = sorted({value for stage in report["stages"] for value in stage["output_sha256_values"]})
    report["output_sha256_values"] = hashes
    report["output_sha256_consistent"] = len(hashes) == 1
    _, report["telemetry_health"] = http_json("GET", f"{args.base_url}/servingrom/telemetry/health")
    _, report["final_state"] = http_json("GET", f"{args.base_url}/servingrom/control/state")
    report["finished_wall_ns"] = time.time_ns()
    report["passed"] = (
        report["output_sha256_consistent"]
        and report["final_state"]["control_mode"] == "baseline"
        and report["telemetry_health"].get("events_dropped_queue_full", 0) == 0
        and report["telemetry_health"].get("events_dropped_writer_failed", 0) == 0
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(output), "hashes": hashes}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
