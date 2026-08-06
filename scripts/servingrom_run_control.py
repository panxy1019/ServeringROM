#!/usr/bin/env python3
"""Atomically switch all process-local ServingROM writers inside one warm Pod."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_acks(ack_dir: Path) -> dict[str, dict]:
    values = {}
    for path in ack_dir.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            values[value["identity"]] = value
        except (OSError, ValueError, KeyError):
            continue
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("activate", "deactivate", "status"))
    parser.add_argument("--run-id")
    parser.add_argument("--experiment-id", default=os.getenv("SERVINGROM_EXPERIMENT_ID"))
    parser.add_argument("--config-id", default=os.getenv("SERVINGROM_CONFIG_ID"))
    parser.add_argument("--results-root", default=os.getenv("SERVINGROM_RESULTS_ROOT", "/servingrom-results"))
    parser.add_argument("--control-file", type=Path, default=Path(os.getenv("SERVINGROM_RUN_CONTROL_FILE", "/var/run/qwen36-pd/servingrom-run-control.json")))
    parser.add_argument("--ack-dir", type=Path, default=Path(os.getenv("SERVINGROM_RUN_CONTROL_ACK_DIR", "/var/run/qwen36-pd/servingrom-run-control-acks")))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    current = json.loads(args.control_file.read_text(encoding="utf-8"))
    if args.action == "status":
        print(json.dumps({"control": current, "acks": list(read_acks(args.ack_dir).values())}, indent=2))
        return 0
    if not args.run_id:
        parser.error("--run-id is required for activate/deactivate")
    if not args.experiment_id or not args.config_id:
        parser.error("experiment-id and config-id are required")
    expected = set(read_acks(args.ack_dir))
    generation = int(current["generation"]) + 1
    run_root = str(Path(args.results_root) / args.experiment_id / args.run_id)
    if args.action == "activate" and Path(run_root).exists():
        raise RuntimeError(f"refusing to reuse existing run path: {run_root}")
    control = {
        "generation": generation,
        "active": args.action == "activate",
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "config_id": args.config_id,
        "run_root": run_root,
        "changed_wall_ns": time.time_ns(),
    }
    atomic_json(args.control_file, control)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        acks = read_acks(args.ack_dir)
        pending = [identity for identity in expected if acks.get(identity, {}).get("generation") != generation]
        failed = [value for value in acks.values() if value.get("generation") == generation and (value.get("error") or not value.get("close_ok", False))]
        if failed:
            raise RuntimeError(f"run-control ACK failure: {failed}")
        if not pending:
            print(json.dumps({"control": control, "acknowledged": len(expected), "acks": list(acks.values())}, indent=2))
            return 0
        time.sleep(0.1)
    raise TimeoutError(f"run-control generation {generation} pending ACKs: {pending}")


if __name__ == "__main__":
    raise SystemExit(main())
