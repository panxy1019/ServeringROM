#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from servingrom_telemetry.schema import validate_event


def input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.glob("*.jsonl")) if path.is_dir() else [path])
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ServingROM JSONL files")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    errors: list[str] = []
    error_count = 0
    last_sequence: dict[tuple[str, str], int] = {}
    event_count = 0
    for path in input_files(args.paths):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    event = json.loads(line)
                    validate_event(event)
                    process_key = (event["run_id"], event["process_instance_id"])
                    previous = last_sequence.get(process_key)
                    last_sequence[process_key] = event["event_seq"]
                    if previous is None and event["event_seq"] != 1:
                        raise ValueError(
                            f"first event_seq for {process_key} must be 1, "
                            f"got {event['event_seq']}"
                        )
                    if previous is not None and event["event_seq"] != previous + 1:
                        raise ValueError(
                            f"event_seq discontinuity for {process_key}: "
                            f"expected {previous + 1}, got {event['event_seq']}"
                        )
                    event_count += 1
                except Exception as exc:
                    error_count += 1
                    if len(errors) < args.max_errors:
                        errors.append(f"{path}:{line_number}: {exc}")

    report = {
        "files": len(input_files(args.paths)),
        "events": event_count,
        "malformed_or_invalid": error_count,
        "errors": errors,
        "processes": len(last_sequence),
        "valid": error_count == 0,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
