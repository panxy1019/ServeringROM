#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    event_counts: Counter[str] = Counter()
    engine_counts: Counter[str] = Counter()
    request_ids: set[str] = set()
    damaged_lines = 0
    capability_markers = []
    for path in sorted((args.run_root / "raw" / "mooncake").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                damaged_lines += 1
                continue
            event_type = event["event_type"]
            event_counts[event_type] += 1
            payload = event.get("payload", {})
            if event_type == "kv_transfer_runtime_capability":
                capability_markers.append(payload)
            if event_type in {
                "kv_transfer_enqueued",
                "kv_transfer_started",
                "kv_transfer_completed",
                "kv_transfer_failed",
            }:
                request_ids.add(event.get("request_id"))
                engine_counts[payload.get("engine_instance")] += 1
    result = {
        "event_counts": dict(event_counts),
        "engine_event_counts": dict(engine_counts),
        "request_id_count": len(request_ids - {None}),
        "damaged_lines": damaged_lines,
        "capability_marker_count": len(capability_markers),
        "capability_markers": capability_markers,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
