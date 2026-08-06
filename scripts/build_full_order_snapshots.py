#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingrom_pipeline.snapshot_builder import SnapshotConfig, build_snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description="Build immutable Full-order snapshots from ServingROM events.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--period-ms", type=int, default=200)
    args = parser.parse_args()
    result = build_snapshots(args.run_root, SnapshotConfig(period_ms=args.period_ms))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
