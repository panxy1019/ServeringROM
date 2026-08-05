#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--allow-violations", action="store_true")
    args = parser.parse_args()
    report = json.loads((args.run_root / "reports" / "internal_data_quality.json").read_text())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if args.allow_violations or report["violation_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
