#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an already-built Proxy lifecycle report.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--allow-violations", action="store_true")
    args = parser.parse_args()
    report_path = args.run_root / "reports" / "proxy_lifecycle_quality.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics["violation_count"] and not args.allow_violations:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
