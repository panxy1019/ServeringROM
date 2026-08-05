#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingrom_pipeline.proxy_lifecycle_builder import build_proxy_lifecycle


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Proxy trace and attempt lifecycle Parquet.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    analysis = build_proxy_lifecycle(args.run_root)
    print(json.dumps(analysis.metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
