#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from servingrom_pipeline.snapshot_validation import validate_snapshots, write_snapshot_quality

parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
args = parser.parse_args()
result = validate_snapshots(args.run_root)
write_snapshot_quality(args.run_root, result)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["valid"] else 2)
