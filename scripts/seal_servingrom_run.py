#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from servingrom_pipeline.snapshot_validation import seal_run

parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
args = parser.parse_args()
result = seal_run(args.run_root)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["status"] == "SEALED" else 2)
