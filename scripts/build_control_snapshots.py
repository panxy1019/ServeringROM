#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from servingrom_pipeline.control_snapshot import build_control_snapshots

parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
args = parser.parse_args()
print(json.dumps(build_control_snapshots(args.run_root), indent=2))
