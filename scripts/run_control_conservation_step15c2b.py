#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servingrom_control_modeling.conservation import run_conservation_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ServingROM Step 15C-2B")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--representation-root", required=True, type=Path)
    parser.add_argument("--forcing-root", required=True, type=Path)
    parser.add_argument("--outflow-root", required=True, type=Path)
    parser.add_argument("--step15c2a-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_conservation_pipeline(
        dataset_root=args.dataset_root,
        representation_root=args.representation_root,
        forcing_root=args.forcing_root,
        outflow_root=args.outflow_root,
        step15c2a_root=args.step15c2a_root,
        output_root=args.output_root,
        config=json.loads(args.config.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
