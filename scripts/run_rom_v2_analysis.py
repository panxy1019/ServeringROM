#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingrom_v2 import run_v2_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only ServingROM v2 failure attribution and view redesign.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--modeling-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_v2_analysis(args.dataset_root, args.modeling_root, args.output_root, config)
    print(json.dumps({
        "status": result["status"],
        "elapsed_seconds": result["elapsed_seconds"],
        "selected_state_candidate": result["selected_state_candidate"],
        "selected_output_candidate": result["selected_output_candidate"],
        "single_rate_feasible": result["single_rate_feasible"],
        "reports": result["reports"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
