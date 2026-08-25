#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from servingrom_control_modeling.phase_transition import run_phase_transition_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ServingROM Step 15C-2B.3")
    for name in ("dataset-root", "forcing-root", "outflow-root", "transition-root", "age-root", "output-root", "config"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = run_phase_transition_pipeline(
        dataset_root=args.dataset_root,
        forcing_root=args.forcing_root,
        outflow_root=args.outflow_root,
        transition_root=args.transition_root,
        age_root=args.age_root,
        output_root=args.output_root,
        config=json.loads(args.config.read_text(encoding="utf-8")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
