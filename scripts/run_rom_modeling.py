#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingrom_modeling.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stop-after", choices=("step9", "step10", "step11", "step12", "step13"), default="step13")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["config_path"] = str(args.config.resolve())
    result = run_pipeline(args.dataset_root, args.index_dir, args.output_root, config, args.stop_after)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
