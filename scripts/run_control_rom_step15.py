#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingrom_control_modeling import run_control_rom_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit the frozen Step 15 control-aware ROM")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--config",
        default=Path("configs/servingrom_control_rom_v1.json"),
        type=Path,
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_control_rom_pipeline(args.dataset_root, args.output_root, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
