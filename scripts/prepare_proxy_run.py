#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingrom_telemetry.run_metadata import RunLayout, write_run_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an immutable ServingROM Proxy run layout.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--deployment-yaml", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_json.read_text(encoding="utf-8"))
    layout = RunLayout.create(args.results_root, run["experiment_id"], run["run_id"])
    write_run_metadata(
        layout,
        run,
        deployment_yaml=args.deployment_yaml.read_text(encoding="utf-8"),
    )
    print(layout.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
