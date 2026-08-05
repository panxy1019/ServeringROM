#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingrom_telemetry.run_metadata import RunLayout, build_sha256_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the SHA256 manifest for a drained run.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    layout = RunLayout(root, root.parent.name, root.name)
    print(json.dumps(build_sha256_manifest(layout), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
