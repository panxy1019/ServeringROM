#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from servingrom_pipeline.internal_event_reader import read_internal_events
from servingrom_pipeline.internal_reconstruction import reconstruct_internal_tables, write_internal_parquet
from servingrom_pipeline.internal_validation import validate_internal_data, write_internal_quality_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    dataset = read_internal_events(args.run_root)
    tables = reconstruct_internal_tables(args.run_root, dataset)
    write_internal_parquet(args.run_root, tables)
    report = validate_internal_data(args.run_root, tables, dataset)
    write_internal_quality_report(args.run_root, report)
    print(json.dumps({"table_counts": report["table_counts"], "violation_count": report["violation_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
