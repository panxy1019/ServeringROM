#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-ids", default="10,11,12,13,14,15")
    parser.add_argument("--output", default="/var/run/qwen36-pd/npu-mapping.json")
    args = parser.parse_args()

    wanted = {item.strip() for item in args.physical_ids.split(",") if item.strip()}
    output = subprocess.run(
        ["npu-smi", "info", "-m"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    devices: dict[str, dict[str, object]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[-1] != "Ascend910":
            continue
        npu_id, chip_id, logical_id, physical_id, _ = fields
        if physical_id not in wanted:
            continue
        device_path = Path(f"/dev/davinci{logical_id}")
        if not device_path.exists():
            raise RuntimeError(f"{device_path} does not exist for Phy-ID {physical_id}")
        devices[physical_id] = {
            "physical_id": int(physical_id),
            "logical_id": int(logical_id),
            "npu_id": int(npu_id),
            "chip_id": int(chip_id),
            "device_path": str(device_path),
            "device_realpath": os.path.realpath(device_path),
        }

    missing = wanted - devices.keys()
    if missing:
        raise RuntimeError(f"missing physical NPU mappings: {sorted(missing)}")

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"devices": devices, "raw_npu_smi_mapping": output},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(devices, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
