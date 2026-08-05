#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys


TARGET_NPU_IDS = {5, 6, 7}


def main() -> None:
    output = subprocess.run(
        ["npu-smi", "info"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    occupied: list[tuple[int, int, int, str]] = []
    in_process_table = False
    row = re.compile(
        r"^\|\s*(\d+)\s+(\d+)\s+\|\s*(\d+)\s+\|\s*([^|]+?)\s+\|"
    )
    for line in output.splitlines():
        if "Process id" in line and "Process memory" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        match = row.match(line)
        if not match:
            continue
        npu_id, chip_id, pid, process_name = match.groups()
        if int(npu_id) in TARGET_NPU_IDS:
            occupied.append(
                (int(npu_id), int(chip_id), int(pid), process_name.strip())
            )

    if occupied:
        print("目标板号 NPU 5/6/7 正在被占用：", file=sys.stderr)
        for npu_id, chip_id, pid, name in occupied:
            print(
                f"  NPU={npu_id} chip={chip_id} pid={pid} process={name}",
                file=sys.stderr,
            )
        raise SystemExit(2)

    print("PASS: 板号 NPU 5/6/7 均无运行进程，可以启动 PD Worker。")


if __name__ == "__main__":
    main()
