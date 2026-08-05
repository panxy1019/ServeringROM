#!/usr/bin/env python3
"""Sample Linux /proc counters for selected PIDs without installing sysstat."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def read_status(pid: int) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key] = value.strip()
    return values


def read_sample(pid: int) -> dict[str, int | str]:
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    status = read_status(pid)
    io_values: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/io").read_text().splitlines():
        key, value = line.split(":", 1)
        io_values[key] = int(value.strip())
    return {
        "pid": pid,
        "comm": stat[1].strip("()"),
        "state": stat[2],
        "minflt": int(stat[9]),
        "majflt": int(stat[11]),
        "utime_ticks": int(stat[13]),
        "stime_ticks": int(stat[14]),
        "threads": int(stat[19]),
        "vmsize_kb": int(status.get("VmSize", "0 kB").split()[0]),
        "vmrss_kb": int(status.get("VmRSS", "0 kB").split()[0]),
        "voluntary_ctxt": int(status.get("voluntary_ctxt_switches", "0")),
        "nonvoluntary_ctxt": int(status.get("nonvoluntary_ctxt_switches", "0")),
        "read_bytes": io_values.get("read_bytes", 0),
        "write_bytes": io_values.get("write_bytes", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pids", nargs="+", type=int)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    before = {pid: read_sample(pid) for pid in args.pids if Path(f"/proc/{pid}").exists()}
    started = time.monotonic()
    time.sleep(args.interval)
    elapsed = time.monotonic() - started
    result = []
    for pid, first in before.items():
        try:
            current = read_sample(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        current["interval_seconds"] = round(elapsed, 6)
        current["cpu_user_percent"] = round(
            100 * (int(current["utime_ticks"]) - int(first["utime_ticks"])) / hz / elapsed,
            3,
        )
        current["cpu_system_percent"] = round(
            100 * (int(current["stime_ticks"]) - int(first["stime_ticks"])) / hz / elapsed,
            3,
        )
        for key in (
            "minflt",
            "majflt",
            "voluntary_ctxt",
            "nonvoluntary_ctxt",
            "read_bytes",
            "write_bytes",
        ):
            current[f"delta_{key}"] = int(current[key]) - int(first[key])
        result.append(current)
    print(json.dumps({"time": time.time(), "processes": result}, ensure_ascii=True))


if __name__ == "__main__":
    main()
