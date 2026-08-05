#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
import urllib.request
from pathlib import Path
from typing import Any

from servingrom_telemetry import create_emitter


PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+0-9.eE]+)$"
)
NPU_METRIC_HINTS = ("npu", "aicore", "hbm", "device_memory")


def read_process_metrics(pid: int, clock_ticks: int, page_size: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text(encoding="utf-8").split()
        status = (root / "status").read_text(encoding="utf-8").splitlines()
        io_lines = (root / "io").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return {"pid": pid, "available": False}

    status_map = {
        key.rstrip(":"): value.strip()
        for line in status
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    io_map = {
        key.rstrip(":"): int(value.strip())
        for line in io_lines
        if ":" in line
        for key, value in [line.split(":", 1)]
    }
    return {
        "pid": pid,
        "available": True,
        "state": stat[2],
        "ppid": int(stat[3]),
        "cpu_user_seconds": int(stat[13]) / clock_ticks,
        "cpu_system_seconds": int(stat[14]) / clock_ticks,
        "rss_bytes": int(stat[23]) * page_size,
        "vm_rss": status_map.get("VmRSS"),
        "vm_hwm": status_map.get("VmHWM"),
        "threads": int(status_map.get("Threads", "0")),
        "read_bytes": io_map.get("read_bytes"),
        "write_bytes": io_map.get("write_bytes"),
    }


def read_network_metrics() -> dict[str, dict[str, int]]:
    metrics: dict[str, dict[str, int]] = {}
    try:
        lines = Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]
    except (FileNotFoundError, PermissionError):
        return metrics
    for line in lines:
        interface, values = line.split(":", 1)
        fields = values.split()
        metrics[interface.strip()] = {
            "rx_bytes": int(fields[0]),
            "rx_packets": int(fields[1]),
            "rx_errors": int(fields[2]),
            "tx_bytes": int(fields[8]),
            "tx_packets": int(fields[9]),
            "tx_errors": int(fields[10]),
        }
    return metrics


def read_exporter_metrics(url: str | None, timeout_s: float) -> tuple[dict[str, Any], str | None]:
    if not url:
        return {}, "exporter_url_not_configured"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    selected: dict[str, Any] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        match = PROM_LINE.match(line)
        if not match or not any(hint in match["name"].lower() for hint in NPU_METRIC_HINTS):
            continue
        key = match["name"]
        if match["labels"]:
            key = f"{key}{{{match['labels']}}}"
        try:
            selected[key] = float(match["value"])
        except ValueError:
            selected[key] = None
    return selected, None


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", action="append", type=int, default=[])
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--exporter-url", default=os.getenv("SERVINGROM_NPU_EXPORTER_URL"))
    parser.add_argument("--capability-file", type=Path)
    args = parser.parse_args()
    if args.interval_ms < 50:
        parser.error("--interval-ms must be at least 50")

    emitter = create_emitter()
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    clock_ticks = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    capability_file = args.capability_file or (
        Path(os.getenv("SERVINGROM_OUTPUT_DIR", ".")).parent.parent
        / "metadata"
        / "device_telemetry_capabilities.json"
    )
    capabilities = {
        "sampling_interval_ms": args.interval_ms,
        "collector_pid": os.getpid(),
        "process_metrics": "procfs",
        "network_metrics": "procfs",
        "npu_metrics": "prometheus_exporter" if args.exporter_url else None,
        "npu_metrics_unavailable_reason": (
            None if args.exporter_url else "SERVINGROM_NPU_EXPORTER_URL not configured"
        ),
        "forbidden_subprocess_collectors": ["ps", "npu-smi"],
    }
    atomic_json(capability_file, capabilities)

    interval_s = args.interval_ms / 1000.0
    next_sample = time.monotonic()
    while not stopped:
        sample_begin = time.monotonic_ns()
        npu_metrics, exporter_error = read_exporter_metrics(
            args.exporter_url, min(interval_s * 0.8, 0.15)
        )
        emitter.emit(
            "device_metric",
            {
                "sample_interval_ms": args.interval_ms,
                "processes": [
                    read_process_metrics(pid, clock_ticks, page_size) for pid in args.pid
                ],
                "network": read_network_metrics(),
                "npu_metrics": npu_metrics or None,
                "npu_exporter_error": exporter_error,
                "collector_duration_ns": time.monotonic_ns() - sample_begin,
            },
        )
        next_sample += interval_s
        time.sleep(max(0.0, next_sample - time.monotonic()))

    emitter.close(10.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
