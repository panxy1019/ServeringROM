#!/usr/bin/env python3
"""Sample vLLM, process, proxy, and Ascend NPU state as JSON Lines."""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request


PORTS = {"prefill": 13700, "decode_a": 13701, "decode_b": 13702}
METRICS = (
    "num_requests_running",
    "num_requests_waiting",
    "kv_cache_usage_perc",
    "prompt_tokens_total",
    "generation_tokens_total",
    "request_success_total",
    "external_prefix_cache_hits_total",
    "external_prefix_cache_queries_total",
)


def fetch_text(url, timeout=2.0):
    with request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_prometheus(text):
    values = {name: 0.0 for name in METRICS}
    waiting_reasons = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"vllm:([^\s{]+)(?:\{([^}]*)\})?\s+([^\s]+)$", line)
        if not match:
            continue
        name, labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if name in values:
            values[name] += value
        elif name == "num_requests_waiting_by_reason":
            reason = re.search(r'reason="([^"]+)"', labels or "")
            waiting_reasons[reason.group(1) if reason else "unknown"] = value
    values["waiting_by_reason"] = waiting_reasons
    return values


def read_service_roots():
    roots = {}
    for name in ("prefill", "decode-a", "decode-b", "proxy"):
        path = Path(f"/var/run/qwen36-pd/{name}.pid")
        if path.exists():
            try:
                roots[name.replace("-", "_")] = int(path.read_text().strip())
            except ValueError:
                pass
    return roots


def process_stats():
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,pcpu=,rss=,comm="], text=True
    )
    rows = {}
    children = {}
    for line in output.splitlines():
        parts = line.split(None, 4)
        if len(parts) != 5:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        rows[pid] = {"ppid": ppid, "cpu_percent": float(parts[2]), "rss_kib": int(parts[3])}
        children.setdefault(ppid, []).append(pid)

    result = {}
    for service, root in read_service_roots().items():
        pending = [root]
        pids = []
        while pending:
            pid = pending.pop()
            if pid in pids:
                continue
            pids.append(pid)
            pending.extend(children.get(pid, ()))
        existing = [rows[pid] for pid in pids if pid in rows]
        result[service] = {
            "root_pid": root,
            "process_count": len(existing),
            "cpu_percent": round(sum(row["cpu_percent"] for row in existing), 2),
            "rss_mib": round(sum(row["rss_kib"] for row in existing) / 1024, 2),
        }
    return result


def npu_stats():
    output = subprocess.check_output(["npu-smi", "info"], text=True, errors="replace")
    result = {}
    pattern = re.compile(
        r"^\|\s*\d+\s+(\d+)\s+\|\s*[0-9A-Fa-f:.]+\s+\|"
        r"\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        phy_id, aicore, hbm_used, hbm_total = (int(value) for value in match.groups())
        if 2 <= phy_id <= 7:
            result[str(phy_id)] = {
                "aicore_percent": aicore,
                "hbm_used_mib": hbm_used,
                "hbm_total_mib": hbm_total,
            }
    return result


def sample():
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monotonic_seconds": time.monotonic(),
        "engines": {},
    }
    errors = []
    for name, port in PORTS.items():
        try:
            record["engines"][name] = parse_prometheus(
                fetch_text(f"http://127.0.0.1:{port}/metrics")
            )
        except Exception as exc:  # Keep the time series alive across transient scrape failures.
            errors.append(f"{name}: {exc!r}")
    try:
        record["proxy"] = json.loads(fetch_text("http://127.0.0.1:8080/healthcheck"))
    except Exception as exc:
        errors.append(f"proxy: {exc!r}")
    try:
        record["processes"] = process_stats()
    except Exception as exc:
        errors.append(f"processes: {exc!r}")
    try:
        record["npus"] = npu_stats()
    except Exception as exc:
        errors.append(f"npus: {exc!r}")
    if errors:
        record["errors"] = errors
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    count = 0
    while args.max_samples <= 0 or count < args.max_samples:
        started = time.monotonic()
        print(json.dumps(sample(), ensure_ascii=False), flush=True)
        count += 1
        time.sleep(max(0.0, args.interval - (time.monotonic() - started)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

