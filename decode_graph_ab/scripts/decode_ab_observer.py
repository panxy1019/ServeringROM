#!/usr/bin/env python3
"""One-second PD telemetry without invoking ps or walking unrelated processes."""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

PORTS = {"prefill": 13700, "decode_a": 13701, "decode_b": 13702}
METRICS = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
           "prompt_tokens_total", "generation_tokens_total", "request_success_total")
NPU_IDS = tuple(range(10, 16))
CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
previous_ticks = {}
previous_at = None


def fetch(url):
    with urlopen(url, timeout=2) as response:
        return response.read().decode(errors="replace")


def prometheus(text):
    out = {key: 0.0 for key in METRICS}
    for line in text.splitlines():
        match = re.match(r"vllm:([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)$", line)
        if match and match.group(1) in out:
            try: out[match.group(1)] += float(match.group(2))
            except ValueError: pass
    return out


def descendants(root):
    found, pending = set(), [root]
    while pending:
        pid = pending.pop()
        if pid in found or not Path(f"/proc/{pid}").exists(): continue
        found.add(pid)
        try:
            raw = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            pending.extend(int(item) for item in raw)
        except (OSError, ValueError): pass
    return found


def proc_values(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        ticks = int(fields[13]) + int(fields[14])
        status = Path(f"/proc/{pid}/status").read_text()
        rss = re.search(r"^VmRSS:\s+(\d+)", status, re.M)
        return ticks, int(rss.group(1)) if rss else 0
    except (OSError, ValueError, IndexError):
        return 0, 0


def process_stats(now):
    global previous_at
    elapsed = now - previous_at if previous_at else None
    result = {}
    for service in ("prefill", "decode-a", "decode-b", "proxy"):
        path = Path(f"/var/run/qwen36-pd/{service}.pid")
        if not path.exists(): continue
        root = int(path.read_text())
        pids = descendants(root)
        ticks = rss = 0
        for pid in pids:
            pticks, prss = proc_values(pid); ticks += pticks; rss += prss
        old = previous_ticks.get(service)
        cpu = ((ticks - old) / CLK_TCK / elapsed * 100) if old is not None and elapsed else None
        previous_ticks[service] = ticks
        result[service.replace("-", "_")] = {
            "root_pid": root, "process_count": len(pids),
            "cpu_percent": round(cpu, 2) if cpu is not None else None,
            "rss_mib": round(rss / 1024, 2),
        }
    previous_at = now
    return result


def npu_stats():
    output = subprocess.check_output(["npu-smi", "info"], text=True, errors="replace", timeout=8)
    result = {}
    pattern = re.compile(r"^\|\s*\d+\s+(\d+)\s+\|\s*[0-9A-Fa-f:.]+\s+\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)")
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            phy, core, used, total = map(int, match.groups())
            if phy in NPU_IDS:
                result[str(phy)] = {"aicore_percent": core, "hbm_used_mib": used, "hbm_total_mib": total}
    return result


def sample():
    now = time.monotonic()
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "monotonic_seconds": now, "engines": {}}
    errors = []
    for name, port in PORTS.items():
        try: record["engines"][name] = prometheus(fetch(f"http://127.0.0.1:{port}/metrics"))
        except Exception as exc: errors.append(f"{name}: {exc!r}")
    try: record["proxy"] = json.loads(fetch("http://127.0.0.1:8080/healthcheck"))
    except Exception as exc: errors.append(f"proxy: {exc!r}")
    try: record["processes"] = process_stats(now)
    except Exception as exc: errors.append(f"processes: {exc!r}")
    try: record["npus"] = npu_stats()
    except Exception as exc: errors.append(f"npus: {exc!r}")
    if errors: record["errors"] = errors
    return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    while True:
        started = time.monotonic(); print(json.dumps(sample(), ensure_ascii=False), flush=True)
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__": main()
