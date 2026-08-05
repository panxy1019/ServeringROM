#!/usr/bin/env python3
"""Focused behavioral checks for the custom PD proxy scheduler."""

import importlib.util
import sys
from pathlib import Path


def load_proxy():
    path = Path(__file__).with_name("pd_proxy.py")
    spec = importlib.util.spec_from_file_location("pd_proxy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    proxy = load_proxy()
    scheduler = proxy.SharedProxyScheduler(
        [("127.0.0.1", 13700)],
        [("127.0.0.1", 13701), ("127.0.0.1", 13702)],
        max_prefill_inflight_tokens=100,
    )

    selected = []
    for _ in range(4):
        backend = scheduler.pick_decoder(16)
        selected.append(backend["port"])
        scheduler.release_decoder(backend["key"], 16)
    assert selected == [13701, 13702, 13701, 13702], selected

    first = scheduler.begin_request(60)
    try:
        scheduler.begin_request(50)
    except proxy.PrefillOverloadedError:
        pass
    else:
        raise AssertionError("prefill admission did not reject the over-budget request")
    health = scheduler.healthcheck()
    assert health["prefill_inflight_tokens"] == 60, health
    assert health["prefill_admission_rejected_total"] == 1, health
    scheduler.release_prefill_kv(first["key"], 60)
    scheduler.finish_request(first["key"], 60, None, 0, release_prefill_kv=False)
    assert scheduler.healthcheck()["prefill_inflight_tokens"] == 0
    assert scheduler.healthcheck()["request_num"] == 0
    print("PASS: fair Decode ties and token-aware prefill admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
