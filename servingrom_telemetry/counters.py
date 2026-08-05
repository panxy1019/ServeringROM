from __future__ import annotations

import threading
from collections.abc import Iterable


COUNTER_NAMES = (
    "events_attempted",
    "events_enqueued",
    "events_written",
    "events_dropped_queue_full",
    "events_dropped_writer_failed",
    "serialization_errors",
    "event_build_errors",
    "write_errors",
    "flush_errors",
    "queue_depth_current",
    "queue_depth_high_watermark",
    "writer_batches",
    "writer_bytes",
    "writer_write_latency_ns_total",
    "writer_write_latency_ns_max",
)


class TelemetryCounters:
    __slots__ = ("_lock", "_values")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = {name: 0 for name in COUNTER_NAMES}

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._values[name] += amount

    def update_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._values["queue_depth_current"] = depth
            if depth > self._values["queue_depth_high_watermark"]:
                self._values["queue_depth_high_watermark"] = depth

    def record_write_latency(self, latency_ns: int) -> None:
        with self._lock:
            self._values["writer_write_latency_ns_total"] += latency_ns
            self._values["writer_write_latency_ns_max"] = max(
                self._values["writer_write_latency_ns_max"], latency_ns
            )

    def add_many(self, updates: Iterable[tuple[str, int]]) -> None:
        with self._lock:
            for name, amount in updates:
                self._values[name] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._values)
