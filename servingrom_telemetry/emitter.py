from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .async_writer import AsyncJSONLWriter
from .clock import ProcessClock
from .config import TelemetryConfig
from .counters import COUNTER_NAMES, TelemetryCounters
from .ids import EventSequence, build_process_instance_id
from .jsonl_sink import RotatingJSONLSink
from .schema import build_event


class Emitter(Protocol):
    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        trace_id: str | None = None,
        attempt_id: int | None = None,
        request_id: str | None = None,
        external_request_id: str | None = None,
    ) -> bool: ...

    def flush(self, timeout_s: float | None = None) -> bool: ...

    def close(self, timeout_s: float | None = None) -> bool: ...

    def health_snapshot(self) -> dict[str, Any]: ...


class NullEmitter:
    """Zero-infrastructure disabled path: no queue, lock, file, or thread."""

    __slots__ = ()

    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        trace_id: str | None = None,
        attempt_id: int | None = None,
        request_id: str | None = None,
        external_request_id: str | None = None,
    ) -> bool:
        return False

    def flush(self, timeout_s: float | None = None) -> bool:
        return True

    def close(self, timeout_s: float | None = None) -> bool:
        return True

    def health_snapshot(self) -> dict[str, Any]:
        return {"enabled": False, **{name: 0 for name in COUNTER_NAMES}}


SinkFactory = Callable[[Path, str, str, int], RotatingJSONLSink]


class AsyncTelemetryEmitter:
    def __init__(
        self,
        config: TelemetryConfig,
        *,
        sink_factory: SinkFactory = RotatingJSONLSink,
    ) -> None:
        if not config.enabled:
            raise ValueError("AsyncTelemetryEmitter requires enabled config")
        self._config = config
        self._clock = ProcessClock()
        self._process_id = os.getpid()
        self._process_instance_id = build_process_instance_id(
            host_id=config.host_id,
            component=config.component,
            process_id=self._process_id,
            process_start_wall_ns=self._clock.process_start_wall_ns,
            process_start_mono_ns=self._clock.process_start_mono_ns,
        )
        self._sequence = EventSequence()
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(config.queue_capacity)
        self._counters = TelemetryCounters()
        self._emit_lock = threading.Lock()
        self._accepting = True
        sink = sink_factory(
            config.output_dir,
            config.component,
            self._process_instance_id,
            config.max_file_bytes,
        )
        self._writer = AsyncJSONLWriter(
            event_queue=self._queue,
            counters=self._counters,
            sink=sink,
            batch_size=config.batch_size,
            flush_interval_ms=config.flush_interval_ms,
            output_dir=config.output_dir,
            component=config.component,
            process_instance_id=self._process_instance_id,
        )
        self._writer.start()

    @property
    def process_instance_id(self) -> str:
        return self._process_instance_id

    @property
    def output_paths(self) -> tuple[Path, ...]:
        return self._writer.paths

    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        trace_id: str | None = None,
        attempt_id: int | None = None,
        request_id: str | None = None,
        external_request_id: str | None = None,
    ) -> bool:
        self._counters.increment("events_attempted")
        with self._emit_lock:
            if not self._accepting:
                self._counters.increment("events_dropped_writer_failed")
                return False
            try:
                sample = self._clock.sample()
                event = build_event(
                    event_type=event_type,
                    ts_wall_ns=sample.wall_ns,
                    ts_mono_ns=sample.mono_ns,
                    process_start_wall_ns=self._clock.process_start_wall_ns,
                    process_start_mono_ns=self._clock.process_start_mono_ns,
                    host_id=self._config.host_id,
                    component=self._config.component,
                    process_id=self._process_id,
                    process_instance_id=self._process_instance_id,
                    event_seq=self._sequence.next(),
                    experiment_id=self._config.experiment_id,
                    run_id=self._config.run_id,
                    config_id=self._config.config_id,
                    trace_id=trace_id,
                    attempt_id=attempt_id,
                    request_id=request_id,
                    external_request_id=external_request_id,
                    payload=payload,
                )
            except Exception:
                self._counters.increment("event_build_errors")
                return False
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._counters.increment("events_dropped_queue_full")
                self._counters.update_queue_depth(self._queue.qsize())
                return False
            self._counters.increment("events_enqueued")
            self._counters.update_queue_depth(self._queue.qsize())
            return True

    def flush(self, timeout_s: float | None = None) -> bool:
        target = self._counters.snapshot()["events_enqueued"]
        return self._writer.request_flush(target, timeout_s)

    def close(self, timeout_s: float | None = None) -> bool:
        with self._emit_lock:
            self._accepting = False
            target = self._counters.snapshot()["events_enqueued"]
        return self._writer.close(target, timeout_s)

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "accepting": self._accepting,
            "writer_alive": self._writer.thread.is_alive(),
            "process_id": self._process_id,
            "process_instance_id": self._process_instance_id,
            "output_files": [str(path) for path in self._writer.paths],
            **self._counters.snapshot(),
        }


def create_emitter(config: TelemetryConfig | None = None) -> Emitter:
    resolved = TelemetryConfig.from_env() if config is None else config
    if not resolved.enabled:
        return NullEmitter()
    return AsyncTelemetryEmitter(resolved)
