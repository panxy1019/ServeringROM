from __future__ import annotations

import os
import json
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import replace
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
        return {
            "enabled": False,
            "emit_latency_ns": {"count": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0},
            **{name: 0 for name in COUNTER_NAMES},
        }


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
        self._emit_latencies_ns: deque[int] = deque(maxlen=100_000)
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
    def config(self) -> TelemetryConfig:
        return self._config

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
        started_ns = time.perf_counter_ns()
        self._counters.increment("events_attempted")
        with self._emit_lock:
            if not self._accepting:
                self._counters.increment("events_dropped_writer_failed")
                return self._finish_emit(False, started_ns)
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
                return self._finish_emit(False, started_ns)
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._counters.increment("events_dropped_queue_full")
                self._counters.update_queue_depth(self._queue.qsize())
                return self._finish_emit(False, started_ns)
            self._counters.increment("events_enqueued")
            self._counters.update_queue_depth(self._queue.qsize())
            return self._finish_emit(True, started_ns)

    def _finish_emit(self, result: bool, started_ns: int) -> bool:
        self._emit_latencies_ns.append(time.perf_counter_ns() - started_ns)
        return result

    def flush(self, timeout_s: float | None = None) -> bool:
        target = self._counters.snapshot()["events_enqueued"]
        return self._writer.request_flush(target, timeout_s)

    def close(self, timeout_s: float | None = None) -> bool:
        with self._emit_lock:
            self._accepting = False
            target = self._counters.snapshot()["events_enqueued"]
        return self._writer.close(target, timeout_s)

    def health_snapshot(self) -> dict[str, Any]:
        with self._emit_lock:
            accepting = self._accepting
            latencies = sorted(self._emit_latencies_ns)

        def percentile(percent: float) -> int:
            if not latencies:
                return 0
            index = min(len(latencies) - 1, max(0, int((len(latencies) - 1) * percent)))
            return latencies[index]

        return {
            "enabled": True,
            "accepting": accepting,
            "writer_alive": self._writer.thread.is_alive(),
            "process_id": self._process_id,
            "process_instance_id": self._process_instance_id,
            "output_files": [str(path) for path in self._writer.paths],
            "emit_latency_ns": {
                "count": len(latencies),
                "p50": percentile(0.50),
                "p95": percentile(0.95),
                "p99": percentile(0.99),
                "max": latencies[-1] if latencies else 0,
            },
            **self._counters.snapshot(),
        }


class RunControlEmitter:
    """Hot-rotate process-local writers without restarting the model process."""

    def __init__(self, config: TelemetryConfig, control_file: Path, ack_dir: Path) -> None:
        self._base_config = config
        self._control_file = Path(control_file)
        self._ack_dir = Path(ack_dir)
        self._ack_dir.mkdir(parents=True, exist_ok=True)
        self._identity = (
            f"{config.component.replace('/', '_')}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        )
        self._lock = threading.Lock()
        self._emitter: Emitter = NullEmitter()
        self._generation = -1
        self._active = False
        self._closed = False
        self._last_error: str | None = None
        self._thread = threading.Thread(
            target=self._watch,
            name=f"servingrom-run-control-{self._identity}",
            daemon=True,
        )
        self._thread.start()

    def _read_control(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self._control_file.read_text(encoding="utf-8"))
            if not isinstance(value.get("generation"), int):
                raise ValueError("generation must be an integer")
            if not isinstance(value.get("active"), bool):
                raise ValueError("active must be a boolean")
            if value["active"]:
                for name in ("experiment_id", "run_id", "config_id", "run_root"):
                    if not isinstance(value.get(name), str) or not value[name]:
                        raise ValueError(f"active control is missing {name}")
            return value
        except FileNotFoundError:
            return None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None

    def _write_ack(self, control: Mapping[str, Any], close_ok: bool) -> None:
        value = {
            "identity": self._identity,
            "component": self._base_config.component,
            "process_id": os.getpid(),
            "generation": control["generation"],
            "active": control["active"],
            "run_id": control.get("run_id"),
            "close_ok": close_ok,
            "error": self._last_error,
            "ts_wall_ns": time.time_ns(),
        }
        path = self._ack_dir / f"{self._identity}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _transition(self, control: Mapping[str, Any]) -> None:
        with self._lock:
            previous = self._emitter
            self._emitter = NullEmitter()
        close_ok = previous.close(30.0)
        replacement: Emitter = NullEmitter()
        self._last_error = None
        if control["active"]:
            try:
                output_dir = (
                    Path(control["run_root"]) / "raw" / self._base_config.component
                )
                replacement = AsyncTelemetryEmitter(
                    replace(
                        self._base_config,
                        experiment_id=control["experiment_id"],
                        run_id=control["run_id"],
                        config_id=control["config_id"],
                        output_dir=output_dir,
                    )
                )
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            if not self._closed:
                self._emitter = replacement
            else:
                replacement.close(5.0)
        self._generation = int(control["generation"])
        self._active = bool(control["active"] and self._last_error is None)
        self._write_ack(control, close_ok)

    def _watch(self) -> None:
        while not self._closed:
            control = self._read_control()
            if control is not None and control["generation"] != self._generation:
                try:
                    self._transition(control)
                except Exception as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.1)

    def emit(self, event_type: str, payload: Mapping[str, Any], trace_id: str | None = None,
             attempt_id: int | None = None, request_id: str | None = None,
             external_request_id: str | None = None) -> bool:
        with self._lock:
            return self._emitter.emit(
                event_type, payload, trace_id, attempt_id,
                request_id, external_request_id,
            )

    def flush(self, timeout_s: float | None = None) -> bool:
        with self._lock:
            emitter = self._emitter
        return emitter.flush(timeout_s)

    def close(self, timeout_s: float | None = None) -> bool:
        self._closed = True
        self._thread.join(timeout=timeout_s)
        with self._lock:
            emitter = self._emitter
            self._emitter = NullEmitter()
        return emitter.close(timeout_s)

    def health_snapshot(self) -> dict[str, Any]:
        with self._lock:
            health = self._emitter.health_snapshot()
        return {
            **health,
            "run_control": True,
            "control_identity": self._identity,
            "control_generation": self._generation,
            "control_active": self._active,
            "control_error": self._last_error,
        }


def create_emitter(config: TelemetryConfig | None = None) -> Emitter:
    resolved = TelemetryConfig.from_env() if config is None else config
    if not resolved.enabled:
        return NullEmitter()
    control_file = os.getenv("SERVINGROM_RUN_CONTROL_FILE")
    if control_file:
        ack_dir = os.getenv(
            "SERVINGROM_RUN_CONTROL_ACK_DIR",
            str(Path(control_file).with_name("run-control-acks")),
        )
        return RunControlEmitter(resolved, Path(control_file), Path(ack_dir))
    return AsyncTelemetryEmitter(resolved)
