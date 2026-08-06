from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

from .counters import TelemetryCounters
from .jsonl_sink import RotatingJSONLSink


class AsyncJSONLWriter:
    """Drain one process-local queue into one process-local JSONL file set."""

    def __init__(
        self,
        *,
        event_queue: queue.Queue[dict[str, Any]],
        counters: TelemetryCounters,
        sink: RotatingJSONLSink,
        batch_size: int,
        flush_interval_ms: int,
        output_dir: Path,
        component: str,
        process_instance_id: str,
    ) -> None:
        self._queue = event_queue
        self._counters = counters
        self._sink = sink
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_ms / 1000.0
        self._summary_interval_s = max(1.0, self._flush_interval_s * 20)
        self._output_dir = Path(output_dir)
        self._component = component.replace("/", "_")
        self._process_instance_id = process_instance_id
        self._stop_requested = threading.Event()
        self._flush_requested = threading.Event()
        self._condition = threading.Condition()
        self._processed = 0
        self._flush_target = 0
        self._flush_ack = 0
        self._flush_ok = True
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"servingrom-writer-{process_instance_id[:8]}",
            daemon=True,
        )

    @property
    def thread(self) -> threading.Thread:
        return self._thread

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self._sink.paths)

    def start(self) -> None:
        self._thread.start()

    def request_flush(self, target: int, timeout_s: float | None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            self._flush_target = max(self._flush_target, target)
            self._flush_requested.set()
            self._condition.notify_all()
            while self._flush_ack < target:
                if not self._thread.is_alive():
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return self._flush_ok

    def close(self, target: int, timeout_s: float | None) -> bool:
        if self._closed:
            return not self._thread.is_alive() and self._flush_ok
        self._stop_requested.set()
        self._flush_target = max(self._flush_target, target)
        self._flush_requested.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=timeout_s)
        self._closed = not self._thread.is_alive()
        return self._closed and self._flush_ok

    def _serialize(self, event: dict[str, Any]) -> bytes | None:
        try:
            return (
                json.dumps(
                    event,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            self._counters.increment("serialization_errors")
            return None

    def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        encoded = [item for event in batch if (item := self._serialize(event)) is not None]
        if not encoded:
            return
        started = time.monotonic_ns()
        try:
            written = self._sink.write_batch(encoded)
        except Exception:
            self._counters.add_many(
                (("write_errors", 1), ("events_dropped_writer_failed", len(encoded)))
            )
            return
        latency = time.monotonic_ns() - started
        self._counters.add_many(
            (
                ("events_written", len(encoded)),
                ("writer_batches", 1),
                ("writer_bytes", written),
            )
        )
        self._counters.record_write_latency(latency)

    def _flush(self) -> bool:
        try:
            self._sink.flush()
        except Exception:
            self._counters.increment("flush_errors")
            return False
        return True

    def _ack_flush_if_ready(self) -> None:
        with self._condition:
            if self._flush_requested.is_set() and self._processed >= self._flush_target:
                self._flush_ok = self._flush() and self._flush_ok
                self._flush_ack = self._processed
                self._flush_requested.clear()
                self._condition.notify_all()

    def _write_summary(self) -> None:
        summary = self._counters.snapshot()
        summary.update(
            {
                "component": self._component,
                "process_instance_id": self._process_instance_id,
                "jsonl_files": [path.name for path in self._sink.paths],
            }
        )
        path = self._output_dir / f"{self._process_instance_id}.summary.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except Exception:
            self._counters.increment("write_errors")
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _run(self) -> None:
        last_flush = time.monotonic()
        last_summary = last_flush
        try:
            while not self._stop_requested.is_set() or not self._queue.empty():
                batch: list[dict[str, Any]] = []
                try:
                    batch.append(self._queue.get(timeout=min(self._flush_interval_s, 0.05)))
                except queue.Empty:
                    pass
                while len(batch) < self._batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                if batch:
                    self._write_batch(batch)
                    for _ in batch:
                        self._queue.task_done()
                    with self._condition:
                        self._processed += len(batch)
                        self._condition.notify_all()
                    self._counters.update_queue_depth(self._queue.qsize())
                now = time.monotonic()
                if now - last_flush >= self._flush_interval_s:
                    self._flush_ok = self._flush() and self._flush_ok
                    last_flush = now
                if now - last_summary >= self._summary_interval_s:
                    self._write_summary()
                    last_summary = now
                self._ack_flush_if_ready()
            self._flush_ok = self._flush() and self._flush_ok
            with self._condition:
                self._flush_ack = max(self._flush_ack, self._processed)
                self._condition.notify_all()
        finally:
            try:
                self._sink.close()
            except Exception:
                self._counters.increment("flush_errors")
                self._flush_ok = False
            self._counters.update_queue_depth(self._queue.qsize())
            self._write_summary()
            with self._condition:
                self._closed = True
                self._condition.notify_all()
