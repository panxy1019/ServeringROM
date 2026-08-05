from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from servingrom_telemetry.config import TelemetryConfig
from servingrom_telemetry.emitter import AsyncTelemetryEmitter, NullEmitter, create_emitter
from servingrom_telemetry.jsonl_sink import RotatingJSONLSink
from servingrom_telemetry.schema import validate_event


def config(path: str, **overrides: object) -> TelemetryConfig:
    values: dict[str, object] = {
        "enabled": True,
        "experiment_id": "experiment",
        "run_id": "run",
        "config_id": "config",
        "component": "test",
        "host_id": "host",
        "output_dir": Path(path),
        "queue_capacity": 20_000,
        "batch_size": 128,
        "flush_interval_ms": 10,
        "max_file_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return TelemetryConfig(**values)


class BlockingSink(RotatingJSONLSink):
    started = threading.Event()
    release = threading.Event()

    def write_batch(self, encoded_events: list[bytes]) -> int:
        self.started.set()
        self.release.wait(timeout=5)
        return super().write_batch(encoded_events)


class FailingWriteSink(RotatingJSONLSink):
    def write_batch(self, encoded_events: list[bytes]) -> int:
        raise OSError("injected write failure")


class FailingFlushSink(RotatingJSONLSink):
    def flush(self) -> None:
        raise OSError("injected flush failure")


class EmitterWriterTest(TestCase):
    def test_null_emitter_has_no_thread_or_file(self) -> None:
        with TemporaryDirectory() as directory:
            before = {thread.ident for thread in threading.enumerate()}
            emitter = create_emitter(TelemetryConfig(output_dir=Path(directory)))
            self.assertIsInstance(emitter, NullEmitter)
            self.assertFalse(emitter.emit("ignored", {"not": "materialized"}))
            self.assertTrue(emitter.flush())
            self.assertTrue(emitter.close())
            after = {thread.ident for thread in threading.enumerate()}
            self.assertEqual(before, after)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_null_emitter_does_not_materialize_event_or_clock(self) -> None:
        class ExplodingPayload(dict[str, object]):
            def __iter__(self):  # type: ignore[override]
                raise AssertionError("NullEmitter accessed payload")

        with patch(
            "servingrom_telemetry.emitter.ProcessClock",
            side_effect=AssertionError("NullEmitter constructed a clock"),
        ):
            emitter = create_emitter(TelemetryConfig())
            self.assertFalse(emitter.emit("ignored", ExplodingPayload()))
        self.assertFalse(hasattr(emitter, "__dict__"))

    def test_single_thread_flush_close_and_parse(self) -> None:
        with TemporaryDirectory() as directory:
            emitter = AsyncTelemetryEmitter(config(directory))
            for index in range(100):
                self.assertTrue(
                    emitter.emit(
                        "test_event",
                        {"index": index},
                        trace_id="trace",
                        attempt_id=0,
                        request_id="request",
                    )
                )
            self.assertTrue(emitter.flush(5))
            self.assertTrue(emitter.close(5))
            events = []
            for path in Path(directory).glob("*.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    event = json.loads(line)
                    validate_event(event)
                    events.append(event)
            self.assertEqual(len(events), 100)
            self.assertEqual([event["event_seq"] for event in events], list(range(1, 101)))
            health = emitter.health_snapshot()
            self.assertEqual(health["events_written"], health["events_enqueued"])
            self.assertTrue(list(Path(directory).glob("*.summary.json")))

    def test_multithread_writes_have_unique_sequence(self) -> None:
        with TemporaryDirectory() as directory:
            emitter = AsyncTelemetryEmitter(config(directory))

            def produce(worker: int) -> None:
                for index in range(1_000):
                    self.assertTrue(emitter.emit("concurrent", {"worker": worker, "index": index}))

            threads = [threading.Thread(target=produce, args=(worker,)) for worker in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertTrue(emitter.close(10))
            sequences = []
            for path in sorted(Path(directory).glob("*.jsonl")):
                sequences.extend(
                    json.loads(line)["event_seq"]
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
            self.assertEqual(sequences, list(range(1, 8_001)))

    def test_queue_full_drops_without_blocking(self) -> None:
        with TemporaryDirectory() as directory:
            BlockingSink.started.clear()
            BlockingSink.release.clear()
            emitter = AsyncTelemetryEmitter(
                config(directory, queue_capacity=1, batch_size=1),
                sink_factory=BlockingSink,
            )
            self.assertTrue(emitter.emit("first", {}))
            self.assertTrue(BlockingSink.started.wait(timeout=2))
            self.assertTrue(emitter.emit("second", {}))
            started = time.monotonic()
            self.assertFalse(emitter.emit("dropped", {}))
            self.assertLess(time.monotonic() - started, 0.05)
            BlockingSink.release.set()
            self.assertTrue(emitter.close(5))
            self.assertEqual(emitter.health_snapshot()["events_dropped_queue_full"], 1)

    def test_rotation(self) -> None:
        with TemporaryDirectory() as directory:
            emitter = AsyncTelemetryEmitter(config(directory, batch_size=2, max_file_bytes=700))
            for index in range(20):
                self.assertTrue(emitter.emit("rotate", {"index": index, "text": "x" * 80}))
            self.assertTrue(emitter.close(5))
            paths = list(Path(directory).glob("*.jsonl"))
            self.assertGreater(len(paths), 1)
            self.assertEqual(sum(len(path.read_text().splitlines()) for path in paths), 20)

    def test_serialization_and_disk_failures_are_isolated(self) -> None:
        with TemporaryDirectory() as directory:
            emitter = AsyncTelemetryEmitter(config(directory))
            self.assertTrue(emitter.emit("bad-json", {"bad": {1, 2, 3}}))
            self.assertTrue(emitter.emit("non-finite-json", {"bad": float("nan")}))
            self.assertTrue(emitter.close(5))
            self.assertEqual(emitter.health_snapshot()["serialization_errors"], 2)

        with TemporaryDirectory() as directory:
            emitter = AsyncTelemetryEmitter(config(directory), sink_factory=FailingWriteSink)
            self.assertTrue(emitter.emit("write-fails", {}))
            self.assertTrue(emitter.close(5))
            health = emitter.health_snapshot()
            self.assertEqual(health["write_errors"], 1)
            self.assertEqual(health["events_dropped_writer_failed"], 1)

        with TemporaryDirectory() as directory:
            emitter = AsyncTelemetryEmitter(config(directory), sink_factory=FailingFlushSink)
            self.assertTrue(emitter.emit("flush-fails", {}))
            self.assertFalse(emitter.close(5))
            self.assertFalse(emitter.close(5))
            self.assertGreaterEqual(emitter.health_snapshot()["flush_errors"], 1)
