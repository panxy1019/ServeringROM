import threading
import time
from unittest import TestCase

from servingrom_telemetry.clock import ProcessClock
from servingrom_telemetry.ids import (
    EventSequence,
    build_process_instance_id,
    new_request_id,
    new_trace_id,
)
from servingrom_telemetry.schema import REQUIRED_EVENT_FIELDS, build_event, validate_event


class IdentityTest(TestCase):
    def test_uuid_uniqueness(self) -> None:
        self.assertEqual(len({new_trace_id() for _ in range(10_000)}), 10_000)
        self.assertEqual(len({new_request_id() for _ in range(10_000)}), 10_000)

    def test_process_instance_is_reproducible_with_nonce(self) -> None:
        args = dict(
            host_id="host",
            component="proxy",
            process_id=42,
            process_start_wall_ns=100,
            process_start_mono_ns=50,
            nonce="fixed",
        )
        self.assertEqual(build_process_instance_id(**args), build_process_instance_id(**args))

    def test_multithread_sequence_strictly_increasing(self) -> None:
        sequence = EventSequence()
        values: list[int] = []
        values_lock = threading.Lock()

        def produce() -> None:
            local = [sequence.next() for _ in range(5_000)]
            with values_lock:
                values.extend(local)

        threads = [threading.Thread(target=produce) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(values), list(range(1, 40_001)))


class ClockTest(TestCase):
    def test_dual_clock_and_monotonic_duration(self) -> None:
        clock = ProcessClock()
        first = clock.sample()
        time.sleep(0.001)
        second = clock.sample()
        self.assertGreaterEqual(first.wall_ns, clock.process_start_wall_ns)
        self.assertGreaterEqual(first.mono_ns, clock.process_start_mono_ns)
        self.assertGreater(second.mono_ns, first.mono_ns)
        self.assertEqual(
            ProcessClock.duration_ns(first.mono_ns, second.mono_ns),
            second.mono_ns - first.mono_ns,
        )
        with self.assertRaises(ValueError):
            ProcessClock.duration_ns(2, 1)

    def test_process_start_is_stable_across_emitters(self) -> None:
        first = ProcessClock()
        time.sleep(0.001)
        second = ProcessClock()
        self.assertEqual(first.process_start_wall_ns, second.process_start_wall_ns)
        self.assertEqual(first.process_start_mono_ns, second.process_start_mono_ns)


class SchemaTest(TestCase):
    def test_event_keeps_nullable_request_fields(self) -> None:
        event = build_event(
            event_type="test",
            ts_wall_ns=10,
            ts_mono_ns=9,
            process_start_wall_ns=1,
            process_start_mono_ns=1,
            host_id="host",
            component="component",
            process_id=7,
            process_instance_id="instance",
            event_seq=1,
            experiment_id="experiment",
            run_id="run",
            config_id="config",
            payload={"value": 1},
        )
        validate_event(event)
        self.assertEqual(set(REQUIRED_EVENT_FIELDS), set(event))
        self.assertIsNone(event["trace_id"])
        self.assertIsNone(event["attempt_id"])

    def test_invalid_schema_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_event({"schema_version": "wrong"})
        with self.assertRaises(ValueError):
            build_event(
                event_type="",
                ts_wall_ns=1,
                ts_mono_ns=1,
                process_start_wall_ns=1,
                process_start_mono_ns=1,
                host_id="h",
                component="c",
                process_id=1,
                process_instance_id="p",
                event_seq=1,
                experiment_id="e",
                run_id="r",
                config_id="c",
                payload={},
            )
