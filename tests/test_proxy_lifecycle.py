from __future__ import annotations

import itertools
from unittest import TestCase

from servingrom_pipeline.proxy_state_machine import analyze_proxy_lifecycle
from servingrom_telemetry.schema import build_event


class Events:
    def __init__(self) -> None:
        self.seq = itertools.count(1)
        self.clock = itertools.count(1_000_000, 1_000)

    def add(
        self,
        event_type: str,
        *,
        trace: str = "trace-1",
        attempt: int = 0,
        request: str = "request-0",
        payload: dict | None = None,
    ) -> dict:
        mono = next(self.clock)
        return build_event(
            event_type=event_type,
            ts_wall_ns=2_000_000 + mono,
            ts_mono_ns=mono,
            process_start_wall_ns=1,
            process_start_mono_ns=1,
            host_id="host",
            component="proxy",
            process_id=1,
            process_instance_id="process-1",
            event_seq=next(self.seq),
            experiment_id="exp",
            run_id="run",
            config_id="config",
            trace_id=trace,
            attempt_id=attempt,
            request_id=request,
            external_request_id="external",
            payload=payload or {},
        )


def accepted_prefix(events: Events, *, stream: bool = True) -> list[dict]:
    return [
        events.add(
            "request_arrival",
            payload={
                "arrival_wall_ns": 3_000_000,
                "arrival_mono_ns": 999_000,
                "input_tokens": 32,
                "expected_output_tokens": 16,
                "stream": stream,
            },
        ),
        events.add("admission_decision", payload={"accepted": True}),
        events.add("prefill_http_submit", payload={"backend_endpoint": "prefill"}),
        events.add("prefill_http_complete", payload={"duration_ns": 100}),
        events.add(
            "p_to_d_route",
            payload={"selected_decoder": "decode-a", "route_reason": "fair_tie_round_robin"},
        ),
        events.add("decode_http_submit", payload={"backend_endpoint": "decode-a"}),
        events.add("decode_first_byte", payload={"backend_endpoint": "decode-a"}),
    ]


class ProxyLifecycleTest(TestCase):
    def assert_clean(self, records: list[dict]) -> None:
        summary = {
            "process_instance_id": "process-1",
            "events_enqueued": len(records),
            "events_written": len(records),
        }
        analysis = analyze_proxy_lifecycle(records, [summary])
        self.assertEqual(analysis.violations, [])

    def test_normal_streaming_and_non_streaming(self) -> None:
        for stream in (True, False):
            events = Events()
            records = accepted_prefix(events, stream=stream)
            records.append(events.add("decode_stream_chunk", payload={"chunk_index": 0}))
            records.append(
                events.add(
                    "request_complete",
                    payload={"output_tokens": 8, "output_sha256": "abc"},
                )
            )
            self.assert_clean(records)

    def test_rejection_cancel_and_final_error(self) -> None:
        for terminal in ("request_rejected", "request_cancel", "request_error"):
            events = Events()
            records = [
                events.add("request_arrival", payload={"input_tokens": 9}),
                events.add(
                    "admission_decision",
                    payload={"accepted": terminal != "request_rejected"},
                ),
            ]
            if terminal != "request_rejected":
                records.extend(
                    [
                        events.add("prefill_http_submit"),
                        events.add("prefill_http_complete"),
                        events.add("p_to_d_route"),
                        events.add("decode_http_submit"),
                    ]
                )
            records.append(events.add(terminal))
            self.assert_clean(records)

    def test_prefill_and_decode_retries_do_not_change_attempt(self) -> None:
        events = Events()
        records = accepted_prefix(events)
        records.insert(3, events.add("backend_retry", payload={"backend_role": "prefill"}))
        records.append(events.add("backend_retry", payload={"backend_role": "decode"}))
        records.append(events.add("request_complete"))
        # Restore process event order after inserting a later-created event.
        records.sort(key=lambda item: item["event_seq"])
        self.assert_clean(records)
        analysis = analyze_proxy_lifecycle(records)
        self.assertEqual(analysis.attempt_rows[0]["backend_retries"], 2)

    def test_recompute_preserves_trace_and_creates_second_attempt(self) -> None:
        events = Events()
        records = accepted_prefix(events)
        records.extend(
            [
                events.add(
                    "attempt_recomputed",
                    attempt=1,
                    request="request-1",
                    payload={
                        "previous_attempt_id": 0,
                        "previous_request_id": "request-0",
                        "new_attempt_id": 1,
                        "new_request_id": "request-1",
                    },
                ),
                events.add("prefill_http_submit", attempt=1, request="request-1"),
                events.add("prefill_http_complete", attempt=1, request="request-1"),
                events.add("p_to_d_route", attempt=1, request="request-1"),
                events.add("decode_http_submit", attempt=1, request="request-1"),
                events.add("decode_first_byte", attempt=1, request="request-1"),
                events.add("request_complete", attempt=1, request="request-1"),
            ]
        )
        self.assert_clean(records)
        analysis = analyze_proxy_lifecycle(records)
        self.assertEqual(len(analysis.attempt_rows), 2)
        self.assertEqual(analysis.trace_rows[0]["attempt_count"], 2)

    def test_duplicate_terminal_and_out_of_order_are_reported(self) -> None:
        events = Events()
        records = accepted_prefix(events)
        records.extend([events.add("request_complete"), events.add("request_error")])
        # Force route to precede prefill completion in monotonic time.
        route = next(item for item in records if item["event_type"] == "p_to_d_route")
        complete = next(item for item in records if item["event_type"] == "prefill_http_complete")
        route["ts_mono_ns"] = complete["ts_mono_ns"] - 1
        analysis = analyze_proxy_lifecycle(records)
        codes = {item["code"] for item in analysis.violations}
        self.assertIn("duplicate_terminal_event", codes)
        self.assertIn("event_order_violation", codes)

    def test_writer_mismatch_and_request_id_reuse_are_reported(self) -> None:
        events = Events()
        records = accepted_prefix(events)
        records.extend(
            [
                events.add(
                    "attempt_recomputed",
                    attempt=1,
                    request="request-0",
                    payload={
                        "previous_attempt_id": 0,
                        "previous_request_id": "request-0",
                        "new_attempt_id": 1,
                        "new_request_id": "request-0",
                    },
                ),
                events.add("request_error", attempt=1, request="request-0"),
            ]
        )
        analysis = analyze_proxy_lifecycle(
            records,
            [{"process_instance_id": "process-1", "events_enqueued": 2, "events_written": 1}],
        )
        codes = {item["code"] for item in analysis.violations}
        self.assertIn("request_id_reused_across_attempts", codes)
        self.assertIn("writer_counter_mismatch", codes)
