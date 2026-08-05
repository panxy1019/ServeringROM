from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pyarrow.parquet as pq

from servingrom_pipeline.proxy_event_reader import read_proxy_events
from servingrom_pipeline.proxy_lifecycle_builder import build_proxy_lifecycle
from servingrom_telemetry.schema import build_event


def event(event_type: str, sequence: int, payload: dict | None = None) -> dict:
    return build_event(
        event_type=event_type,
        ts_wall_ns=10_000 + sequence,
        ts_mono_ns=20_000 + sequence,
        process_start_wall_ns=1,
        process_start_mono_ns=1,
        host_id="host",
        component="proxy",
        process_id=7,
        process_instance_id="process-io",
        event_seq=sequence,
        experiment_id="exp",
        run_id="run",
        config_id="config",
        trace_id="trace-io",
        attempt_id=0,
        request_id="request-io",
        external_request_id="external-io",
        payload=payload or {},
    )


class ProxyLifecycleIOTest(TestCase):
    def test_reader_builder_and_parquet_round_trip(self) -> None:
        events = [
            event("request_arrival", 1, {"input_tokens": 4, "arrival_mono_ns": 20_001}),
            event("admission_decision", 2, {"accepted": True}),
            event("prefill_http_submit", 3),
            event("prefill_http_complete", 4),
            event("p_to_d_route", 5, {"selected_decoder": "decode-a"}),
            event("decode_http_submit", 6),
            event("decode_first_byte", 7),
            event("decode_stream_chunk", 8),
            event("request_complete", 9, {"output_tokens": 2, "output_sha256": "abc"}),
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw" / "proxy"
            raw.mkdir(parents=True)
            (raw / "process-io.00000.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
            )
            (raw / "process-io.summary.json").write_text(
                json.dumps(
                    {
                        "process_instance_id": "process-io",
                        "events_enqueued": len(events),
                        "events_written": len(events),
                    }
                ),
                encoding="utf-8",
            )
            dataset = read_proxy_events(raw)
            self.assertEqual(len(dataset.events), len(events))
            analysis = build_proxy_lifecycle(root)
            self.assertEqual(analysis.metrics["violation_count"], 0)
            trace_table = pq.read_table(root / "derived" / "trace_lifecycle.parquet")
            attempt_table = pq.read_table(root / "derived" / "attempt_lifecycle.parquet")
            self.assertEqual(trace_table.num_rows, 1)
            self.assertEqual(attempt_table.num_rows, 1)

    def test_reader_reports_damaged_jsonl(self) -> None:
        with TemporaryDirectory() as directory:
            raw = Path(directory)
            (raw / "process.00000.jsonl").write_text("{bad json}\n", encoding="utf-8")
            dataset = read_proxy_events(raw)
            self.assertEqual(len(dataset.damaged_lines), 1)
