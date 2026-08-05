from __future__ import annotations

import itertools
import tempfile
from pathlib import Path
from unittest import TestCase

from servingrom_pipeline.internal_event_reader import InternalEventDataset
from servingrom_pipeline.internal_reconstruction import reconstruct_internal_tables
from servingrom_pipeline.internal_validation import validate_internal_data
from servingrom_telemetry.schema import build_event


class InternalPipelineTest(TestCase):
    def event(self, seq, event_type, payload, request_id="req-1", component="prefill"):
        return build_event(
            event_type=event_type,
            ts_wall_ns=1_000_000 + seq,
            ts_mono_ns=2_000_000 + seq,
            process_start_wall_ns=1,
            process_start_mono_ns=1,
            host_id="host",
            component=component,
            process_id=7,
            process_instance_id=f"proc-{component}",
            event_seq=seq,
            experiment_id="exp",
            run_id="run",
            config_id="config",
            request_id=request_id,
            payload={
                "engine_role": component,
                "engine_instance": component,
                "tp_rank": 0,
                "is_driver_rank": True,
                **payload,
            },
        )

    def test_reconstruction_and_iteration_reconciliation(self):
        events = [
            self.event(1, "engine_request_added", {"prompt_tokens": 8, "max_output_tokens": 1}),
            self.event(
                2,
                "scheduler_membership",
                {"iteration_id": 1, "members": [{"request_id": "req-1", "scheduled_tokens": 8}]},
            ),
            self.event(
                3,
                "scheduler_iteration",
                {"iteration_id": 1, "scheduled_tokens_total": 8, "scheduled_request_count": 1},
            ),
            self.event(
                4,
                "engine_output_batch",
                {"iteration_id": 1, "members": [{"request_id": "req-1", "new_token_count": 1}]},
            ),
            self.event(5, "engine_request_terminal", {"finish_reason": "stop"}),
        ]
        dataset = InternalEventDataset(
            events=events,
            summaries=[
                {
                    "process_instance_id": "proc-prefill",
                    "events_enqueued": 5,
                    "events_written": 5,
                    "events_dropped_queue_full": 0,
                    "events_dropped_writer_failed": 0,
                }
            ],
            damaged_lines=[],
            source_files=["prefill/events.jsonl"],
        )
        root = Path(tempfile.mkdtemp())
        tables = reconstruct_internal_tables(root, dataset)
        report = validate_internal_data(root, tables, dataset)
        self.assertEqual(report["violation_count"], 0)
        self.assertEqual(report["table_counts"]["scheduler_membership"], 1)
        self.assertEqual(report["prefill_token_reconciliation"][0]["classification"], "exact")

    def test_corrupt_membership_is_rejected(self):
        events = [
            self.event(
                1,
                "scheduler_membership",
                {"iteration_id": 1, "members": [{"request_id": "req-1", "scheduled_tokens": 7}]},
            ),
            self.event(
                2,
                "scheduler_iteration",
                {"iteration_id": 1, "scheduled_tokens_total": 8, "scheduled_request_count": 1},
            ),
        ]
        dataset = InternalEventDataset(events, [], [], [])
        tables = reconstruct_internal_tables(Path(tempfile.mkdtemp()), dataset)
        report = validate_internal_data(Path(tempfile.mkdtemp()), tables, dataset)
        self.assertIn("membership_token_sum_mismatch", report["violation_counts"])


if __name__ == "__main__":
    import unittest

    unittest.main()
