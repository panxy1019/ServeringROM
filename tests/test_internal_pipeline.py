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

    def test_wrapped_engine_request_id_uses_proxy_uuid(self):
        request_id = "cb033b37-b4dd-49fb-881e-5c7e864c72d9"
        event = self.event(
            1,
            "engine_request_added",
            {"prompt_tokens": 8},
            request_id=f"cmpl-{request_id}-0-suffix",
        )
        root = Path(tempfile.mkdtemp())
        derived = root / "derived"
        derived.mkdir()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is required")
        pq.write_table(
            pa.Table.from_pylist(
                [{"request_id": request_id, "trace_id": "trace", "attempt_id": 0}]
            ),
            derived / "attempt_lifecycle.parquet",
        )
        tables = reconstruct_internal_tables(
            root, InternalEventDataset([event], [], [], [])
        )
        row = tables["engine_requests"][0]
        self.assertEqual(row["request_id"], request_id)
        self.assertEqual(row["engine_request_id"], f"cmpl-{request_id}-0-suffix")

        output = self.event(
            2,
            "engine_output_batch",
            {
                "iteration_id": 1,
                "members": [
                    {
                        "request_id": f"cmpl-{request_id}-0-suffix",
                        "new_token_count": 1,
                    }
                ],
            },
        )
        token_tables = reconstruct_internal_tables(
            root, InternalEventDataset([output], [], [], [])
        )
        self.assertEqual(token_tables["token_emissions"][0]["request_id"], request_id)

    def test_kv_rank_events_aggregate_to_one_request_transfer(self):
        events = []
        sequence = itertools.count(1)
        for rank in (0, 1):
            for event_type, payload in (
                (
                    "kv_transfer_enqueued",
                    {"enqueue_wall_ns": 100 + rank, "enqueue_mono_ns": 200 + rank},
                ),
                (
                    "kv_transfer_started",
                    {"start_wall_ns": 110 + rank, "start_mono_ns": 210 + rank},
                ),
                (
                    "kv_transfer_completed",
                    {
                        "complete_wall_ns": 130 + rank,
                        "complete_mono_ns": 230 + rank,
                        "success": True,
                    },
                ),
            ):
                events.append(
                    self.event(
                        next(sequence),
                        event_type,
                        {
                            "engine_role": "decode",
                            "engine_instance": "decode-0",
                            "tp_rank": rank,
                            "tp_size": 2,
                            "remote_request_id": "prefill-req-1",
                            "transfer_role": "receive",
                            "source_engine": "prefill-engine",
                            "target_engine": "decode-engine",
                            "block_count": 4,
                            "actual_bytes": 1024,
                            "descriptor_count": 8,
                            **payload,
                        },
                        component=f"decode-rank-{rank}",
                    )
                )
        tables = reconstruct_internal_tables(
            Path(tempfile.mkdtemp()), InternalEventDataset(events, [], [], [])
        )
        self.assertEqual(len(tables["kv_transfer_ranks"]), 2)
        self.assertEqual(len(tables["kv_transfers"]), 1)
        transfer = tables["kv_transfers"][0]
        self.assertTrue(transfer["success"])
        self.assertEqual(transfer["actual_total_bytes"], 2048)
        self.assertEqual(transfer["completed_rank_count"], 2)
        self.assertEqual(transfer["missing_ranks_json"], "[]")
        self.assertEqual(transfer["kv_ready_mono_ns"], 231)


if __name__ == "__main__":
    import unittest

    unittest.main()
