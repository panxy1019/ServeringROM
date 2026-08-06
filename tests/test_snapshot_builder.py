from __future__ import annotations

from unittest import TestCase

import json
import tempfile
from pathlib import Path

from servingrom_pipeline.snapshot_builder import SnapshotConfig, _histogram, _measurement_contract, _state_at


class SnapshotBuilderTest(TestCase):
    def test_half_open_kv_state_machine(self):
        row = {
            "arrival_wall_ns": 10,
            "admission_accepted": True,
            "prefill_submit_wall_ns": 15,
            "prefill_added_wall_ns": 20,
            "prefill_terminal_wall_ns": 30,
            "kv_enqueue_wall_ns": 40,
            "kv_first_start_wall_ns": 50,
            "kv_ready_wall_ns": 60,
            "decode_added_wall_ns": 70,
        }
        self.assertEqual(_state_at(row, 10), "ADMITTED")
        self.assertEqual(_state_at(row, 15), "PREFILL_WAITING")
        self.assertEqual(_state_at(row, 20), "PREFILL_RUNNING")
        self.assertEqual(_state_at(row, 30), "HANDOFF_WAITING")
        self.assertEqual(_state_at(row, 40), "KV_QUEUED")
        self.assertEqual(_state_at(row, 50), "KV_TRANSFERRING")
        self.assertEqual(_state_at(row, 60), "KV_READY")
        self.assertEqual(_state_at(row, 70), "DECODE_RUNNING")

    def test_histogram_clamps_to_overflow_bin(self):
        self.assertEqual(_histogram([0, 1, 512, 10000, None], maximum=512, width=256), [2, 0, 2])

    def test_measurement_contract_is_exactly_period_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata"
            metadata.mkdir()
            (metadata / "measurement.json").write_text(json.dumps({
                "measurement_start_wall_ns": 1_000_000_000,
                "measurement_end_wall_ns": 1_400_000_000,
                "snapshot_period_ms": 200,
            }))
            (metadata / "workload.json").write_text(json.dumps({
                "ttft_slo_ms": 2000,
                "tpot_slo_ms": 100,
            }))
            config, start, end = _measurement_contract(Path(directory), SnapshotConfig())
            self.assertEqual((start, end), (1_000_000_000, 1_400_000_000))
            self.assertEqual(config.default_ttft_slo_ms, 2000)
            self.assertEqual(config.default_tpot_slo_ms, 100)

    def test_measurement_contract_rejects_partial_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata"
            metadata.mkdir()
            (metadata / "measurement.json").write_text(json.dumps({
                "measurement_start_wall_ns": 1,
                "measurement_end_wall_ns": 200_000_001,
                "snapshot_period_ms": 200,
            }))
            with self.assertRaises(ValueError):
                _measurement_contract(Path(directory), SnapshotConfig())
