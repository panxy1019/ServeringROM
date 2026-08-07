from __future__ import annotations

from unittest import TestCase

from servingrom_pipeline.snapshot_builder import SnapshotConfig, _measurement_contract

import json
import tempfile
from pathlib import Path


class SloCanonicalizationTest(TestCase):
    def test_forced_slo_overrides_workload_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            (root / "metadata/workload.json").write_text(json.dumps({"ttft_slo_ms": 5000, "tpot_slo_ms": 250}))
            config, _, _ = _measurement_contract(root, SnapshotConfig(force_ttft_slo_ms=2000, force_tpot_slo_ms=100))
            self.assertEqual(config.default_ttft_slo_ms, 2000)
            self.assertEqual(config.default_tpot_slo_ms, 100)
