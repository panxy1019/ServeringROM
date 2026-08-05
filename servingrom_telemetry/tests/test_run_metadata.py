from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from servingrom_telemetry.run_metadata import RunLayout, build_sha256_manifest, write_run_metadata


def run_record() -> dict[str, object]:
    return {
        "experiment_id": "exp-1",
        "run_id": "run-1",
        "config_id": "qwen36-1p2d-d2-full-decode-only-async-v1",
        "model": "Qwen3.6-27B-W8A8",
        "tokenizer_revision": "local",
        "image_tag": "registry/image:tag",
        "image_digest": "sha256:abc",
        "git_commit": "0123456",
        "deployment": "d2",
        "pod": "pod-1",
        "pod_uid": "uid-1",
        "prefill_endpoints": ["127.0.0.1:13700"],
        "decode_endpoints": ["127.0.0.1:13710", "127.0.0.1:13720"],
        "graph_mode": "FULL_DECODE_ONLY",
        "async_scheduling": True,
        "tp": {"prefill": 2, "decode_a": 2, "decode_b": 2},
        "telemetry": {"enabled": True},
        "workload": "smoke",
        "random_seed": 7,
    }


class RunMetadataTest(TestCase):
    def test_layout_metadata_and_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            layout = RunLayout.create(Path(directory), "exp-1", "run-1")
            write_run_metadata(layout, run_record(), deployment_yaml="kind: Deployment\n")
            (layout.raw_proxy / "events.jsonl").write_text("{}\n", encoding="utf-8")
            manifest = build_sha256_manifest(layout)
            self.assertTrue((layout.metadata / "run.yaml").is_file())
            self.assertEqual(json.loads((layout.metadata / "run.yaml").read_text())["run_id"], "run-1")
            self.assertIn("raw/proxy/events.jsonl", {item["path"] for item in manifest["files"]})

    def test_unsafe_identifiers_and_missing_metadata_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                RunLayout.create(Path(directory), "../escape", "run")
            layout = RunLayout.create(Path(directory), "exp", "run")
            with self.assertRaises(ValueError):
                write_run_metadata(layout, {"experiment_id": "exp", "run_id": "run"}, deployment_yaml="")
