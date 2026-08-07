from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "validation", "test", "test/transient")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunSlice:
    run_id: str
    split: str
    start: int
    end: int
    workload: str
    arrival_process: str
    transient_pattern: str | None


class RomDataset:
    def __init__(self, root: Path, index_dir: Path) -> None:
        self.root = root.resolve()
        self.index_dir = index_dir.resolve()
        self.state_index = json.loads((self.root / "state_index.json").read_text(encoding="utf-8"))
        self.disturbance_index = json.loads((self.index_dir / "disturbance_index.json").read_text(encoding="utf-8"))
        self.output_index = json.loads((self.index_dir / "output_index.json").read_text(encoding="utf-8"))
        static = json.loads((self.index_dir / "static_config.json").read_text(encoding="utf-8"))
        self.static_index = [
            {"index": i, "name": name, "block": "static_config", "unit": "scalar", "quantity": name}
            for i, name in enumerate(static["numeric_index"])
        ]
        self._run_rows = self._read_run_manifest()
        self._validate_contract()

    def _read_run_manifest(self) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq

        return pq.read_table(self.root / "run_manifest.parquet").to_pylist()

    def array(self, split: str, name: str) -> np.ndarray:
        return np.load(self.root / split / f"{name}.npy", mmap_mode="r")

    def run_slices(self, split: str) -> list[RunSlice]:
        rows = [
            row for row in self._run_rows
            if ("test/transient" if row.get("transient_pattern") else row["split"]) == split
        ]
        offset = 0
        slices = []
        for row in rows:
            count = int(row["window_count"])
            slices.append(RunSlice(
                run_id=row["run_id"], split=split, start=offset, end=offset + count,
                workload=row["workload"], arrival_process=row["arrival_process"],
                transient_pattern=row.get("transient_pattern"),
            ))
            offset += count
        return slices

    def _validate_contract(self) -> None:
        manifest = json.loads((self.root / "dataset_manifest.json").read_text(encoding="utf-8"))
        expected = {row["path"]: row["sha256"] for row in manifest["files"]}
        for relative in ("state_index.json", "run_manifest.parquet", "snapshot_schema.json"):
            if sha256(self.root / relative) != expected.get(relative):
                raise ValueError(f"dataset manifest mismatch: {relative}")
        dimensions = {"X": 1804, "X_next": 1804, "D": 31, "Y": 19, "MU": 12}
        seen_runs: set[str] = set()
        for split in SPLITS:
            rows = None
            for name, dimension in dimensions.items():
                array = self.array(split, name)
                if array.ndim != 2 or array.shape[1] != dimension:
                    raise ValueError(f"invalid {split}/{name} shape: {array.shape}")
                rows = rows or array.shape[0]
                if array.shape[0] != rows:
                    raise ValueError(f"row mismatch in split {split}")
            slices = self.run_slices(split)
            if sum(item.end - item.start for item in slices) != rows:
                raise ValueError(f"run boundaries do not cover split {split}")
            for item in slices:
                if item.run_id in seen_runs:
                    raise ValueError(f"run appears in more than one split: {item.run_id}")
                seen_runs.add(item.run_id)
        if len(seen_runs) != len(self._run_rows):
            raise ValueError("run manifest is not fully represented by split arrays")

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.root),
            "dataset_manifest_sha256": sha256(self.root / "dataset_manifest.json"),
            "run_manifest_sha256": sha256(self.root / "run_manifest.parquet"),
            "state_index_sha256": sha256(self.root / "state_index.json"),
            "disturbance_index_sha256": sha256(self.index_dir / "disturbance_index.json"),
            "output_index_sha256": sha256(self.index_dir / "output_index.json"),
            "static_config_sha256": sha256(self.index_dir / "static_config.json"),
            "runs_by_split": {split: len(self.run_slices(split)) for split in SPLITS},
        }

