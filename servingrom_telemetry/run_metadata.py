from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .schema import SCHEMA_VERSION


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REQUIRED_RUN_FIELDS = (
    "experiment_id",
    "run_id",
    "config_id",
    "model",
    "tokenizer_revision",
    "image_tag",
    "image_digest",
    "git_commit",
    "deployment",
    "pod",
    "pod_uid",
    "prefill_endpoints",
    "decode_endpoints",
    "graph_mode",
    "async_scheduling",
    "tp",
    "telemetry",
    "workload",
    "random_seed",
)


def _validate_id(name: str, value: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} contains unsafe characters: {value!r}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class RunLayout:
    root: Path
    experiment_id: str
    run_id: str

    @classmethod
    def create(cls, results_root: Path, experiment_id: str, run_id: str) -> "RunLayout":
        layout = cls(
            Path(results_root) / _validate_id("experiment_id", experiment_id) / _validate_id("run_id", run_id),
            experiment_id,
            run_id,
        )
        for directory in (layout.metadata, layout.raw_proxy, layout.derived, layout.reports):
            directory.mkdir(parents=True, exist_ok=True)
        return layout

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def raw_proxy(self) -> Path:
        return self.root / "raw" / "proxy"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def write_run_metadata(layout: RunLayout, run: Mapping[str, Any], *, deployment_yaml: str) -> None:
    missing = [name for name in REQUIRED_RUN_FIELDS if name not in run]
    if missing:
        raise ValueError(f"missing run metadata fields: {missing}")
    if run["experiment_id"] != layout.experiment_id or run["run_id"] != layout.run_id:
        raise ValueError("run metadata identifiers do not match the run layout")

    _atomic_json(layout.metadata / "run.yaml", dict(run))
    (layout.metadata / "deployment.yaml").write_text(deployment_yaml, encoding="utf-8")
    _atomic_json(layout.metadata / "git.json", run.get("git", {"commit": run["git_commit"]}))
    _atomic_json(
        layout.metadata / "image.json",
        {"tag": run["image_tag"], "digest": run["image_digest"]},
    )
    _atomic_json(
        layout.metadata / "process.json",
        run.get("process", {"pod": run["pod"], "pod_uid": run["pod_uid"]}),
    )
    _atomic_json(layout.metadata / "telemetry_config.json", run["telemetry"])
    _atomic_json(
        layout.metadata / "schema_versions.json",
        {"proxy_event": SCHEMA_VERSION, "run_metadata": "servingrom.run.v1"},
    )


def build_sha256_manifest(layout: RunLayout) -> dict[str, Any]:
    manifest_path = layout.metadata / "sha256_manifest.json"
    files: list[dict[str, Any]] = []
    for path in sorted(layout.root.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.endswith(".tmp"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(
            {
                "path": path.relative_to(layout.root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    manifest = {"algorithm": "sha256", "file_count": len(files), "files": files}
    _atomic_json(manifest_path, manifest)
    return manifest
