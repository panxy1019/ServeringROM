from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from servingrom_telemetry.run_metadata import RAW_COMPONENTS
from servingrom_telemetry.schema import validate_event


@dataclass(slots=True)
class InternalEventDataset:
    events: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    damaged_lines: list[dict[str, Any]]
    source_files: list[str]


def read_internal_events(run_root: Path) -> InternalEventDataset:
    root = Path(run_root) / "raw"
    events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    damaged: list[dict[str, Any]] = []
    source_files: list[str] = []
    for component in RAW_COMPONENTS:
        component_dir = root / component
        for path in sorted(component_dir.glob("*.jsonl")):
            relative = path.relative_to(root).as_posix()
            source_files.append(relative)
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        event = json.loads(line)
                        validate_event(event)
                    except Exception as exc:
                        damaged.append(
                            {
                                "file": relative,
                                "line": line_number,
                                "error": type(exc).__name__,
                            }
                        )
                        continue
                    event["_source_component"] = component
                    event["_source_file"] = relative
                    event["_source_line"] = line_number
                    events.append(event)
        for path in sorted(component_dir.glob("*.summary.json")):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                summary["_source_component"] = component
                summary["_source_file"] = path.relative_to(root).as_posix()
                summaries.append(summary)
            except Exception as exc:
                damaged.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "line": None,
                        "error": type(exc).__name__,
                    }
                )
    events.sort(
        key=lambda event: (
            event["process_instance_id"],
            event["event_seq"],
        )
    )
    return InternalEventDataset(events, summaries, damaged, source_files)
