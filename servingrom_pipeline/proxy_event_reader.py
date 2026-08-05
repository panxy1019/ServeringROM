from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from servingrom_telemetry.schema import validate_event


@dataclass(slots=True)
class ProxyEventDataset:
    events: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    damaged_lines: list[dict[str, Any]]
    source_files: list[str]


def read_proxy_events(raw_proxy_dir: Path) -> ProxyEventDataset:
    root = Path(raw_proxy_dir)
    events: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    damaged: list[dict[str, Any]] = []
    source_files: list[str] = []

    for path in sorted(root.glob("*.jsonl")):
        source_files.append(path.name)
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    event = json.loads(line)
                    validate_event(event)
                except Exception as exc:
                    damaged.append(
                        {"file": path.name, "line": line_number, "error": type(exc).__name__}
                    )
                    continue
                event["_source_file"] = path.name
                event["_source_line"] = line_number
                events.append(event)

    for path in sorted(root.glob("*.summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            damaged.append({"file": path.name, "line": None, "error": type(exc).__name__})

    events.sort(
        key=lambda event: (
            event["process_instance_id"],
            event["event_seq"],
        )
    )
    return ProxyEventDataset(events, summaries, damaged, source_files)
