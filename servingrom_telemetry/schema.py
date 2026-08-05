from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final


SCHEMA_VERSION: Final = "servingrom.telemetry.v1"
REQUIRED_EVENT_FIELDS: Final = (
    "schema_version",
    "event_type",
    "ts_wall_ns",
    "ts_mono_ns",
    "process_start_wall_ns",
    "process_start_mono_ns",
    "host_id",
    "component",
    "process_id",
    "process_instance_id",
    "event_seq",
    "experiment_id",
    "run_id",
    "config_id",
    "trace_id",
    "attempt_id",
    "request_id",
    "external_request_id",
    "payload",
)


def build_event(
    *,
    event_type: str,
    ts_wall_ns: int,
    ts_mono_ns: int,
    process_start_wall_ns: int,
    process_start_mono_ns: int,
    host_id: str,
    component: str,
    process_id: int,
    process_instance_id: str,
    event_seq: int,
    experiment_id: str,
    run_id: str,
    config_id: str,
    payload: Mapping[str, Any],
    trace_id: str | None = None,
    attempt_id: int | None = None,
    request_id: str | None = None,
    external_request_id: str | None = None,
) -> dict[str, Any]:
    if not event_type:
        raise ValueError("event_type must not be empty")
    if event_seq <= 0:
        raise ValueError("event_seq must be positive")
    if attempt_id is not None and attempt_id < 0:
        raise ValueError("attempt_id must not be negative")
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    return {
        "schema_version": SCHEMA_VERSION,
        "event_type": event_type,
        "ts_wall_ns": ts_wall_ns,
        "ts_mono_ns": ts_mono_ns,
        "process_start_wall_ns": process_start_wall_ns,
        "process_start_mono_ns": process_start_mono_ns,
        "host_id": host_id,
        "component": component,
        "process_id": process_id,
        "process_instance_id": process_instance_id,
        "event_seq": event_seq,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "config_id": config_id,
        "trace_id": trace_id,
        "attempt_id": attempt_id,
        "request_id": request_id,
        "external_request_id": external_request_id,
        "payload": dict(payload),
    }


def validate_event(event: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_EVENT_FIELDS if field not in event]
    if missing:
        raise ValueError(f"missing required event fields: {missing}")
    if event["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {event['schema_version']!r}")
    if not isinstance(event["payload"], Mapping):
        raise TypeError("payload must be a mapping")
    if not isinstance(event["event_seq"], int) or event["event_seq"] <= 0:
        raise ValueError("event_seq must be a positive integer")
    attempt_id = event["attempt_id"]
    if attempt_id is not None and (not isinstance(attempt_id, int) or attempt_id < 0):
        raise ValueError("attempt_id must be null or a non-negative integer")
