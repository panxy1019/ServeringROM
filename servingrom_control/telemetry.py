from __future__ import annotations

from typing import Any, Mapping


CONTROL_EVENT_FIELDS = (
    "control_command_id",
    "control_generation",
    "actuator_name",
    "old_value",
    "requested_value",
    "effective_value",
    "requested_wall_ns",
    "applied_wall_ns",
    "reason",
)


def control_event_payload(response: Mapping[str, Any]) -> dict[str, Any]:
    return {name: response.get(name) for name in CONTROL_EVENT_FIELDS} | {
        "validated_wall_ns": response.get("validated_wall_ns"),
        "effective_from": response.get("effective_from"),
        "accepted": response.get("accepted"),
    }
