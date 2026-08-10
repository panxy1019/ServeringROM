from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ControlValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ControlCommand:
    control_command_id: str
    control_generation: int
    actuator_name: str
    requested_value: float | str
    expected_current_value: float | str
    requested_wall_ns: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControlCommand":
        required = {
            "control_command_id",
            "control_generation",
            "actuator_name",
            "requested_value",
            "expected_current_value",
            "requested_wall_ns",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ControlValidationError(f"missing fields: {missing}")
        command_id = str(value["control_command_id"]).strip()
        if not command_id or len(command_id) > 128:
            raise ControlValidationError("control_command_id must contain 1..128 characters")
        generation = value["control_generation"]
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            raise ControlValidationError("control_generation must be a positive integer")
        requested_wall_ns = value["requested_wall_ns"]
        if isinstance(requested_wall_ns, bool) or not isinstance(requested_wall_ns, int) or requested_wall_ns <= 0:
            raise ControlValidationError("requested_wall_ns must be a positive integer")
        return cls(
            control_command_id=command_id,
            control_generation=generation,
            actuator_name=str(value["actuator_name"]),
            requested_value=value["requested_value"],
            expected_current_value=value["expected_current_value"],
            requested_wall_ns=requested_wall_ns,
        )


def command_payload(command: ControlCommand) -> dict[str, Any]:
    return {
        "control_command_id": command.control_command_id,
        "control_generation": command.control_generation,
        "actuator_name": command.actuator_name,
        "requested_value": command.requested_value,
        "expected_current_value": command.expected_current_value,
        "requested_wall_ns": command.requested_wall_ns,
    }
