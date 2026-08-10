from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import ControlCommand


@dataclass
class PreparedControl:
    command: ControlCommand
    validated_wall_ns: int
    validated_mono_ns: int
    expires_mono_ns: int
    old_mode: str
    old_value: float | str


@dataclass
class AppliedControl:
    command: ControlCommand
    response: dict[str, Any]
