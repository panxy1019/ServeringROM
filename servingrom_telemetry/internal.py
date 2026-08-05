from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .emitter import Emitter, NullEmitter, create_emitter


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class EngineIdentity:
    engine_role: str
    engine_instance: str
    tp_rank: int | None
    is_driver_rank: bool

    @classmethod
    def from_env(cls) -> "EngineIdentity":
        raw_rank = os.environ.get("SERVINGROM_TP_RANK")
        return cls(
            engine_role=os.environ.get("SERVINGROM_ENGINE_ROLE", "unknown"),
            engine_instance=os.environ.get("SERVINGROM_ENGINE_INSTANCE", "unknown"),
            tp_rank=int(raw_rank) if raw_rank not in (None, "") else None,
            is_driver_rank=_env_bool("SERVINGROM_IS_DRIVER_RANK", True),
        )


class InternalTelemetry:
    """Exception-isolated adapter used by patched inference components."""

    __slots__ = ("emitter", "identity", "enabled", "_iteration", "_lock")

    def __init__(self, emitter: Emitter | None = None) -> None:
        self.emitter = create_emitter() if emitter is None else emitter
        self.identity = EngineIdentity.from_env()
        self.enabled = not isinstance(self.emitter, NullEmitter)
        self._iteration = 0
        self._lock = threading.Lock()

    def next_iteration_id(self) -> int:
        with self._lock:
            self._iteration += 1
            return self._iteration

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        try:
            enriched = {
                "engine_role": self.identity.engine_role,
                "engine_instance": self.identity.engine_instance,
                "tp_rank": self.identity.tp_rank,
                "is_driver_rank": self.identity.is_driver_rank,
                **payload,
            }
            return self.emitter.emit(
                event_type,
                enriched,
                request_id=request_id,
            )
        except Exception:
            return False

    def close(self, timeout_s: float | None = 5.0) -> bool:
        try:
            return self.emitter.close(timeout_s)
        except Exception:
            return False


_INSTANCE: InternalTelemetry | None = None
_INSTANCE_LOCK = threading.Lock()


def get_internal_telemetry() -> InternalTelemetry:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = InternalTelemetry()
    return _INSTANCE


def reset_internal_telemetry_for_tests() -> None:
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            _INSTANCE.close()
        _INSTANCE = None
