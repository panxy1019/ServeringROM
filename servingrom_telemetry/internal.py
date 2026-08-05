from __future__ import annotations

import os
import atexit
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .emitter import AsyncTelemetryEmitter, Emitter, NullEmitter, create_emitter
from .config import TelemetryConfig


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


_INSTANCES: dict[str, InternalTelemetry] = {}
_INSTANCE_LOCK = threading.Lock()


def get_internal_telemetry(component: str | None = None) -> InternalTelemetry:
    key = component or "__default__"
    instance = _INSTANCES.get(key)
    default_instance = _INSTANCES.get("__default__")
    needs_component_upgrade = (
        component is not None
        and instance is not None
        and not instance.enabled
        and default_instance is not None
        and default_instance.enabled
    )
    if instance is None or needs_component_upgrade:
        with _INSTANCE_LOCK:
            instance = _INSTANCES.get(key)
            default_instance = _INSTANCES.get("__default__")
            needs_component_upgrade = (
                component is not None
                and instance is not None
                and not instance.enabled
                and default_instance is not None
                and default_instance.enabled
            )
            if instance is None or needs_component_upgrade:
                if component is None:
                    instance = InternalTelemetry()
                else:
                    config = TelemetryConfig.from_env()
                    run_root = os.environ.get("SERVINGROM_RUN_ROOT")
                    if (
                        not config.enabled
                        and default_instance is not None
                        and isinstance(default_instance.emitter, AsyncTelemetryEmitter)
                    ):
                        config = default_instance.emitter.config
                    output_dir = Path(run_root) / "raw" / component if run_root else (
                        config.output_dir.parent / component
                    )
                    instance = InternalTelemetry(
                        create_emitter(
                            replace(
                                config,
                                component=component,
                                output_dir=output_dir,
                            )
                        )
                    )
                _INSTANCES[key] = instance
                if instance.enabled:
                    atexit.register(instance.close)
    assert instance is not None
    return instance


def reset_internal_telemetry_for_tests() -> None:
    with _INSTANCE_LOCK:
        for instance in _INSTANCES.values():
            instance.close()
        _INSTANCES.clear()
