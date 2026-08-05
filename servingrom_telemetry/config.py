from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise ValueError(f"{name} must be one of true/false, 1/0, yes/no, on/off")


def _parse_positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    enabled: bool = False
    experiment_id: str = "unset"
    run_id: str = "unset"
    config_id: str = "unset"
    component: str = "unset"
    host_id: str = ""
    output_dir: Path = Path("results/servingrom/raw")
    queue_capacity: int = 65_536
    batch_size: int = 1_024
    flush_interval_ms: int = 250
    max_file_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.host_id:
            object.__setattr__(self, "host_id", socket.gethostname())
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        for name in (
            "queue_capacity",
            "batch_size",
            "flush_interval_ms",
            "max_file_bytes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.batch_size > self.queue_capacity:
            raise ValueError("batch_size cannot exceed queue_capacity")
        if self.enabled:
            for name in ("experiment_id", "run_id", "config_id", "component", "host_id"):
                if not getattr(self, name) or getattr(self, name) == "unset":
                    raise ValueError(f"{name} is required when telemetry is enabled")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "TelemetryConfig":
        values = os.environ if env is None else env
        enabled = _parse_bool(
            "SERVINGROM_TELEMETRY_ENABLED",
            values.get("SERVINGROM_TELEMETRY_ENABLED", "false"),
        )
        return cls(
            enabled=enabled,
            experiment_id=values.get("SERVINGROM_EXPERIMENT_ID", "unset"),
            run_id=values.get("SERVINGROM_RUN_ID", "unset"),
            config_id=values.get("SERVINGROM_CONFIG_ID", "unset"),
            component=values.get("SERVINGROM_COMPONENT", "unset"),
            host_id=values.get("SERVINGROM_HOST_ID", socket.gethostname()),
            output_dir=Path(values.get("SERVINGROM_OUTPUT_DIR", "results/servingrom/raw")),
            queue_capacity=_parse_positive_int(
                "SERVINGROM_QUEUE_CAPACITY",
                values.get("SERVINGROM_QUEUE_CAPACITY", "65536"),
            ),
            batch_size=_parse_positive_int(
                "SERVINGROM_BATCH_SIZE",
                values.get("SERVINGROM_BATCH_SIZE", "1024"),
            ),
            flush_interval_ms=_parse_positive_int(
                "SERVINGROM_FLUSH_INTERVAL_MS",
                values.get("SERVINGROM_FLUSH_INTERVAL_MS", "250"),
            ),
            max_file_bytes=_parse_positive_int(
                "SERVINGROM_MAX_FILE_BYTES",
                values.get("SERVINGROM_MAX_FILE_BYTES", str(256 * 1024 * 1024)),
            ),
        )
