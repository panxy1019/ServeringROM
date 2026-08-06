"""ServingROM side-band telemetry primitives.

This package is intentionally independent from Proxy, vLLM, vLLM-Ascend, and
Mooncake. Runtime integration belongs to later phases.
"""

from .config import TelemetryConfig
from .emitter import AsyncTelemetryEmitter, NullEmitter, RunControlEmitter, create_emitter
from .internal import EngineIdentity, InternalTelemetry, get_internal_telemetry
from .run_metadata import (
    RunLayout,
    build_component_inventory,
    build_sha256_manifest,
    write_run_metadata,
)

__all__ = [
    "AsyncTelemetryEmitter",
    "NullEmitter",
    "RunControlEmitter",
    "TelemetryConfig",
    "EngineIdentity",
    "InternalTelemetry",
    "RunLayout",
    "build_sha256_manifest",
    "build_component_inventory",
    "create_emitter",
    "get_internal_telemetry",
    "write_run_metadata",
]
