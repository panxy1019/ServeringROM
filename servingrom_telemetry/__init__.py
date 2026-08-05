"""ServingROM side-band telemetry primitives.

This package is intentionally independent from Proxy, vLLM, vLLM-Ascend, and
Mooncake. Runtime integration belongs to later phases.
"""

from .config import TelemetryConfig
from .emitter import AsyncTelemetryEmitter, NullEmitter, create_emitter
from .run_metadata import RunLayout, build_sha256_manifest, write_run_metadata

__all__ = [
    "AsyncTelemetryEmitter",
    "NullEmitter",
    "TelemetryConfig",
    "RunLayout",
    "build_sha256_manifest",
    "create_emitter",
    "write_run_metadata",
]
