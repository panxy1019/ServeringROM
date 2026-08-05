"""ServingROM side-band telemetry primitives.

This package is intentionally independent from Proxy, vLLM, vLLM-Ascend, and
Mooncake. Runtime integration belongs to later phases.
"""

from .config import TelemetryConfig
from .emitter import AsyncTelemetryEmitter, NullEmitter, create_emitter

__all__ = [
    "AsyncTelemetryEmitter",
    "NullEmitter",
    "TelemetryConfig",
    "create_emitter",
]
