from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingSafetyConfig:
    minimum_rho_a: float = 0.2
    maximum_rho_a: float = 0.8
    maximum_step: float = 0.2
    minimum_dwell_ns: int = 5_000_000_000
    prepare_ttl_ns: int = 30_000_000_000
    maximum_load_skew_tokens: float = 2048.0
    recent_window_size: int = 100
