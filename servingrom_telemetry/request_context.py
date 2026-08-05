from __future__ import annotations

import time
from dataclasses import dataclass

from .ids import new_request_id, new_trace_id


@dataclass(slots=True)
class RequestTraceContext:
    trace_id: str
    external_request_id: str | None
    attempt_id: int
    request_id: str
    arrival_wall_ns: int
    arrival_mono_ns: int

    @classmethod
    def create(cls, external_request_id: str | None = None) -> "RequestTraceContext":
        return cls(
            trace_id=new_trace_id(),
            external_request_id=external_request_id,
            attempt_id=0,
            request_id=new_request_id(),
            arrival_wall_ns=time.time_ns(),
            arrival_mono_ns=time.monotonic_ns(),
        )

    def begin_recompute(self) -> tuple[int, str]:
        previous = (self.attempt_id, self.request_id)
        self.attempt_id += 1
        self.request_id = new_request_id()
        return previous
