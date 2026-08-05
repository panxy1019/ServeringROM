from __future__ import annotations

import os
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClockSample:
    wall_ns: int
    mono_ns: int


_PROCESS_ID = os.getpid()
_PROCESS_START_WALL_NS = time.time_ns()
_PROCESS_START_MONO_NS = time.monotonic_ns()


def _process_start() -> tuple[int, int]:
    global _PROCESS_ID, _PROCESS_START_WALL_NS, _PROCESS_START_MONO_NS
    process_id = os.getpid()
    if process_id != _PROCESS_ID:
        _PROCESS_ID = process_id
        _PROCESS_START_WALL_NS = time.time_ns()
        _PROCESS_START_MONO_NS = time.monotonic_ns()
    return _PROCESS_START_WALL_NS, _PROCESS_START_MONO_NS


class ProcessClock:
    __slots__ = ("process_start_wall_ns", "process_start_mono_ns")

    def __init__(self) -> None:
        self.process_start_wall_ns, self.process_start_mono_ns = _process_start()

    @staticmethod
    def sample() -> ClockSample:
        return ClockSample(wall_ns=time.time_ns(), mono_ns=time.monotonic_ns())

    @staticmethod
    def duration_ns(start_mono_ns: int, end_mono_ns: int | None = None) -> int:
        end = time.monotonic_ns() if end_mono_ns is None else end_mono_ns
        if end < start_mono_ns:
            raise ValueError("monotonic end precedes start")
        return end - start_mono_ns
