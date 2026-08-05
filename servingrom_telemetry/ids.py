from __future__ import annotations

import hashlib
import threading
import uuid


def new_trace_id() -> str:
    return str(uuid.uuid4())


def new_request_id() -> str:
    return str(uuid.uuid4())


def build_process_instance_id(
    *,
    host_id: str,
    component: str,
    process_id: int,
    process_start_wall_ns: int,
    process_start_mono_ns: int,
    nonce: str | None = None,
) -> str:
    material = "\0".join(
        (
            host_id,
            component,
            str(process_id),
            str(process_start_wall_ns),
            str(process_start_mono_ns),
            nonce or uuid.uuid4().hex,
        )
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:32]


class EventSequence:
    """A process-local, thread-safe, strictly increasing sequence."""

    __slots__ = ("_lock", "_value")

    def __init__(self, start: int = 0) -> None:
        if start < 0:
            raise ValueError("start must not be negative")
        self._value = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
