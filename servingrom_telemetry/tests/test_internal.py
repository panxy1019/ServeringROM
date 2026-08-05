from __future__ import annotations

from servingrom_telemetry.emitter import NullEmitter
from servingrom_telemetry.internal import (
    EngineIdentity,
    InternalTelemetry,
    get_internal_telemetry,
    reset_internal_telemetry_for_tests,
)


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def emit(self, event_type, payload, **ids):
        self.events.append((event_type, dict(payload), ids.get("request_id")))
        return True

    def flush(self, timeout_s=None):
        return True

    def close(self, timeout_s=None):
        return True

    def health_snapshot(self):
        return {}


def test_disabled_internal_telemetry_is_inert() -> None:
    telemetry = InternalTelemetry(NullEmitter())
    assert not telemetry.enabled
    assert telemetry.emit("ignored", {"value": 1}) is False


def test_iteration_ids_are_strictly_increasing() -> None:
    telemetry = InternalTelemetry(RecordingEmitter())
    assert [telemetry.next_iteration_id() for _ in range(4)] == [1, 2, 3, 4]


def test_internal_event_includes_engine_identity(monkeypatch) -> None:
    monkeypatch.setenv("SERVINGROM_ENGINE_ROLE", "decode")
    monkeypatch.setenv("SERVINGROM_ENGINE_INSTANCE", "decode-1")
    monkeypatch.setenv("SERVINGROM_TP_RANK", "0")
    monkeypatch.setenv("SERVINGROM_IS_DRIVER_RANK", "true")
    emitter = RecordingEmitter()
    telemetry = InternalTelemetry(emitter)
    telemetry.identity = EngineIdentity.from_env()

    assert telemetry.emit("scheduler_iteration", {"iteration_id": 7})
    _, payload, _ = emitter.events[0]
    assert payload == {
        "engine_role": "decode",
        "engine_instance": "decode-1",
        "tp_rank": 0,
        "is_driver_rank": True,
        "iteration_id": 7,
    }


def test_emitter_failure_is_isolated() -> None:
    class BrokenEmitter(RecordingEmitter):
        def emit(self, event_type, payload, **ids):
            raise RuntimeError("sink failed")

    telemetry = InternalTelemetry(BrokenEmitter())
    assert telemetry.emit("scheduler_iteration", {}) is False


def test_named_emitter_inherits_enabled_worker_config(monkeypatch, tmp_path) -> None:
    reset_internal_telemetry_for_tests()
    monkeypatch.setenv("SERVINGROM_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("SERVINGROM_EXPERIMENT_ID", "experiment")
    monkeypatch.setenv("SERVINGROM_RUN_ID", "run")
    monkeypatch.setenv("SERVINGROM_CONFIG_ID", "config")
    monkeypatch.setenv("SERVINGROM_COMPONENT", "prefill")
    monkeypatch.setenv("SERVINGROM_OUTPUT_DIR", str(tmp_path / "prefill"))
    default = get_internal_telemetry()
    assert default.enabled

    monkeypatch.setenv("SERVINGROM_TELEMETRY_ENABLED", "false")
    mooncake = get_internal_telemetry("mooncake")
    assert mooncake.enabled
    assert mooncake.emit("kv_transfer_started", {"total_bytes": 1})
    assert mooncake.close(5)
    assert list((tmp_path / "mooncake").glob("*.jsonl"))
    reset_internal_telemetry_for_tests()
