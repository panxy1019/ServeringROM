from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from servingrom_telemetry.config import TelemetryConfig
from servingrom_telemetry.emitter import RunControlEmitter


def _write(path: Path, generation: int, active: bool, root: Path, run_id: str) -> None:
    value = {
        "generation": generation,
        "active": active,
        "experiment_id": "experiment",
        "run_id": run_id,
        "config_id": "config",
        "run_root": str(root / run_id),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def _wait_generation(emitter: RunControlEmitter, generation: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if emitter.health_snapshot()["control_generation"] == generation:
            return
        time.sleep(0.02)
    raise AssertionError(f"generation {generation} was not acknowledged")


def test_hot_rotation_isolates_runs_and_resets_sequence(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    ack_dir = tmp_path / "acks"
    config = TelemetryConfig(
        enabled=True, experiment_id="bootstrap", run_id="bootstrap",
        config_id="config", component="proxy", host_id="host",
        output_dir=tmp_path / "bootstrap", queue_capacity=4096,
        batch_size=32, flush_interval_ms=10, max_file_bytes=1 << 20,
    )
    emitter = RunControlEmitter(config, control, ack_dir)
    try:
        _write(control, 1, True, tmp_path, "run-a")
        _wait_generation(emitter, 1)
        threads = [threading.Thread(target=lambda: [emitter.emit("sample", {"v": i}) for i in range(100)]) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        _write(control, 2, False, tmp_path, "run-a")
        _wait_generation(emitter, 2)
        assert not emitter.emit("between_runs", {})
        _write(control, 3, True, tmp_path, "run-b")
        _wait_generation(emitter, 3)
        assert emitter.emit("sample", {"v": 1})
        _write(control, 4, False, tmp_path, "run-b")
        _wait_generation(emitter, 4)
    finally:
        emitter.close(5)

    a_events = [json.loads(line) for path in (tmp_path / "run-a" / "raw" / "proxy").glob("*.jsonl") for line in path.read_text().splitlines()]
    b_events = [json.loads(line) for path in (tmp_path / "run-b" / "raw" / "proxy").glob("*.jsonl") for line in path.read_text().splitlines()]
    assert len(a_events) == 400
    assert len(b_events) == 1
    assert {event["run_id"] for event in a_events} == {"run-a"}
    assert {event["run_id"] for event in b_events} == {"run-b"}
    assert [event["event_seq"] for event in b_events] == [1]
    assert len(list(ack_dir.glob("*.json"))) == 1
