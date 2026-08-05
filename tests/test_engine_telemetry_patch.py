from __future__ import annotations

import ast
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from servingrom_telemetry.internal import InternalTelemetry


def load_patched_engine_core():
    source_path = Path("/vllm-workspace/vllm/vllm/v1/engine/core.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EngineCore"
    )
    wanted = {
        "_servingrom_queue_counts",
        "_servingrom_schedule",
        "_servingrom_execute_model",
        "_servingrom_update_from_output",
    }
    methods = [node for node in engine_class.body if getattr(node, "name", None) in wanted]
    assert {node.name for node in methods} == wanted
    synthetic = ast.Module(
        body=[ast.ClassDef(name="PatchedEngineCore", bases=[], keywords=[], body=methods)],
        type_ignores=[],
    )
    ast.fix_missing_locations(synthetic)
    namespace = {"time": time, "SchedulerOutput": object}
    exec(compile(synthetic, str(source_path), "exec"), namespace)
    return namespace["PatchedEngineCore"], source_path.read_text(encoding="utf-8")


EngineCore, PATCHED_SOURCE = load_patched_engine_core()


class RecordingEmitter:
    def __init__(self):
        self.events = []

    def emit(self, event_type, payload, **ids):
        self.events.append((event_type, dict(payload), ids.get("request_id")))
        return True

    def flush(self, timeout_s=None):
        return True

    def close(self, timeout_s=None):
        return True

    def health_snapshot(self):
        return {}


class FakeQueue(list):
    pass


def scheduler_output(request_id: str, scheduled: int, output_before: int = 0):
    cached = SimpleNamespace(
        req_ids=[request_id],
        num_computed_tokens=[32 + output_before],
        num_output_tokens=[output_before],
        resumed_req_ids=set(),
        new_block_ids=[([1, 2],)],
    )
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={request_id: scheduled},
        total_num_scheduled_tokens=scheduled,
        preempted_req_ids=set(),
        finished_req_ids=set(),
    )


class FakeScheduler:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.running = [object()]
        self.waiting = FakeQueue([object()])
        self.skipped_waiting = FakeQueue()
        self.requests = {
            "request-a": SimpleNamespace(status=SimpleNamespace(name="RUNNING")),
            "request-b": SimpleNamespace(status=SimpleNamespace(name="RUNNING")),
        }
        self.max_num_scheduled_tokens = 2560

    def schedule(self):
        return next(self.outputs)

    def update_from_output(self, scheduled, model_output):
        request_id = next(iter(scheduled.num_scheduled_tokens))
        output = SimpleNamespace(
            request_id=request_id,
            new_token_ids=model_output,
            finish_reason=None,
            stop_reason=None,
        )
        return {0: SimpleNamespace(outputs=[output])}


class FakeExecutor:
    def execute_model(self, scheduled, non_block=True):
        future = Future()
        future.set_result([101])
        return future


def build_engine(outputs):
    engine = EngineCore.__new__(EngineCore)
    recorder = RecordingEmitter()
    engine._servingrom = InternalTelemetry(recorder)
    engine.scheduler = FakeScheduler(outputs)
    engine.model_executor = FakeExecutor()
    engine.async_scheduling = True
    return engine, recorder


def test_two_inflight_iterations_keep_distinct_ids():
    engine, recorder = build_engine(
        [scheduler_output("request-a", 32), scheduler_output("request-b", 1, 4)]
    )
    first = engine._servingrom_schedule()
    second = engine._servingrom_schedule()
    first_future = engine._servingrom_execute_model(first)
    second_future = engine._servingrom_execute_model(second)
    engine._servingrom_update_from_output(first, first_future.result())
    engine._servingrom_update_from_output(second, second_future.result())

    iterations = [
        payload["iteration_id"]
        for event, payload, _ in recorder.events
        if event == "scheduler_iteration"
    ]
    assert iterations == [1, 2]
    memberships = [
        payload for event, payload, _ in recorder.events
        if event == "scheduler_membership"
    ]
    assert sum(item["scheduled_tokens"] for item in memberships[0]["members"]) == 32
    assert memberships[1]["members"][0]["output_tokens_before"] == 4


def test_disabled_helper_does_not_change_scheduler_result():
    engine, _ = build_engine([scheduler_output("request-a", 1)])
    scheduled = engine._servingrom_schedule()
    future = engine._servingrom_execute_model(scheduled)
    result = engine._servingrom_update_from_output(scheduled, future.result())
    assert result[0].outputs[0].new_token_ids == [101]


if __name__ == "__main__":
    assert '"engine_request_added"' in PATCHED_SOURCE
    assert '"engine_request_aborted"' in PATCHED_SOURCE
    assert "self._servingrom_schedule()" in PATCHED_SOURCE
    assert "self._servingrom_update_from_output(" in PATCHED_SOURCE
    test_two_inflight_iterations_keep_distinct_ids()
    test_disabled_helper_does_not_change_scheduler_result()
    print("ServingROM EngineCore telemetry tests: PASS")
