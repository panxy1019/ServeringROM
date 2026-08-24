from __future__ import annotations

import numpy as np

from servingrom_control_modeling.transition_inventory import (
    SharedFlowModel,
    _add_interval,
    _aggregate,
    _replay,
    _stage,
)


def test_stage_boundaries_are_half_open() -> None:
    assert _stage(10, 10, 20, 30, 40) == 0
    assert _stage(20, 10, 20, 30, 40) == 1
    assert _stage(30, 10, 20, 30, 40) == 2
    assert _stage(40, 10, 20, 30, 40) is None


def test_observed_replay_exact_simple_transition() -> None:
    state = np.zeros((3, 2, 3, 2))
    flow = {name: np.zeros((2, 2, 2)) for name in ("route", "kv_ready", "admission", "service_handoff", "service_waiting", "service_running", "terminal")}
    flow["route"][0, 0] = [1, 10]
    flow["kv_ready"][1, 0] = [1, 10]
    state[1, 0, 0] = [1, 10]
    state[2, 0, 1] = [1, 10]
    prediction, residual = _replay({"state": state, **flow})
    assert np.array_equal(prediction, state)
    assert all(np.count_nonzero(value) == 0 for value in residual.values())


def test_token_emission_at_boundary_is_visible_after_that_boundary() -> None:
    boundaries = np.asarray([0, 200, 400], dtype=np.int64)
    count_delta = np.zeros((len(boundaries) + 1, 2, 3))
    token_delta = np.zeros_like(count_delta)
    _add_interval(
        count_delta,
        token_delta,
        side=0,
        stage=2,
        start_ts=0,
        end_ts=400,
        expected=10,
        emissions=[(200, 3)],
        starts=boundaries,
    )
    snapshots = np.cumsum(token_delta, axis=0)[:len(boundaries)]
    assert np.array_equal(snapshots[:, 0, 2], np.asarray([10.0, 10.0, 0.0]))


def test_emission_before_stage_snapshot_is_not_subtracted_twice() -> None:
    boundaries = np.asarray([0, 200, 400, 600], dtype=np.int64)
    count_delta = np.zeros((len(boundaries) + 1, 2, 3))
    token_delta = np.zeros_like(count_delta)
    _add_interval(
        count_delta,
        token_delta,
        side=0,
        stage=2,
        start_ts=350,
        end_ts=600,
        expected=10,
        emissions=[(375, 1), (450, 1)],
        starts=boundaries,
    )
    snapshots = np.cumsum(token_delta, axis=0)[:len(boundaries)]
    assert np.array_equal(snapshots[:, 0, 2], np.asarray([0.0, 0.0, 9.0, 0.0]))


def test_aggregate_preserves_boundary_and_sums_flows() -> None:
    state = np.arange(11 * 2 * 3 * 2).reshape(11, 2, 3, 2)
    run = {"state": state}
    for name in ("route", "kv_ready", "admission", "service_handoff", "service_waiting", "service_running", "terminal"):
        run[name] = np.ones((10, 2, 2))
    value = _aggregate(run, 5)
    assert np.array_equal(value["state"], state[[0, 5, 10]])
    assert np.all(value["route"] == 5)


def test_shared_flow_model_clips_to_available_inventory() -> None:
    theta = {name: np.ones((8, 2)) * 100 for name in ("kv_ready", "admission", "running_outflow")}
    scales = {name: np.ones(8) for name in theta}; target = {name: np.ones(2) for name in theta}
    model = SharedFlowModel(0.0, theta, scales, target)
    source = np.asarray([[1.0, 10.0], [2.0, 20.0]]); inflow = np.asarray([[1.0, 5.0], [0.0, 0.0]])
    prediction = model.predict("kv_ready", source, inflow)
    assert np.array_equal(prediction, source + inflow)
