from __future__ import annotations

import numpy as np

from servingrom_control_modeling.forcing import (
    DifferentialIncrementModel,
    FrozenGlobalModel,
    _augmented_spectral_radius,
    _design,
    _fit_model,
    _forcing_values,
    _transition_rows,
)
from servingrom_control_modeling.pipeline import RunRange


def test_effective_forcing_is_signed_a_minus_b() -> None:
    row = {
        "routed_request_count": 9,
        "routed_A_request_count": 6,
        "routed_expected_token_mass": 1000,
        "routed_A_expected_token_mass": 700,
    }
    assert _forcing_values(row) == (3.0, 400.0)


def test_transition_rows_never_cross_runs() -> None:
    runs = [RunRange("a", "train", 0, 4), RunRange("b", "train", 4, 8)]
    assert _transition_rows(runs).tolist() == [1, 2, 3, 5, 6, 7]


def test_candidate_design_has_only_requested_inputs() -> None:
    state = np.zeros((4, 14)); d = np.zeros((4, 3)); u = np.zeros((4, 1)); forcing = np.zeros((4, 2))
    rows = np.asarray([1, 2, 3])
    command = _design("command_only", state, d, u, forcing, rows)
    actual = _design("actual_forcing_only", state, d, u, forcing, rows)
    both = _design("actual_forcing_plus_command", state, d, u, forcing, rows)
    assert command.shape[1] + 1 == actual.shape[1]
    assert actual.shape[1] + 1 == both.shape[1]


def test_increment_model_recovers_forcing_coefficient() -> None:
    rng = np.random.default_rng(17)
    rows = 500
    state = np.zeros((rows, 14)); target = np.zeros_like(state)
    d = rng.normal(size=(rows, 2)); u = rng.normal(size=(rows, 1)); forcing = rng.normal(size=(rows, 2))
    for index in range(1, rows):
        delta = np.asarray([0.4, -0.2]) * forcing[index, 0] + np.asarray([0.1, 0.3]) * forcing[index, 1]
        target[index, -2:] = state[index, -2:] + delta
        if index + 1 < rows:
            state[index + 1, -2:] = target[index, -2:]
    model = _fit_model(
        "actual_forcing_only", 1e-8, state, target, d, u, forcing,
        [RunRange("synthetic", "train", 0, rows)],
    )
    assert np.allclose(model.Bf[:, 0], [0.4, -0.2], atol=1e-4)
    assert np.allclose(model.Bf[:, 1], [0.1, 0.3], atol=1e-4)
    assert np.allclose(model.Bu, 0.0)


def test_augmented_radius_includes_increment_integrator() -> None:
    global_model = FrozenGlobalModel(
        A=np.zeros((12, 14)), L=np.zeros((12, 14)), E=np.zeros((12, 1)),
        M=np.zeros((12, 1)), B=np.zeros((12, 1)), c=np.zeros(12),
    )
    model = DifferentialIncrementModel(
        candidate="actual_forcing_only", ridge=0.0,
        K=-0.5 * np.eye(2), L=np.zeros((2, 2)), C=np.zeros((2, 12)),
        E=np.zeros((2, 1)), M=np.zeros((2, 1)), Bf=np.zeros((2, 2)),
        Bu=np.zeros((2, 1)), c=np.zeros(2),
    )
    assert np.isclose(_augmented_spectral_radius(global_model, model), 0.5)
