from __future__ import annotations

import numpy as np

from servingrom_control_modeling.pipeline import (
    DynamicModel,
    RunRange,
    _fit_dynamic,
    _rollout,
    _transition_rows,
)


def test_transition_rows_do_not_cross_run_boundaries() -> None:
    runs = [RunRange("a", "train", 0, 4), RunRange("b", "train", 4, 8)]
    assert _transition_rows(runs).tolist() == [1, 2, 3, 5, 6, 7]


def test_augmented_spectral_radius_includes_delta_memory() -> None:
    model = DynamicModel(
        rank=1,
        ridge=0.0,
        model_type="linear",
        A=np.asarray([[0.5]]),
        L=np.asarray([[0.2]]),
        E=np.zeros((1, 1)),
        M=np.zeros((1, 1)),
        B=np.zeros((1, 1)),
        c=np.zeros(1),
    )
    expected = max(abs(np.linalg.eigvals(np.asarray([[0.7, -0.2], [1.0, 0.0]]))))
    assert np.isclose(model.spectral_radius(), expected)


def test_linear_control_model_recovers_synthetic_dynamics() -> None:
    rng = np.random.default_rng(7)
    rows = 240
    z = np.zeros((rows, 2))
    d = rng.normal(size=(rows, 1))
    u = rng.normal(size=(rows, 1))
    d[0] = 0.0
    u[0] = 0.0
    z_next = np.zeros_like(z)
    for index in range(1, rows):
        z_next[index] = (
            0.65 * z[index]
            + 0.10 * (z[index] - z[index - 1])
            + np.asarray([0.2, -0.1]) * d[index, 0]
            + np.asarray([0.4, -0.3]) * u[index, 0]
        )
        if index + 1 < rows:
            z[index + 1] = z_next[index]
    run = [RunRange("synthetic", "train", 0, rows)]
    model = _fit_dynamic(z, z_next, d, u, run, 1e-8, bilinear=False)
    assert np.allclose(model.B[:, 0], [0.4, -0.3], atol=1e-4)
    result = _rollout(model, z, d, u, run)
    assert result["state_nrmse"] < 1e-3


def test_bilinear_term_is_explicit() -> None:
    model = DynamicModel(
        rank=1, ridge=0.0, model_type="bilinear",
        A=np.zeros((1, 1)), L=np.zeros((1, 1)), E=np.zeros((1, 1)),
        M=np.zeros((1, 1)), B=np.zeros((1, 1)), c=np.zeros(1), N=np.asarray([[2.0]]),
    )
    value = model.predict(
        np.asarray([[3.0]]), np.asarray([[3.0]]), np.zeros((1, 1)),
        np.zeros((1, 1)), np.asarray([[0.5]]),
    )
    assert value[0, 0] == 3.0
