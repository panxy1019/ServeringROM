from __future__ import annotations

import math

import numpy as np

from servingrom_control_modeling.memory import (
    _forcing_audit,
    _history_feature,
    _memory_matrix,
    _valid_rows,
)
from servingrom_control_modeling.pipeline import RunRange


def test_raw_memory_does_not_cross_run_boundary() -> None:
    signal = np.arange(8, dtype=np.float64)[:, None]
    runs = [RunRange("a", "train", 0, 4), RunRange("b", "train", 4, 8)]
    memory = _memory_matrix("raw_lag", 2, (2,), signal, runs, 0.2)
    assert memory[4].tolist() == [4.0, 4.0]
    assert memory[5].tolist() == [4.0, 4.0]
    assert _valid_rows(runs, 2).tolist() == [2, 3, 6, 7]


def test_rollout_history_uses_run_local_predicted_state() -> None:
    states = np.zeros((4, 14), dtype=np.float64)
    states[:, -2:] = [[10, 11], [20, 21], [30, 31], [40, 41]]
    disturbance = np.arange(30, dtype=np.float64).reshape(10, 3)
    control = np.arange(10, dtype=np.float64)[:, None]
    result = _history_feature(
        "raw_lag", 2, (2,), states, disturbance, control,
        np.asarray([0, 2]), run_start=5, index=8, dt=0.2, exp_memory=None,
    )
    assert result[:2].tolist() == [30.0, 31.0]
    assert result[2] == 7.0
    assert result[3:5].tolist() == [21.0, 23.0]
    assert result[5:7].tolist() == [20.0, 21.0]


def test_exponential_memory_is_causal_and_run_local() -> None:
    signal = np.asarray([[0.0], [10.0], [20.0], [100.0], [110.0], [120.0]])
    runs = [RunRange("a", "train", 0, 3), RunRange("b", "train", 3, 6)]
    memory = _memory_matrix("exponential", 5, (5,), signal, runs, 0.2)
    alpha = math.exp(-1.0 / 5.0)
    assert memory[0, 0] == 0.0
    assert memory[1, 0] == 0.0
    assert np.isclose(memory[2, 0], (1.0 - alpha) * 10.0)
    assert memory[3, 0] == 100.0


def test_effective_forcing_is_audited_but_not_used_in_step15c1() -> None:
    audit = _forcing_audit()
    assert audit["effective_forcing_available"] is True
    assert audit["used_by_step15c1_model"] is False
    assert "routed_expected_token_mass_imbalance" in audit["fields"]
