from __future__ import annotations

import numpy as np

from servingrom_control_modeling.phase_transition import LogDelayModel, PhaseRecord, _fit_stratified_delay, _phase


def test_phase_is_normalized_inside_window() -> None:
    assert _phase(0) == 0.0
    assert _phase(100_000_000) == 0.5
    assert _phase(250_000_000) == 0.25


def test_log_delay_prediction_is_nonnegative() -> None:
    model = LogDelayModel(("x",), np.asarray([0.0]), np.asarray([1.0]), np.asarray([-100.0, 0.0]), 0.0)
    assert model.predict_ns(np.asarray([[0.0]])).item() == 0.0


def test_constant_delay_model_accepts_empty_features() -> None:
    model = LogDelayModel(tuple(), np.empty(0), np.empty(0), np.asarray([np.log1p(0.025)]), 0.0)
    assert np.isclose(model.predict_ns(np.empty((1, 0))).item(), 25_000_000.0)


def test_stratified_delay_is_shared_across_decoders() -> None:
    records = [
        PhaseRecord("x-balanced-l55-poisson-train-x", side, str(side), 0, 10, delay, delay + 1, delay + 2, 128, 64, 100, 8)
        for side, delay in ((0, 20), (1, 40))
    ]
    model = _fit_stratified_delay(records, "handoff")
    assert model.predict_record_ns(records[0]) == 30.0
    assert model.predict_record_ns(records[1]) == 30.0
