from __future__ import annotations

import numpy as np

from servingrom_control_modeling.conservation import (
    ConservationModel,
    _design,
    _scale_only,
    _symmetry_audit,
    _transform_scale_only,
)


def model(candidate: str) -> ConservationModel:
    return ConservationModel(
        candidate=candidate, ridge=0.0,
        B0=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        B1=np.asarray([[0.2, 0.0], [0.0, 0.2]]),
        Q=-0.1 * np.eye(2),
        interactions=np.asarray([0.01 * np.eye(2), -0.02 * np.eye(2)]),
    )


def test_scale_only_preserves_physical_zero() -> None:
    values = np.asarray([[-2.0, -20.0], [0.0, 0.0], [2.0, 20.0]])
    normalizer = _scale_only(values, ("a", "b"))
    transformed = _transform_scale_only(values, normalizer)
    assert normalizer["mean_subtracted"] is False
    assert np.array_equal(transformed[1], np.zeros(2))


def test_nested_design_dimensions() -> None:
    q = np.zeros((5, 2)); f = np.zeros((5, 2)); common = np.zeros((5, 2)); rows = np.asarray([1, 2, 3])
    assert _design("M0_inflow_only", q, f, common, rows).shape[1] == 2
    assert _design("M1_minimal_service_closure", q, f, common, rows).shape[1] == 6
    assert _design("M2_load_dependent_service_closure", q, f, common, rows).shape[1] == 10


def test_m1_zero_bias_and_symmetry() -> None:
    audit = _symmetry_audit(model("M1_minimal_service_closure"), np.zeros((20, 2)))
    assert audit["symmetry_pass"]
    assert audit["zero_bias_pass"]


def test_m2_common_load_preserves_odd_symmetry() -> None:
    audit = _symmetry_audit(model("M2_load_dependent_service_closure"), np.ones((20, 2)))
    assert audit["symmetry_pass"]
    assert audit["zero_bias_pass"]


def test_restoring_term_is_stable() -> None:
    transition = model("M1_minimal_service_closure").transition_matrix()
    assert np.allclose(transition, 0.9 * np.eye(2))
