from __future__ import annotations

import numpy as np

from servingrom_control_modeling.age_transition import HazardModel, _advance, _bin_index


def test_age_bin_edges_are_half_open() -> None:
    edges = np.asarray([0.0, 0.2, 0.4, np.inf])
    assert _bin_index(edges, 0.0) == 0
    assert _bin_index(edges, 0.199) == 0
    assert _bin_index(edges, 0.2) == 1
    assert _bin_index(edges, 9.0) == 2


def test_advance_conserves_each_quantity() -> None:
    state = np.zeros((2, 3, 2))
    state[0, 0] = [2.0, 20.0]
    inflow = np.asarray([[1.0, 10.0], [0.0, 0.0]])
    hazards = np.asarray([[0.25, 0.5], [0.5, 0.5], [1.0, 1.0]])
    next_state, outflow = _advance(state, inflow, hazards)
    assert np.all(next_state >= 0)
    assert np.all(outflow >= 0)
    assert np.allclose(next_state.sum(axis=1) + outflow, state.sum(axis=1) + inflow)


def test_hazard_is_shared_across_decoders() -> None:
    model = HazardModel(
        edges_seconds={"handoff": np.asarray([0.0, 0.2, np.inf])},
        request_hazard={"handoff": np.asarray([0.2, 0.8])},
        token_hazard={"handoff": np.asarray([0.1, 0.7])},
        smoothing=1.0,
    )
    hazards = model.hazard_for_steps("handoff", 3)
    state = np.ones((2, 3, 2))
    next_state, outflow = _advance(state, np.zeros((2, 2)), hazards)
    assert np.array_equal(outflow[0], outflow[1])
    assert np.array_equal(next_state[0], next_state[1])


def test_zero_inventory_and_inflow_stay_zero() -> None:
    state = np.zeros((2, 4, 2)); inflow = np.zeros((2, 2)); hazards = np.full((4, 2), 0.5)
    next_state, outflow = _advance(state, inflow, hazards)
    assert np.count_nonzero(next_state) == 0
    assert np.count_nonzero(outflow) == 0
