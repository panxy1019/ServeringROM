from __future__ import annotations

from unittest import TestCase

import numpy as np

from servingrom_modeling.dataset import RunSlice
from servingrom_v2.pipeline import (
    aggregate_run_array,
    augment_markov_state,
    select_parsimonious_output_candidate,
)


class RomV2Test(TestCase):
    def test_flow_aggregation_is_conservative(self):
        values = np.arange(24, dtype=np.float64).reshape(12, 2)
        aggregated = aggregate_run_array(values, 3)
        self.assertEqual(aggregated.shape, (4, 2))
        np.testing.assert_allclose(aggregated.sum(axis=0), values.sum(axis=0))

    def test_markov_memory_resets_at_run_boundary(self):
        state = np.asarray([[1.0], [3.0], [10.0], [13.0]])
        state_next = np.asarray([[3.0], [5.0], [13.0], [15.0]])
        disturbance = np.asarray([[2.0], [4.0], [7.0], [8.0]])
        runs = [
            RunSlice("a", "train", 0, 2, "w", "p", None),
            RunSlice("b", "train", 2, 4, "w", "p", None),
        ]
        augmented, augmented_next = augment_markov_state(state, state_next, disturbance, runs)
        np.testing.assert_allclose(augmented[:, 1], [0.0, 2.0, 0.0, 3.0])
        np.testing.assert_allclose(augmented[:, 2], [0.0, 2.0, 0.0, 7.0])
        np.testing.assert_allclose(augmented_next[:, 2], disturbance[:, 0])

    def test_output_selection_prefers_smaller_rank_inside_error_band(self):
        def row(rank, error):
            return {
                "rank": rank,
                "factor": 25,
                "memory": True,
                "validation": {"per_output_nrmse": {"goodput": error}},
            }

        selected = select_parsimonious_output_candidate(
            [row(64, 0.4663), row(16, 0.4667)], ["goodput"], tolerance=0.005,
        )
        self.assertEqual(selected["rank"], 16)
