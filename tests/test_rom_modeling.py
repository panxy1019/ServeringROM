from __future__ import annotations

from unittest import TestCase

import numpy as np

from servingrom_modeling.dynamics import fit_model, one_step_metrics
from servingrom_modeling.preprocessing import fit_normalizer


class RomModelingTest(TestCase):
    def test_block_balanced_normalizer_ignores_constant_dimension(self):
        values = np.asarray([[0.0, 1000.0, 7.0], [1.0, 2000.0, 7.0], [2.0, 4000.0, 7.0]])
        index = [
            {"name": "count", "block": "a", "unit": "requests"},
            {"name": "bytes", "block": "b", "unit": "bytes"},
            {"name": "constant", "block": "b", "unit": "scalar"},
        ]
        normalizer, _ = fit_normalizer(values, index, chunk_size=2)
        transformed = normalizer.transform(values)
        self.assertFalse(normalizer.active[2])
        self.assertTrue(np.all(transformed[:, 2] == 0))
        self.assertAlmostEqual(float(np.var(transformed[:, 0])), 1.0)
        self.assertAlmostEqual(float(np.var(transformed[:, 1])), 1.0)

    def test_linear_dynamics_is_recovered(self):
        rng = np.random.default_rng(7)
        z = rng.normal(size=(500, 2))
        d = rng.normal(size=(500, 1))
        z_next = z @ np.asarray([[0.8, 0.1], [0.0, 0.7]]) + d @ np.asarray([[0.2, -0.1]]) + 0.05
        y = z @ np.asarray([[1.0], [2.0]]) + d * 0.3
        model = fit_model(z, z_next, d, y, ridge=1e-8)
        metrics = one_step_metrics(model, z, z_next, d, y)
        self.assertLess(metrics["state_nrmse"], 1e-8)
        self.assertLess(metrics["output_nrmse"], 1e-8)

