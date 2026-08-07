from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import RunSlice
from .preprocessing import Normalizer, chunks


@dataclass
class ReducedModel:
    rank: int
    ridge: float
    theta: np.ndarray
    gamma: np.ndarray
    spectral_radius: float

    def save(self, path) -> None:
        np.savez_compressed(
            path, rank=self.rank, ridge=self.ridge, theta=self.theta,
            gamma=self.gamma, spectral_radius=self.spectral_radius,
        )


def project(array: np.ndarray, normalizer: Normalizer, basis: np.ndarray, rank: int, chunk_size: int) -> np.ndarray:
    output = np.empty((array.shape[0], rank), dtype=np.float64)
    for section in chunks(array.shape[0], chunk_size):
        output[section] = normalizer.transform(array[section]) @ basis[:, :rank]
    return output


def transformed(array: np.ndarray, normalizer: Normalizer, chunk_size: int) -> np.ndarray:
    output = np.empty(array.shape, dtype=np.float64)
    for section in chunks(array.shape[0], chunk_size):
        output[section] = normalizer.transform(array[section], weighted=False)
    return output


def fit_model(z: np.ndarray, z_next: np.ndarray, d: np.ndarray, y: np.ndarray, ridge: float) -> ReducedModel:
    design = np.concatenate([z, d, np.ones((z.shape[0], 1))], axis=1)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[-1, -1] = 0.0
    theta = np.linalg.solve(gram + penalty, design.T @ z_next)
    gamma = np.linalg.solve(gram + penalty, design.T @ y)
    eigenvalues = np.linalg.eigvals(theta[:z.shape[1], :].T)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    return ReducedModel(z.shape[1], ridge, theta, gamma, spectral_radius)


def one_step_metrics(model: ReducedModel, z: np.ndarray, z_next: np.ndarray, d: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.concatenate([z, d, np.ones((z.shape[0], 1))], axis=1)
    z_prediction = design @ model.theta
    y_prediction = design @ model.gamma
    persistence = z
    z_sse = float(np.square(z_prediction - z_next).sum())
    persistence_sse = float(np.square(persistence - z_next).sum())
    y_sse = float(np.square(y_prediction - y).sum())
    y_mean_sse = float(np.square(y).sum())
    return {
        "state_nrmse": float(np.sqrt(z_sse / max(float(np.square(z_next).sum()), 1e-30))),
        "state_skill_vs_persistence": float(1.0 - z_sse / max(persistence_sse, 1e-30)),
        "output_nrmse": float(np.sqrt(y_sse / max(float(np.square(y).sum()), 1e-30))),
        "output_skill_vs_train_mean": float(1.0 - y_sse / max(y_mean_sse, 1e-30)),
    }


def rollout_metrics(
    model: ReducedModel, z: np.ndarray, d: np.ndarray, y: np.ndarray,
    runs: list[RunSlice], output_names: list[str],
) -> dict[str, Any]:
    state_sse = state_den = persistence_sse = 0.0
    output_sse = output_den = 0.0
    per_output_sse = np.zeros(y.shape[1], dtype=np.float64)
    per_output_den = np.zeros(y.shape[1], dtype=np.float64)
    per_run = []
    finite = True
    for run in runs:
        actual_z = z[run.start:run.end]
        actual_y = y[run.start:run.end]
        inputs = d[run.start:run.end]
        current = actual_z[0].copy()
        initial = current.copy()
        run_state_sse = run_state_den = run_output_sse = run_output_den = 0.0
        for offset in range(run.end - run.start):
            design = np.concatenate([current, inputs[offset], np.ones(1)])
            predicted_y = design @ model.gamma
            dz = current - actual_z[offset]
            dy = predicted_y - actual_y[offset]
            run_state_sse += float(dz @ dz)
            run_state_den += float(actual_z[offset] @ actual_z[offset])
            run_output_sse += float(dy @ dy)
            run_output_den += float(actual_y[offset] @ actual_y[offset])
            per_output_sse += np.square(dy)
            per_output_den += np.square(actual_y[offset])
            current = design @ model.theta
            if not np.isfinite(current).all():
                finite = False
                break
        baseline = np.tile(initial, (len(actual_z), 1))
        persistence_sse += float(np.square(baseline - actual_z).sum())
        state_sse += run_state_sse
        state_den += run_state_den
        output_sse += run_output_sse
        output_den += run_output_den
        per_run.append({
            "run_id": run.run_id, "workload": run.workload,
            "arrival_process": run.arrival_process, "transient_pattern": run.transient_pattern,
            "state_nrmse": float(np.sqrt(run_state_sse / max(run_state_den, 1e-30))),
            "output_nrmse": float(np.sqrt(run_output_sse / max(run_output_den, 1e-30))),
            "finite": finite,
        })
        if not finite:
            break
    return {
        "finite": finite,
        "state_nrmse": float(np.sqrt(state_sse / max(state_den, 1e-30))) if finite else float("inf"),
        "state_skill_vs_initial_persistence": float(1.0 - state_sse / max(persistence_sse, 1e-30)) if finite else float("-inf"),
        "output_nrmse": float(np.sqrt(output_sse / max(output_den, 1e-30))) if finite else float("inf"),
        "output_skill_vs_train_mean": float(1.0 - output_sse / max(output_den, 1e-30)) if finite else float("-inf"),
        "per_output_nrmse": {
            name: float(np.sqrt(per_output_sse[i] / max(per_output_den[i], 1e-30)))
            for i, name in enumerate(output_names)
        },
        "per_run": per_run,
    }

