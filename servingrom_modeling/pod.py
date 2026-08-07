from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .preprocessing import Normalizer, chunks


def fit_pod(array: np.ndarray, normalizer: Normalizer, max_rank: int, chunk_size: int) -> dict[str, np.ndarray]:
    active_index = np.flatnonzero(normalizer.active)
    covariance = np.zeros((len(active_index), len(active_index)), dtype=np.float64)
    for section in chunks(array.shape[0], chunk_size):
        value = normalizer.transform(array[section])[:, active_index]
        covariance += value.T @ value
    covariance /= array.shape[0]
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vectors = vectors[:, order]
    rank = min(max_rank, vectors.shape[1])
    basis = np.zeros((array.shape[1], rank), dtype=np.float64)
    basis[active_index] = vectors[:, :rank]
    singular_values = np.sqrt(eigenvalues * array.shape[0])
    return {"basis": basis, "eigenvalues": eigenvalues, "singular_values": singular_values}


def reconstruction_scan(
    array: np.ndarray, normalizer: Normalizer, basis: np.ndarray,
    ranks: list[int], chunk_size: int,
) -> dict[int, float]:
    residual = {rank: 0.0 for rank in ranks}
    total = 0.0
    for section in chunks(array.shape[0], chunk_size):
        value = normalizer.transform(array[section])
        coefficients = value @ basis[:, :max(ranks)]
        total += float(np.square(value).sum())
        captured = np.cumsum(np.square(coefficients), axis=1).sum(axis=0)
        for rank in ranks:
            residual[rank] += float(np.square(value).sum() - captured[rank - 1])
    return {rank: float(np.sqrt(max(value, 0.0) / max(total, 1e-30))) for rank, value in residual.items()}


def mode_block_contributions(basis: np.ndarray, normalizer: Normalizer, modes: int = 16) -> list[dict[str, Any]]:
    result = []
    block_positions: dict[str, list[int]] = defaultdict(list)
    for index, block in enumerate(normalizer.blocks):
        block_positions[block].append(index)
    for mode in range(min(modes, basis.shape[1])):
        contribution = {
            block: float(np.square(basis[positions, mode]).sum())
            for block, positions in block_positions.items()
        }
        result.append({"mode": mode + 1, "block_contribution": contribution})
    return result

