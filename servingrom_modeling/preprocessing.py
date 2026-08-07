from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOG_UNITS = {"requests", "tokens", "bytes", "blocks", "ms"}


def chunks(rows: int, size: int) -> Iterable[slice]:
    for start in range(0, rows, size):
        yield slice(start, min(start + size, rows))


def _transform_raw(values: np.ndarray, log_mask: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    if log_mask.any():
        output[:, log_mask] = np.log1p(np.maximum(output[:, log_mask], 0.0))
    return output


@dataclass
class Normalizer:
    names: list[str]
    blocks: list[str]
    log_mask: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray
    block_weight: np.ndarray

    def transform(self, values: np.ndarray, *, weighted: bool = True) -> np.ndarray:
        output = _transform_raw(values, self.log_mask)
        output -= self.mean
        output /= self.scale
        output[:, ~self.active] = 0.0
        if weighted:
            output *= self.block_weight
        return output

    def inverse(self, values: np.ndarray, *, weighted: bool = True) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64).copy()
        if weighted:
            output /= self.block_weight
        output *= self.scale
        output += self.mean
        if self.log_mask.any():
            output[:, self.log_mask] = np.expm1(output[:, self.log_mask])
        return output

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": "servingrom.normalizer.v1",
            "fit_split": "train",
            "names": self.names,
            "blocks": self.blocks,
            "log_mask": self.log_mask.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "active": self.active.tolist(),
            "block_weight": self.block_weight.tolist(),
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Normalizer":
        return cls(
            names=value["names"], blocks=value["blocks"],
            log_mask=np.asarray(value["log_mask"], dtype=bool),
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            active=np.asarray(value["active"], dtype=bool),
            block_weight=np.asarray(value["block_weight"], dtype=np.float64),
        )


def fit_normalizer(array: np.ndarray, index: list[dict[str, Any]], chunk_size: int) -> tuple[Normalizer, dict[str, Any]]:
    dimensions = array.shape[1]
    names = [row["name"] for row in index]
    blocks = [str(row.get("block") or "unknown") for row in index]
    log_mask = np.asarray([str(row.get("unit") or "") in LOG_UNITS for row in index], dtype=bool)
    total = np.zeros(dimensions, dtype=np.float64)
    total_sq = np.zeros(dimensions, dtype=np.float64)
    nonzero = np.zeros(dimensions, dtype=np.int64)
    raw_min = np.full(dimensions, np.inf)
    raw_max = np.full(dimensions, -np.inf)
    for section in chunks(array.shape[0], chunk_size):
        raw = np.asarray(array[section], dtype=np.float64)
        transformed = _transform_raw(raw, log_mask)
        total += transformed.sum(axis=0)
        total_sq += np.square(transformed).sum(axis=0)
        nonzero += np.count_nonzero(raw, axis=0)
        raw_min = np.minimum(raw_min, raw.min(axis=0))
        raw_max = np.maximum(raw_max, raw.max(axis=0))
    mean = total / array.shape[0]
    variance = np.maximum(total_sq / array.shape[0] - np.square(mean), 0.0)
    std = np.sqrt(variance)
    tolerance = np.maximum(1e-12, np.abs(mean) * 1e-10)
    active = std > tolerance
    scale = np.where(active, std, 1.0)
    counts: dict[str, int] = defaultdict(int)
    for block, enabled in zip(blocks, active):
        if enabled:
            counts[block] += 1
    block_weight = np.asarray([
        1.0 / np.sqrt(counts[block]) if active[i] and counts[block] else 1.0
        for i, block in enumerate(blocks)
    ], dtype=np.float64)
    normalizer = Normalizer(names, blocks, log_mask, mean, scale, active, block_weight)
    feature_rows = []
    for i, row in enumerate(index):
        feature_rows.append({
            **row, "train_mean_transformed": float(mean[i]),
            "train_std_transformed": float(std[i]), "train_nonzero_ratio": float(nonzero[i] / array.shape[0]),
            "train_min_raw": float(raw_min[i]), "train_max_raw": float(raw_max[i]),
            "active": bool(active[i]), "log1p": bool(log_mask[i]),
            "block_weight": float(block_weight[i]),
        })
    block_rows = []
    for block in sorted(set(blocks)):
        positions = np.asarray([name == block for name in blocks])
        block_rows.append({
            "block": block, "dimension_count": int(positions.sum()),
            "active_dimension_count": int((positions & active).sum()),
            "raw_variance_sum": float(variance[positions].sum()),
            "balanced_variance_contribution": float(np.square(block_weight[positions & active]).sum()),
        })
    return normalizer, {"features": feature_rows, "blocks": block_rows}


def split_statistics(array: np.ndarray, normalizer: Normalizer, chunk_size: int) -> dict[str, Any]:
    nonzero = np.zeros(array.shape[1], dtype=np.int64)
    normalized_sum = np.zeros(array.shape[1], dtype=np.float64)
    normalized_sq = np.zeros(array.shape[1], dtype=np.float64)
    for section in chunks(array.shape[0], chunk_size):
        raw = np.asarray(array[section])
        value = normalizer.transform(raw, weighted=False)
        nonzero += np.count_nonzero(raw, axis=0)
        normalized_sum += value.sum(axis=0)
        normalized_sq += np.square(value).sum(axis=0)
    mean = normalized_sum / array.shape[0]
    variance = np.maximum(normalized_sq / array.shape[0] - np.square(mean), 0.0)
    return {
        "rows": int(array.shape[0]), "nonzero_ratio": (nonzero / array.shape[0]).tolist(),
        "normalized_mean": mean.tolist(), "normalized_std": np.sqrt(variance).tolist(),
    }


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

