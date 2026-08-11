from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq

from servingrom_modeling.pod import fit_pod
from servingrom_modeling.preprocessing import fit_normalizer, save_json

from .pipeline import (
    DynamicModel,
    IMBALANCE_FEATURES,
    SLOW_OUTPUTS,
    _build_slow,
    _fit_dynamic,
    _fit_scalar_normalizer,
    _fit_slow_head,
    _load_json,
    _load_runs,
    _load_split,
    _nrmse,
    _one_step,
    _predict_slow,
    _project,
    _save_model,
    _scalar_transform,
    _sha256,
    _verify_dataset,
)


CORE_PAIRS = [
    ("running_imbalance", "decode_d1_running_count", "decode_d2_running_count"),
    ("waiting_imbalance", "decode_d1_waiting_count", "decode_d2_waiting_count"),
    (
        "remaining_token_imbalance",
        "decode_d1_expected_remaining_tokens",
        "decode_d2_expected_remaining_tokens",
    ),
]

DIRECT_CONTROL_DESCRIPTORS = [
    ("route_request_imbalance", "decode_route_imbalance_requests"),
    ("route_token_imbalance", "decode_route_imbalance_tokens"),
]

BINNED_FAMILIES = [
    "handoff_wait.count",
    "handoff_wait.bytes",
    "kv_queue.count",
    "kv_queue.bytes",
    "kv_inflight.count",
    "kv_inflight.bytes",
    "kv_ready_wait.count",
    "kv_ready_wait.bytes",
    "wait_count",
    "running_count",
    "remaining_output_mass",
    "context_token_mass",
    "first_token_pending",
]


@dataclass(frozen=True)
class Descriptor:
    name: str
    kind: str
    left: tuple[int, ...]
    right: tuple[int, ...] = ()
    source_names: tuple[str, ...] = ()


def _descriptor_manifest(state_index: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in state_index}
    position = {name: int(row["index"]) for name, row in by_name.items()}
    core = [
        Descriptor(name, "pair_difference", (position[left],), (position[right],), (left, right))
        for name, left, right in CORE_PAIRS
    ]
    direct = [
        Descriptor(name, "direct_existing_state", (position[source],), (), (source,))
        for name, source in DIRECT_CONTROL_DESCRIPTORS if source in position
    ]
    binned = []
    for family in BINNED_FAMILIES:
        left_names = sorted(
            name for name in by_name
            if name.startswith("decode-0.") and family in name
            and name.replace("decode-0.", "decode-1.", 1) in by_name
        )
        if not left_names:
            continue
        right_names = [name.replace("decode-0.", "decode-1.", 1) for name in left_names]
        binned.append(Descriptor(
            f"binned_{family}_imbalance", "paired_bin_sum_difference",
            tuple(position[name] for name in left_names),
            tuple(position[name] for name in right_names),
            tuple(left_names + right_names),
        ))
    candidates = {
        "core3": core,
        "all_scalar": core + direct,
        "selected_binned": core + direct + binned,
    }
    return {
        "schema_version": "servingrom.control-descriptors.v1",
        "pairing_policy": "exact decode_d1/d2 or decode-0/1 name substitution only",
        "candidate_order": list(candidates),
        "candidates": {
            name: [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "left_indices": list(item.left),
                    "right_indices": list(item.right),
                    "source_names": list(item.source_names),
                }
                for item in values
            ]
            for name, values in candidates.items()
        },
        "objects": candidates,
    }


def _extract_descriptors(array: np.ndarray, descriptors: list[Descriptor]) -> np.ndarray:
    output = np.empty((array.shape[0], len(descriptors)), dtype=np.float64)
    for column, descriptor in enumerate(descriptors):
        left = np.asarray(array[:, list(descriptor.left)], dtype=np.float64).sum(axis=1)
        if descriptor.right:
            left -= np.asarray(array[:, list(descriptor.right)], dtype=np.float64).sum(axis=1)
        output[:, column] = left
    return output


def _rollout_augmented(
    model: DynamicModel,
    state: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    runs: list[Any],
    pod_rank: int,
    q_names: list[str],
) -> dict[str, Any]:
    total_error = total_energy = base_error = 0.0
    pod_error = pod_energy = 0.0
    q_error = np.zeros(len(q_names))
    q_energy = np.zeros(len(q_names))
    run_metrics = []
    for run in runs:
        actual = state[run.start:run.end]
        predicted = np.empty_like(actual)
        predicted[0] = actual[0]
        previous = actual[0].copy()
        for offset in range(len(actual) - 1):
            index = run.start + offset
            d_previous = d[index - 1] if offset else d[index]
            current = predicted[offset]
            predicted[offset + 1] = model.predict(
                current[None], previous[None], d[index:index + 1],
                d_previous[None], u[index:index + 1],
            )[0]
            previous = current
        error = predicted - actual
        total_error += float(np.square(error).sum())
        total_energy += float(np.square(actual).sum())
        base_error += float(np.square(actual - actual[0]).sum())
        pod_error += float(np.square(error[:, :pod_rank]).sum())
        pod_energy += float(np.square(actual[:, :pod_rank]).sum())
        q_error += np.square(error[:, pod_rank:]).sum(axis=0)
        q_energy += np.square(actual[:, pod_rank:]).sum(axis=0)
        run_metrics.append({
            "run_id": run.run_id,
            "nrmse": math.sqrt(float(np.square(error).sum()) / max(float(np.square(actual).sum()), 1e-30)),
        })
    state_nrmse = math.sqrt(total_error / max(total_energy, 1e-30))
    persistence = math.sqrt(base_error / max(total_energy, 1e-30))
    return {
        "state_nrmse": state_nrmse,
        "global_pod_nrmse": math.sqrt(pod_error / max(pod_energy, 1e-30)),
        "persistence_nrmse": persistence,
        "skill": 1.0 - state_nrmse / max(persistence, 1e-30),
        "descriptor_nrmse": {
            name: math.sqrt(q_error[index] / max(q_energy[index], 1e-30))
            for index, name in enumerate(q_names)
        },
        "runs": run_metrics,
    }


def _one_step_descriptors(
    model: DynamicModel,
    state: np.ndarray,
    state_next: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    runs: list[Any],
    pod_rank: int,
    q_names: list[str],
) -> dict[str, float]:
    rows = np.concatenate([np.arange(run.start + 1, run.end) for run in runs])
    prediction = model.predict(state[rows], state[rows - 1], d[rows], d[rows - 1], u[rows])
    return {
        name: _nrmse(
            state_next[rows, pod_rank + index:pod_rank + index + 1],
            prediction[:, pod_rank + index:pod_rank + index + 1],
            np.zeros(1),
        )
        for index, name in enumerate(q_names)
    }


def _direction_metrics(
    model: DynamicModel,
    state: np.ndarray,
    d: np.ndarray,
    q_normalizer: dict[str, Any],
    u_normalizer: dict[str, Any],
    pod_rank: int,
    q_names: list[str],
) -> dict[str, Any]:
    rows = np.arange(1, len(state), 20)
    low_u = _scalar_transform(np.full((len(rows), 1), 0.4), u_normalizer)
    high_u = _scalar_transform(np.full((len(rows), 1), 0.6), u_normalizer)
    args = (state[rows], state[rows - 1], d[rows], d[rows - 1])
    low = model.predict(*args, low_u)[:, pod_rank:]
    high = model.predict(*args, high_u)[:, pod_rank:]
    scale = np.asarray(q_normalizer["scale"])
    delta = (high - low) * scale
    result = {}
    pass_count = 0
    for name in ("running_imbalance", "waiting_imbalance", "remaining_token_imbalance"):
        index = q_names.index(name)
        fraction = float(np.mean(delta[:, index] > 0.0))
        result[name] = {
            "positive_fraction": fraction,
            "median_high_minus_low": float(np.median(delta[:, index])),
        }
        pass_count += fraction > 0.5
    result["direction_pass_fraction"] = pass_count / 3
    return result


def _fit_slow_for_candidate(
    state_train: np.ndarray,
    state_validation: np.ndarray,
    d_train: np.ndarray,
    d_validation: np.ndarray,
    u_train: np.ndarray,
    u_validation: np.ndarray,
    train_runs: list[Any],
    validation_runs: list[Any],
    slow_train_rows: list[dict[str, Any]],
    slow_validation_rows: list[dict[str, Any]],
    ridges: list[float],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    train_slow = _build_slow(slow_train_rows, train_runs, state_train, d_train, u_train)
    validation_slow = _build_slow(
        slow_validation_rows, validation_runs, state_validation, d_validation, u_validation,
    )
    y_normalizer = _fit_scalar_normalizer(train_slow[3], SLOW_OUTPUTS)
    y_train = _scalar_transform(train_slow[3], y_normalizer)
    y_validation = _scalar_transform(validation_slow[3], y_normalizer)
    winner = None
    scan = []
    for ridge in ridges:
        theta = _fit_slow_head(*train_slow[:3], y_train, ridge)
        value = _nrmse(
            y_validation,
            _predict_slow(theta, *validation_slow[:3]),
            np.zeros(len(SLOW_OUTPUTS)),
        )
        row = {"ridge": ridge, "validation_nrmse": value}
        scan.append(row)
        if winner is None or value < winner[0]["validation_nrmse"]:
            winner = (row, theta)
    assert winner is not None
    return winner[0], winner[1], {"normalizer": y_normalizer, "scan": scan}


def _scheme1_candidate(
    name: str,
    descriptors: list[Descriptor],
    pod_rank: int,
    arrays: dict[str, dict[str, np.ndarray]],
    z: dict[str, np.ndarray],
    z_next: dict[str, np.ndarray],
    d: dict[str, np.ndarray],
    u: dict[str, np.ndarray],
    runs: dict[str, list[Any]],
    slow_rows: dict[str, list[dict[str, Any]]],
    ridges: list[float],
    maximum_radius: float,
) -> dict[str, Any]:
    q_raw = {
        split: _extract_descriptors(arrays[split]["X"], descriptors)
        for split in ("train", "validation")
    }
    q_next_raw = {
        split: _extract_descriptors(arrays[split]["X_next"], descriptors)
        for split in ("train", "validation")
    }
    q_names = [descriptor.name for descriptor in descriptors]
    q_normalizer = _fit_scalar_normalizer(q_raw["train"], q_names)
    q = {split: _scalar_transform(q_raw[split], q_normalizer) for split in q_raw}
    q_next = {split: _scalar_transform(q_next_raw[split], q_normalizer) for split in q_next_raw}
    state = {split: np.concatenate((z[split][:, :pod_rank], q[split]), axis=1) for split in q}
    state_next = {
        split: np.concatenate((z_next[split][:, :pod_rank], q_next[split]), axis=1)
        for split in q
    }
    scan = []
    winner = None
    for ridge in ridges:
        model = _fit_dynamic(
            state["train"], state_next["train"], d["train"], u["train"],
            runs["train"], ridge, bilinear=False,
        )
        radius = model.spectral_radius()
        row = {
            "ridge": ridge,
            "spectral_radius": radius,
            "validation_one_step_nrmse": _one_step(
                model, state["validation"], state_next["validation"],
                d["validation"], u["validation"], runs["validation"],
            ),
            "stable": radius <= maximum_radius,
        }
        scan.append(row)
        if row["stable"] and (
            winner is None
            or row["validation_one_step_nrmse"] < winner[0]["validation_one_step_nrmse"]
        ):
            winner = (row, model)
    if winner is None:
        return {"candidate": name, "pod_rank": pod_rank, "stable": False, "ridge_scan": scan}
    row, model = winner
    rollout = _rollout_augmented(
        model, state["validation"], d["validation"], u["validation"],
        runs["validation"], pod_rank, q_names,
    )
    one_step_q = _one_step_descriptors(
        model, state["validation"], state_next["validation"], d["validation"],
        u["validation"], runs["validation"], pod_rank, q_names,
    )
    direction = _direction_metrics(
        model, state["validation"], d["validation"], q_normalizer,
        _fit_scalar_normalizer(arrays["train"]["U"], ["rho_A"]), pod_rank, q_names,
    )
    slow_row, slow_theta, slow_meta = _fit_slow_for_candidate(
        state["train"], state["validation"], d["train"], d["validation"],
        u["train"], u["validation"], runs["train"], runs["validation"],
        slow_rows["train"], slow_rows["validation"], ridges,
    )
    return {
        "candidate": name,
        "pod_rank": pod_rank,
        "descriptor_count": len(descriptors),
        "reduced_dimension": pod_rank + len(descriptors),
        "stable": True,
        "ridge": row["ridge"],
        "spectral_radius": row["spectral_radius"],
        "validation_one_step_nrmse": row["validation_one_step_nrmse"],
        "validation_descriptor_one_step_nrmse": one_step_q,
        "validation_rollout": rollout,
        "validation_control_direction": direction,
        "validation_slow_kpi_nrmse": slow_row["validation_nrmse"],
        "slow_ridge": slow_row["ridge"],
        "ridge_scan": scan,
        "model": model,
        "q_normalizer": q_normalizer,
        "slow_theta": slow_theta,
        "slow_meta": slow_meta,
        "state": state,
        "state_next": state_next,
    }


def _strip_runtime(value: dict[str, Any]) -> dict[str, Any]:
    excluded = {"model", "q_normalizer", "slow_theta", "slow_meta", "state", "state_next"}
    return {key: item for key, item in value.items() if key not in excluded}


def _fit_dense_pod(values: np.ndarray, max_rank: int) -> tuple[np.ndarray, np.ndarray]:
    covariance = values.T @ values / len(values)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    return vectors[:, order[:max_rank]], eigenvalues


def _scheme2_pairs(state_index: list[dict[str, Any]]) -> list[Descriptor]:
    position = {row["name"]: int(row["index"]) for row in state_index}
    pairs = [
        Descriptor(name, "pair", (position[left],), (position[right],), (left, right))
        for name, left, right in CORE_PAIRS
    ]
    for left in sorted(name for name in position if name.startswith("decode-0.")):
        right = left.replace("decode-0.", "decode-1.", 1)
        if right in position:
            pairs.append(Descriptor(
                left.removeprefix("decode-0.") + ".imbalance",
                "pair", (position[left],), (position[right],), (left, right),
            ))
    return pairs


def _scheme2_raw_blocks(array: np.ndarray, pairs: list[Descriptor], global_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(array[:, [row.left[0] for row in pairs]], dtype=np.float64)
    right = np.asarray(array[:, [row.right[0] for row in pairs]], dtype=np.float64)
    return np.asarray(array[:, global_indices], dtype=np.float64), (left + right) / 2.0, (left - right) / 2.0


def _rollout_scheme2(
    model: DynamicModel,
    state: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    runs: list[Any],
    gc_rank: int,
    diff_rank: int,
    decode_core: Callable[[np.ndarray], np.ndarray],
    actual_core: np.ndarray,
) -> dict[str, Any]:
    state_error = state_energy = base_error = 0.0
    gc_error = gc_energy = diff_error = diff_energy = 0.0
    core_error = np.zeros(3)
    core_energy = np.zeros(3)
    run_rows = []
    for run in runs:
        actual = state[run.start:run.end]
        prediction = np.empty_like(actual)
        prediction[0] = actual[0]
        previous = actual[0].copy()
        for offset in range(len(actual) - 1):
            index = run.start + offset
            d_previous = d[index - 1] if offset else d[index]
            current = prediction[offset]
            prediction[offset + 1] = model.predict(
                current[None], previous[None], d[index:index + 1],
                d_previous[None], u[index:index + 1],
            )[0]
            previous = current
        error = prediction - actual
        state_error += float(np.square(error).sum())
        state_energy += float(np.square(actual).sum())
        base_error += float(np.square(actual - actual[0]).sum())
        gc_error += float(np.square(error[:, :gc_rank]).sum())
        gc_energy += float(np.square(actual[:, :gc_rank]).sum())
        diff_error += float(np.square(error[:, gc_rank:gc_rank + diff_rank]).sum())
        diff_energy += float(np.square(actual[:, gc_rank:gc_rank + diff_rank]).sum())
        predicted_core = decode_core(prediction[:, gc_rank:gc_rank + diff_rank])
        observed_core = actual_core[run.start:run.end]
        core_error += np.square(predicted_core - observed_core).sum(axis=0)
        core_energy += np.square(observed_core).sum(axis=0)
        run_rows.append({
            "run_id": run.run_id,
            "nrmse": math.sqrt(float(np.square(error).sum()) / max(float(np.square(actual).sum()), 1e-30)),
        })
    nrmse = math.sqrt(state_error / max(state_energy, 1e-30))
    persistence = math.sqrt(base_error / max(state_energy, 1e-30))
    return {
        "state_nrmse": nrmse,
        "global_pod_nrmse": math.sqrt(gc_error / max(gc_energy, 1e-30)),
        "differential_pod_nrmse": math.sqrt(diff_error / max(diff_energy, 1e-30)),
        "persistence_nrmse": persistence,
        "skill": 1.0 - nrmse / max(persistence, 1e-30),
        "descriptor_nrmse": {
            name: math.sqrt(core_error[index] / max(core_energy[index], 1e-30))
            for index, name in enumerate(("running_imbalance", "waiting_imbalance", "remaining_token_imbalance"))
        },
        "runs": run_rows,
    }


def _scheme2_direction(
    model: DynamicModel,
    state: np.ndarray,
    d: np.ndarray,
    u_normalizer: dict[str, Any],
    gc_rank: int,
    diff_rank: int,
    decode_core: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    rows = np.arange(1, len(state), 20)
    args = (state[rows], state[rows - 1], d[rows], d[rows - 1])
    low = model.predict(*args, _scalar_transform(np.full((len(rows), 1), 0.4), u_normalizer))
    high = model.predict(*args, _scalar_transform(np.full((len(rows), 1), 0.6), u_normalizer))
    delta = decode_core(high[:, gc_rank:gc_rank + diff_rank]) - decode_core(low[:, gc_rank:gc_rank + diff_rank])
    result = {}
    passes = 0
    for index, name in enumerate(("running_imbalance", "waiting_imbalance", "remaining_token_imbalance")):
        fraction = float(np.mean(delta[:, index] > 0.0))
        result[name] = {"positive_fraction": fraction, "median_high_minus_low": float(np.median(delta[:, index]))}
        passes += fraction > 0.5
    result["direction_pass_fraction"] = passes / 3
    return result


def _run_scheme2(
    dataset_root: Path,
    output_root: Path,
    config: dict[str, Any],
    dataset_audit: dict[str, Any],
    baseline: dict[str, Any],
    state_index: list[dict[str, Any]],
    arrays: dict[str, dict[str, np.ndarray]],
    x_normalizer: Any,
    d_normalizer: Any,
    u_normalizer: dict[str, Any],
    runs: dict[str, list[Any]],
    slow_rows: dict[str, list[dict[str, Any]]],
    scheme1: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = _scheme2_pairs(state_index)
    paired_indices = {index for row in pairs for index in (row.left[0], row.right[0])}
    global_indices = np.asarray([index for index in range(len(state_index)) if index not in paired_indices], dtype=np.int64)
    pair_manifest = {
        "schema_version": "servingrom.common-differential-pairs.v1",
        "pair_count": len(pairs),
        "global_feature_count": len(global_indices),
        "pairs": [{"name": row.name, "left": row.source_names[0], "right": row.source_names[1]} for row in pairs],
        "global_features": [state_index[index]["name"] for index in global_indices],
    }
    save_json(output_root / "scheme2/feature_manifest.json", pair_manifest)
    raw_blocks = {
        split: _scheme2_raw_blocks(arrays[split]["X"], pairs, global_indices)
        for split in ("train", "validation")
    }
    raw_next_blocks = {
        split: _scheme2_raw_blocks(arrays[split]["X_next"], pairs, global_indices)
        for split in ("train", "validation")
    }
    common_norm = _fit_scalar_normalizer(raw_blocks["train"][1], [row.name + ".common" for row in pairs])
    diff_norm = _fit_scalar_normalizer(raw_blocks["train"][2], [row.name for row in pairs])
    core_norm = _fit_scalar_normalizer(
        np.column_stack([
            np.asarray(arrays["train"]["X"][:, row.left[0]]) - np.asarray(arrays["train"]["X"][:, row.right[0]])
            for row in pairs[:3]
        ]),
        [row.name for row in pairs[:3]],
    )
    save_json(output_root / "scheme2/global_normalization.json", {
        "schema_version": "servingrom.scheme2-global-normalization.v1",
        "source": "Step15 train-only full-state normalizer",
        "indices": global_indices.tolist(),
        "names": pair_manifest["global_features"],
        "mean": x_normalizer.mean[global_indices].tolist(),
        "scale": x_normalizer.scale[global_indices].tolist(),
        "block_weight": x_normalizer.block_weight[global_indices].tolist(),
        "log_mask": x_normalizer.log_mask[global_indices].tolist(),
    })
    save_json(output_root / "scheme2/common_normalization.json", common_norm)
    save_json(output_root / "scheme2/differential_normalization.json", diff_norm)
    save_json(output_root / "scheme2/core_imbalance_normalization.json", core_norm)

    gc = {}
    gc_next = {}
    diff = {}
    diff_next = {}
    core = {}
    for split in ("train", "validation"):
        global_value = x_normalizer.transform(arrays[split]["X"])[:, global_indices]
        common_value = _scalar_transform(raw_blocks[split][1], common_norm)
        gc[split] = np.concatenate((global_value, common_value), axis=1)
        gc_next[split] = np.concatenate((
            x_normalizer.transform(arrays[split]["X_next"])[:, global_indices],
            _scalar_transform(raw_next_blocks[split][1], common_norm),
        ), axis=1)
        diff[split] = _scalar_transform(raw_blocks[split][2], diff_norm)
        diff_next[split] = _scalar_transform(raw_next_blocks[split][2], diff_norm)
        core[split] = _scalar_transform(raw_blocks[split][2][:, :3] * 2.0, core_norm)
    max_gc = max(int(value) for value in config["scheme2_global_ranks"])
    max_diff = max(int(value) for value in config["scheme2_diff_ranks"])
    gc_basis, gc_eigenvalues = _fit_dense_pod(gc["train"], max_gc)
    diff_basis, diff_eigenvalues = _fit_dense_pod(diff["train"], max_diff)
    np.save(output_root / "scheme2/global_common_basis_candidates.npy", gc_basis)
    np.save(output_root / "scheme2/differential_basis_candidates.npy", diff_basis)
    save_json(output_root / "scheme2/spectrum.json", {
        "global_common_eigenvalues": gc_eigenvalues.tolist(),
        "differential_eigenvalues": diff_eigenvalues.tolist(),
    })
    zgc = {split: gc[split] @ gc_basis for split in gc}
    zgc_next = {split: gc_next[split] @ gc_basis for split in gc_next}
    zdiff = {split: diff[split] @ diff_basis for split in diff}
    zdiff_next = {split: diff_next[split] @ diff_basis for split in diff_next}
    d = {split: d_normalizer.transform(arrays[split]["D"], weighted=False) for split in arrays}
    u = {split: _scalar_transform(arrays[split]["U"], u_normalizer) for split in arrays}

    candidates = []
    runtime = []
    global_limit = baseline["validation_rollout_nrmse"] * (1.0 + float(config["maximum_global_rollout_degradation"]))
    slow_limit = baseline["validation_slow_kpi_nrmse"] * (1.0 + float(config["maximum_slow_kpi_degradation"]))
    core_limit = float(config["maximum_core_diff_rollout_nrmse"])
    direction_limit = float(config["minimum_control_direction_fraction"])
    for gc_rank in [int(value) for value in config["scheme2_global_ranks"]]:
        for diff_rank in [int(value) for value in config["scheme2_diff_ranks"]]:
            state = {split: np.concatenate((zgc[split][:, :gc_rank], zdiff[split][:, :diff_rank]), axis=1) for split in zgc}
            state_next = {split: np.concatenate((zgc_next[split][:, :gc_rank], zdiff_next[split][:, :diff_rank]), axis=1) for split in zgc_next}
            scale = np.asarray(diff_norm["scale"])
            mean = np.asarray(diff_norm["mean"])
            core_mean = np.asarray(core_norm["mean"])
            core_scale = np.asarray(core_norm["scale"])

            def decode_core(value: np.ndarray, rank: int = diff_rank) -> np.ndarray:
                normalized_diff = value @ diff_basis[:, :rank].T
                raw_diff = normalized_diff * scale + mean
                return (raw_diff[:, :3] * 2.0 - core_mean) / core_scale

            representation_core = decode_core(zdiff["validation"][:, :diff_rank])
            representation_nrmse = {
                pairs[index].name: _nrmse(core["validation"][:, index:index + 1], representation_core[:, index:index + 1], np.zeros(1))
                for index in range(3)
            }
            ridge_scan = []
            winner = None
            for ridge in [float(value) for value in config["candidate_ridges"]]:
                model = _fit_dynamic(
                    state["train"], state_next["train"], d["train"], u["train"],
                    runs["train"], ridge, bilinear=False,
                )
                radius = model.spectral_radius()
                row = {
                    "ridge": ridge, "spectral_radius": radius,
                    "validation_one_step_nrmse": _one_step(
                        model, state["validation"], state_next["validation"],
                        d["validation"], u["validation"], runs["validation"],
                    ),
                    "stable": radius <= float(config["maximum_spectral_radius"]),
                }
                ridge_scan.append(row)
                if row["stable"] and (winner is None or row["validation_one_step_nrmse"] < winner[0]["validation_one_step_nrmse"]):
                    winner = (row, model)
            if winner is None:
                continue
            winner_row, model = winner
            rollout = _rollout_scheme2(
                model, state["validation"], d["validation"], u["validation"],
                runs["validation"], gc_rank, diff_rank, decode_core, core["validation"],
            )
            direction = _scheme2_direction(
                model, state["validation"], d["validation"], u_normalizer,
                gc_rank, diff_rank, decode_core,
            )
            slow_row, slow_theta, slow_meta = _fit_slow_for_candidate(
                state["train"], state["validation"], d["train"], d["validation"],
                u["train"], u["validation"], runs["train"], runs["validation"],
                slow_rows["train"], slow_rows["validation"],
                [float(value) for value in config["candidate_ridges"]],
            )
            gate = {
                "running": rollout["descriptor_nrmse"]["running_imbalance"] < core_limit,
                "waiting": rollout["descriptor_nrmse"]["waiting_imbalance"] < core_limit,
                "remaining": rollout["descriptor_nrmse"]["remaining_token_imbalance"] < core_limit,
                "representation": all(value < core_limit for value in representation_nrmse.values()),
                "global_rollout": rollout["global_pod_nrmse"] <= global_limit,
                "slow_kpi": slow_row["validation_nrmse"] <= slow_limit,
                "direction": direction["direction_pass_fraction"] >= direction_limit,
            }
            row = {
                "scheme": "scheme2_common_differential_block_pod",
                "candidate": f"gc{gc_rank}-diff{diff_rank}",
                "global_common_rank": gc_rank,
                "differential_rank": diff_rank,
                "pod_rank": gc_rank,
                "descriptor_count": diff_rank,
                "reduced_dimension": gc_rank + diff_rank,
                "ridge": winner_row["ridge"],
                "spectral_radius": winner_row["spectral_radius"],
                "validation_one_step_nrmse": winner_row["validation_one_step_nrmse"],
                "validation_representation_core_nrmse": representation_nrmse,
                "validation_rollout": rollout,
                "validation_control_direction": direction,
                "validation_slow_kpi_nrmse": slow_row["validation_nrmse"],
                "slow_ridge": slow_row["ridge"],
                "validation_gate": gate,
                "ridge_scan": ridge_scan,
            }
            candidates.append(row)
            runtime.append((row, model, slow_theta, slow_meta, state, state_next, decode_core))
    save_json(output_root / "scheme2/rank_grid.json", candidates)
    passing = [row for row in runtime if all(row[0]["validation_gate"].values())]
    if not passing:
        result = {
            "schema_version": "servingrom.control-redesign.result.v1",
            "dataset": dataset_audit,
            "baseline": baseline,
            "scheme1_ablation": [_strip_runtime(row) for row in scheme1],
            "scheme2_ablation": candidates,
            "readiness": {"control_representation_ready": False, "control_rom_ready": False},
            "data_isolation": {"heldout_actuator_data_read": False, "test_accessed": False, "mpc_implemented": False},
        }
        save_json(output_root / "evaluation/final_metrics.json", result)
        return result
    selected_row, model, slow_theta, slow_meta, selected_state, selected_state_next, decode_core = min(
        passing,
        key=lambda value: (
            value[0]["reduced_dimension"],
            np.mean(list(value[0]["validation_rollout"]["descriptor_nrmse"].values())),
            value[0]["validation_rollout"]["global_pod_nrmse"],
        ),
    )
    frozen = {
        "scheme": selected_row["scheme"], "candidate": selected_row["candidate"],
        "pod_rank": selected_row["global_common_rank"], "descriptor_count": selected_row["differential_rank"],
        "reduced_dimension": selected_row["reduced_dimension"], "dynamics_ridge": selected_row["ridge"],
        "slow_ridge": selected_row["slow_ridge"], "selection_split": "validation", "test_accessed": False,
    }
    save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen)

    test_arrays = _load_split(dataset_root, "test")
    test_blocks = _scheme2_raw_blocks(test_arrays["X"], pairs, global_indices)
    test_next_blocks = _scheme2_raw_blocks(test_arrays["X_next"], pairs, global_indices)
    gc_test = np.concatenate((x_normalizer.transform(test_arrays["X"])[:, global_indices], _scalar_transform(test_blocks[1], common_norm)), axis=1)
    gc_test_next = np.concatenate((x_normalizer.transform(test_arrays["X_next"])[:, global_indices], _scalar_transform(test_next_blocks[1], common_norm)), axis=1)
    diff_test = _scalar_transform(test_blocks[2], diff_norm)
    diff_test_next = _scalar_transform(test_next_blocks[2], diff_norm)
    gc_rank = selected_row["global_common_rank"]
    diff_rank = selected_row["differential_rank"]
    state_test = np.concatenate((gc_test @ gc_basis[:, :gc_rank], diff_test @ diff_basis[:, :diff_rank]), axis=1)
    state_test_next = np.concatenate((gc_test_next @ gc_basis[:, :gc_rank], diff_test_next @ diff_basis[:, :diff_rank]), axis=1)
    d_test = d_normalizer.transform(test_arrays["D"], weighted=False)
    u_test = _scalar_transform(test_arrays["U"], u_normalizer)
    core_test = _scalar_transform(test_blocks[2][:, :3] * 2.0, core_norm)
    test_rollout = _rollout_scheme2(model, state_test, d_test, u_test, runs["test"], gc_rank, diff_rank, decode_core, core_test)
    test_direction = _scheme2_direction(model, state_test, d_test, u_normalizer, gc_rank, diff_rank, decode_core)
    test_slow_rows = pq.read_table(dataset_root / "slow_kpi_windows.parquet", filters=[("split", "=", "test")]).to_pylist()
    test_slow = _build_slow(test_slow_rows, runs["test"], state_test, d_test, u_test)
    test_y = _scalar_transform(test_slow[3], slow_meta["normalizer"])
    test_slow_nrmse = _nrmse(test_y, _predict_slow(slow_theta, *test_slow[:3]), np.zeros(len(SLOW_OUTPUTS)))
    final = {
        **frozen, "spectral_radius": selected_row["spectral_radius"], "validation": selected_row,
        "test": {"rollout": test_rollout, "control_direction": test_direction, "slow_kpi_nrmse": test_slow_nrmse},
    }
    test_core = test_rollout["descriptor_nrmse"]
    test_pass = all(value < core_limit for value in test_core.values()) and test_direction["direction_pass_fraction"] >= direction_limit
    readiness = {
        "control_representation_ready": test_pass,
        "control_rom_ready": test_pass and test_slow_nrmse <= slow_limit and selected_row["spectral_radius"] <= float(config["maximum_spectral_radius"]),
    }
    result = {
        "schema_version": "servingrom.control-redesign.result.v1", "dataset": dataset_audit, "baseline": baseline,
        "scheme1_ablation": [_strip_runtime(row) for row in scheme1], "scheme2_ablation": candidates,
        "scheme2_executed": True, "final": final, "readiness": readiness,
        "data_isolation": {"heldout_actuator_data_read": False, "test_accessed_after_freeze": True, "mpc_implemented": False},
    }
    np.save(output_root / "scheme2/final_global_common_basis.npy", gc_basis[:, :gc_rank])
    np.save(output_root / "scheme2/final_differential_basis.npy", diff_basis[:, :diff_rank])
    _save_model(output_root / "models/final_control_dynamics.npz", model)
    np.savez_compressed(output_root / "models/final_slow_kpi_head.npz", theta=slow_theta, outputs=np.asarray(SLOW_OUTPUTS), ridge=selected_row["slow_ridge"])
    save_json(output_root / "evaluation/final_metrics.json", result)
    _write_report(output_root, result)
    manifest = {
        "model_id": config["model_id"], "dataset_sha256_manifest": dataset_audit["sha256_manifest"],
        "baseline_manifest_sha256": baseline["manifest_sha256"], "selection": frozen,
        "readiness": readiness, "artifacts": {},
    }
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256_MANIFEST.json", "step15b.log", "step15b.pid"}:
            manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    save_json(output_root / "SHA256_MANIFEST.json", manifest)
    return result


def _write_report(output: Path, result: dict[str, Any]) -> None:
    final = result["final"]
    lines = [
        "# ServingROM Step 15B Control-Relevant Reduced-State Redesign",
        "",
        "## 结论",
        "",
        f"- `control_representation_ready={str(result['readiness']['control_representation_ready']).lower()}`",
        f"- `control_rom_ready={str(result['readiness']['control_rom_ready']).lower()}`",
        f"- 最终方案：`{final['scheme']}` / `{final['candidate']}`",
        f"- reduced state：POD rank {final['pod_rank']} + {final['descriptor_count']} explicit differential coordinates = {final['reduced_dimension']}",
        "- Round 14.3 held-out actuator 数据未读取；未实现 MPC。",
        "",
        "## 三方对照",
        "",
        "| representation | dim | val global rollout | val running diff rollout | val remaining diff rollout | val slow KPI | radius |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    baseline = result["baseline"]
    lines.append(
        f"| Step15 standard POD | 16 | {baseline['validation_rollout_nrmse']:.6f} | n/a | n/a | "
        f"{baseline['validation_slow_kpi_nrmse']:.6f} | {baseline['spectral_radius']:.6f} |"
    )
    for row in result["scheme1_ablation"]:
        rollout = row.get("validation_rollout", {})
        descriptor = rollout.get("descriptor_nrmse", {})
        lines.append(
            f"| Scheme1 {row['candidate']}/r{row['pod_rank']} | {row.get('reduced_dimension', 0)} | "
            f"{rollout.get('global_pod_nrmse', math.nan):.6f} | {descriptor.get('running_imbalance', math.nan):.6f} | "
            f"{descriptor.get('remaining_token_imbalance', math.nan):.6f} | "
            f"{row.get('validation_slow_kpi_nrmse', math.nan):.6f} | {row.get('spectral_radius', math.nan):.6f} |"
        )
    lines += [
        "",
        "## Test（冻结后单次访问）",
        "",
        f"- global POD rollout NRMSE：`{final['test']['rollout']['global_pod_nrmse']:.6f}`",
        f"- running imbalance rollout NRMSE：`{final['test']['rollout']['descriptor_nrmse']['running_imbalance']:.6f}`",
        f"- waiting imbalance rollout NRMSE：`{final['test']['rollout']['descriptor_nrmse']['waiting_imbalance']:.6f}`",
        f"- remaining-token imbalance rollout NRMSE：`{final['test']['rollout']['descriptor_nrmse']['remaining_token_imbalance']:.6f}`",
        f"- Slow KPI NRMSE：`{final['test']['slow_kpi_nrmse']:.6f}`",
        "",
        "## 选择逻辑",
        "",
        "优先选择同时满足核心 differential rollout <0.7、谱半径 <=1.01、global rollout 与 Slow KPI 相对 Step 15 退化不超过10%的最低维 Scheme 1。只有 Scheme 1 全部失败才允许运行 Scheme 2。",
    ]
    (output / "STEP15B_CONTROL_RELEVANT_REDESIGN_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def run_redesign_pipeline(
    dataset_root: Path,
    baseline_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    baseline_root = baseline_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = _verify_dataset(dataset_root, config["dataset_id"])
    baseline_metrics = _load_json(baseline_root / "evaluation/final_metrics.json")
    baseline = {
        "validation_rollout_nrmse": baseline_metrics["metrics"]["validation"]["rollout"]["state_nrmse"],
        "validation_slow_kpi_nrmse": baseline_metrics["slow_head"]["validation_nrmse"],
        "spectral_radius": baseline_metrics["selection"]["spectral_radius"],
        "manifest_sha256": _sha256(baseline_root / "MODEL_MANIFEST.json"),
    }
    runs = _load_runs(dataset_root)
    state_index = _load_json(dataset_root / "state_index.json")
    disturbance_index = _load_json(dataset_root / "disturbance_index.json")
    descriptor_data = _descriptor_manifest(state_index)
    descriptors = descriptor_data.pop("objects")
    save_json(output_root / "scheme1/feature_manifest.json", descriptor_data)

    arrays = {split: _load_split(dataset_root, split) for split in ("train", "validation")}
    x_normalizer, _ = fit_normalizer(arrays["train"]["X"], state_index, int(config["chunk_size"]))
    d_normalizer, _ = fit_normalizer(arrays["train"]["D"], disturbance_index, int(config["chunk_size"]))
    u_normalizer = _fit_scalar_normalizer(arrays["train"]["U"], ["rho_A"])
    save_json(output_root / "normalization/x_train_only.json", x_normalizer.to_json())
    save_json(output_root / "normalization/d_train_only.json", d_normalizer.to_json())
    save_json(output_root / "normalization/u_train_only.json", u_normalizer)

    ranks = [int(value) for value in config["scheme1_pod_ranks"]]
    pod = fit_pod(arrays["train"]["X"], x_normalizer, max(ranks), int(config["chunk_size"]))
    np.save(output_root / "scheme1/global_pod_basis_candidates.npy", pod["basis"])
    z = {split: _project(arrays[split]["X"], x_normalizer, pod["basis"], int(config["chunk_size"])) for split in arrays}
    z_next = {split: _project(arrays[split]["X_next"], x_normalizer, pod["basis"], int(config["chunk_size"])) for split in arrays}
    d = {split: d_normalizer.transform(arrays[split]["D"], weighted=False) for split in arrays}
    u = {split: _scalar_transform(arrays[split]["U"], u_normalizer) for split in arrays}
    slow_table = pq.read_table(
        dataset_root / "slow_kpi_windows.parquet",
        filters=[("split", "in", ["train", "validation"])],
    ).to_pylist()
    slow_rows = {
        split: [row for row in slow_table if row["split"] == split]
        for split in ("train", "validation")
    }

    scheme1 = []
    for candidate_name in descriptor_data["candidate_order"]:
        for rank in ranks:
            scheme1.append(_scheme1_candidate(
                candidate_name, descriptors[candidate_name], rank, arrays, z, z_next, d, u,
                runs, slow_rows, [float(value) for value in config["candidate_ridges"]],
                float(config["maximum_spectral_radius"]),
            ))
    clean_scheme1 = [_strip_runtime(row) for row in scheme1]
    save_json(output_root / "scheme1/ablation.json", clean_scheme1)

    global_limit = baseline["validation_rollout_nrmse"] * (1.0 + float(config["maximum_global_rollout_degradation"]))
    slow_limit = baseline["validation_slow_kpi_nrmse"] * (1.0 + float(config["maximum_slow_kpi_degradation"]))
    core_limit = float(config["maximum_core_diff_rollout_nrmse"])
    direction_limit = float(config["minimum_control_direction_fraction"])
    passing = []
    for row in scheme1:
        if not row.get("stable"):
            continue
        core = row["validation_rollout"]["descriptor_nrmse"]
        row["validation_gate"] = {
            "running": core["running_imbalance"] < core_limit,
            "remaining": core["remaining_token_imbalance"] < core_limit,
            "waiting": core["waiting_imbalance"] < core_limit,
            "global_rollout": row["validation_rollout"]["global_pod_nrmse"] <= global_limit,
            "slow_kpi": row["validation_slow_kpi_nrmse"] <= slow_limit,
            "direction": row["validation_control_direction"]["direction_pass_fraction"] >= direction_limit,
        }
        if all(row["validation_gate"].values()):
            passing.append(row)
    if not passing:
        scheme1_result = {
            "schema_version": "servingrom.control-redesign.scheme1-gate.v1",
            "dataset": dataset_audit, "baseline": baseline,
            "scheme1_ablation": [_strip_runtime(row) for row in scheme1],
            "scheme2_required": True,
            "data_isolation": {"heldout_actuator_data_read": False, "test_accessed": False, "mpc_implemented": False},
        }
        save_json(output_root / "SCHEME1_GATE_RESULT.json", scheme1_result)
        return _run_scheme2(
            dataset_root, output_root, config, dataset_audit, baseline, state_index,
            arrays, x_normalizer, d_normalizer, u_normalizer, runs, slow_rows, scheme1,
        )

    selected = min(
        passing,
        key=lambda row: (
            row["reduced_dimension"],
            np.mean([
                row["validation_rollout"]["descriptor_nrmse"][name]
                for name in ("running_imbalance", "waiting_imbalance", "remaining_token_imbalance")
            ]),
            row["validation_rollout"]["global_pod_nrmse"],
        ),
    )
    frozen = {
        "scheme": "scheme1_explicit_differential_augmentation",
        "candidate": selected["candidate"],
        "pod_rank": selected["pod_rank"],
        "descriptor_count": selected["descriptor_count"],
        "reduced_dimension": selected["reduced_dimension"],
        "dynamics_ridge": selected["ridge"],
        "slow_ridge": selected["slow_ridge"],
        "selection_split": "validation",
        "test_accessed": False,
    }
    save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen)

    # Test is opened only after the Scheme 1 representation and all hyperparameters are frozen.
    test_arrays = _load_split(dataset_root, "test")
    selected_descriptors = descriptors[selected["candidate"]]
    q_names = [item.name for item in selected_descriptors]
    q_test = _scalar_transform(
        _extract_descriptors(test_arrays["X"], selected_descriptors), selected["q_normalizer"],
    )
    q_test_next = _scalar_transform(
        _extract_descriptors(test_arrays["X_next"], selected_descriptors), selected["q_normalizer"],
    )
    z_test = _project(test_arrays["X"], x_normalizer, pod["basis"][:, :selected["pod_rank"]], int(config["chunk_size"]))
    z_test_next = _project(test_arrays["X_next"], x_normalizer, pod["basis"][:, :selected["pod_rank"]], int(config["chunk_size"]))
    state_test = np.concatenate((z_test, q_test), axis=1)
    state_test_next = np.concatenate((z_test_next, q_test_next), axis=1)
    d_test = d_normalizer.transform(test_arrays["D"], weighted=False)
    u_test = _scalar_transform(test_arrays["U"], u_normalizer)
    test_rollout = _rollout_augmented(
        selected["model"], state_test, d_test, u_test, runs["test"],
        selected["pod_rank"], q_names,
    )
    test_one_step = _one_step_descriptors(
        selected["model"], state_test, state_test_next, d_test, u_test, runs["test"],
        selected["pod_rank"], q_names,
    )
    test_direction = _direction_metrics(
        selected["model"], state_test, d_test, selected["q_normalizer"],
        u_normalizer, selected["pod_rank"], q_names,
    )
    test_slow_rows = pq.read_table(
        dataset_root / "slow_kpi_windows.parquet", filters=[("split", "=", "test")],
    ).to_pylist()
    test_slow = _build_slow(test_slow_rows, runs["test"], state_test, d_test, u_test)
    y_normalizer = selected["slow_meta"]["normalizer"]
    test_y = _scalar_transform(test_slow[3], y_normalizer)
    test_slow_nrmse = _nrmse(
        test_y,
        _predict_slow(selected["slow_theta"], *test_slow[:3]),
        np.zeros(len(SLOW_OUTPUTS)),
    )

    final = {
        **frozen,
        "spectral_radius": selected["spectral_radius"],
        "validation": _strip_runtime(selected),
        "test": {
            "rollout": test_rollout,
            "descriptor_one_step_nrmse": test_one_step,
            "control_direction": test_direction,
            "slow_kpi_nrmse": test_slow_nrmse,
        },
    }
    test_core = test_rollout["descriptor_nrmse"]
    test_pass = (
        test_core["running_imbalance"] < core_limit
        and test_core["remaining_token_imbalance"] < core_limit
        and test_core["waiting_imbalance"] < core_limit
        and test_direction["direction_pass_fraction"] >= direction_limit
        and selected["spectral_radius"] <= float(config["maximum_spectral_radius"])
    )
    readiness = {
        "control_representation_ready": test_pass,
        "control_rom_ready": test_pass and test_slow_nrmse <= slow_limit,
    }
    result = {
        "schema_version": "servingrom.control-redesign.result.v1",
        "dataset": dataset_audit,
        "baseline": baseline,
        "scheme1_ablation": [_strip_runtime(row) for row in scheme1],
        "scheme2_executed": False,
        "scheme2_reason": "Scheme 1 passed validation gates; complexity stopped by design.",
        "final": final,
        "readiness": readiness,
        "data_isolation": {"heldout_actuator_data_read": False, "test_accessed_after_freeze": True, "mpc_implemented": False},
    }
    np.save(output_root / "scheme1/final_global_pod_basis.npy", pod["basis"][:, :selected["pod_rank"]])
    save_json(output_root / "scheme1/final_descriptor_normalization.json", selected["q_normalizer"])
    _save_model(output_root / "models/final_control_dynamics.npz", selected["model"])
    np.savez_compressed(
        output_root / "models/final_slow_kpi_head.npz",
        theta=selected["slow_theta"], outputs=np.asarray(SLOW_OUTPUTS), ridge=selected["slow_ridge"],
    )
    save_json(output_root / "evaluation/final_metrics.json", result)
    _write_report(output_root, result)
    manifest = {
        "model_id": config["model_id"],
        "dataset_sha256_manifest": dataset_audit["sha256_manifest"],
        "baseline_manifest_sha256": baseline["manifest_sha256"],
        "selection": frozen,
        "readiness": readiness,
        "artifacts": {},
    }
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256_MANIFEST.json", "step15b.log", "step15b.pid"}:
            manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    save_json(output_root / "SHA256_MANIFEST.json", manifest)
    return result
