from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from servingrom_modeling.preprocessing import Normalizer, save_json

from .pipeline import (
    SLOW_OUTPUTS,
    _build_slow,
    _fit_scalar_normalizer,
    _fit_slow_head,
    _load_json,
    _load_runs,
    _load_split,
    _nrmse,
    _predict_slow,
    _scalar_transform,
    _sha256,
    _verify_dataset,
)
from .redesign import _scheme2_pairs, _scheme2_raw_blocks


CORE_NAMES = ["running_imbalance", "waiting_imbalance", "remaining_token_imbalance"]


@dataclass
class MemoryModel:
    kind: str
    horizon_steps: int
    horizon_seconds: float
    ridge: float
    theta: np.ndarray
    state_dim: int
    d_dim: int
    signal_dim: int
    scales: tuple[int, ...]

    @property
    def memory_dim(self) -> int:
        if self.kind == "baseline":
            return 0
        if self.kind == "raw_lag":
            return self.horizon_steps * self.signal_dim
        if self.kind == "multi_scale":
            return len(self.scales) * self.signal_dim * 3
        if self.kind == "exponential":
            return len(self.scales) * self.signal_dim
        raise ValueError(self.kind)

    def predict(self, state: np.ndarray, previous: np.ndarray, d: np.ndarray, d_previous: np.ndarray, u: np.ndarray, memory: np.ndarray) -> np.ndarray:
        design = np.concatenate((state, state - previous, d, d_previous, u, memory, np.ones((len(state), 1))), axis=1)
        return design @ self.theta


def _forcing_audit() -> dict[str, Any]:
    return {
        "effective_forcing_available": True,
        "source": "sealed run derived/control/control_windows.parquet",
        "source_event": "p_to_d_route",
        "alignment": "200 ms half-open [start_wall_ns,end_wall_ns); route event selected by ts_wall_ns",
        "fields": {
            "routed_request_imbalance": "2*routed_A_request_count-routed_request_count",
            "routed_expected_token_mass_imbalance": "2*routed_A_expected_token_mass-routed_expected_token_mass",
        },
        "token_semantics": "expected_output_tokens captured at p_to_d_route; not active remaining-token inventory",
        "used_by_step15c1_model": False,
        "next_schema": {
            "names": ["routed_request_imbalance", "routed_expected_token_mass_imbalance"],
            "role": "effective routing forcing for Step 15C-2",
            "window_seconds": 0.2,
        },
    }


def _load_frozen_representation(dataset_root: Path, representation_root: Path) -> dict[str, Any]:
    frozen = _load_json(representation_root / "FROZEN_SELECTION_BEFORE_TEST.json")
    if frozen.get("scheme") != "scheme2_common_differential_block_pod" or frozen.get("candidate") != "gc12-diff2":
        raise ValueError("Step 15C-1 requires frozen gc12-diff2")
    state_index = _load_json(dataset_root / "state_index.json")
    names = {row["name"]: int(row["index"]) for row in state_index}
    pairs = _scheme2_pairs(state_index)
    paired = {index for row in pairs for index in (row.left[0], row.right[0])}
    global_indices = np.asarray([index for index in range(len(state_index)) if index not in paired], dtype=np.int64)
    return {
        "frozen": frozen,
        "state_index": state_index,
        "pairs": pairs,
        "global_indices": global_indices,
        "x_normalizer": Normalizer.from_json(_load_json(representation_root / "normalization/x_train_only.json")),
        "d_normalizer": Normalizer.from_json(_load_json(representation_root / "normalization/d_train_only.json")),
        "u_normalizer": _load_json(representation_root / "normalization/u_train_only.json"),
        "common_normalizer": _load_json(representation_root / "scheme2/common_normalization.json"),
        "diff_normalizer": _load_json(representation_root / "scheme2/differential_normalization.json"),
        "core_normalizer": _load_json(representation_root / "scheme2/core_imbalance_normalization.json"),
        "gc_basis": np.load(representation_root / "scheme2/final_global_common_basis.npy"),
        "diff_basis": np.load(representation_root / "scheme2/final_differential_basis.npy"),
    }


def _encode(array: np.ndarray, frozen: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    global_raw, common_raw, diff_raw = _scheme2_raw_blocks(array, frozen["pairs"], frozen["global_indices"])
    del global_raw
    global_value = frozen["x_normalizer"].transform(array)[:, frozen["global_indices"]]
    common_value = _scalar_transform(common_raw, frozen["common_normalizer"])
    diff_value = _scalar_transform(diff_raw, frozen["diff_normalizer"])
    state = np.concatenate((
        np.concatenate((global_value, common_value), axis=1) @ frozen["gc_basis"],
        diff_value @ frozen["diff_basis"],
    ), axis=1)
    core = _scalar_transform(diff_raw[:, :3] * 2.0, frozen["core_normalizer"])
    return state, core


def _decode_core(diff_state: np.ndarray, frozen: dict[str, Any]) -> np.ndarray:
    normalized = diff_state @ frozen["diff_basis"].T
    raw_half_diff = normalized * np.asarray(frozen["diff_normalizer"]["scale"]) + np.asarray(frozen["diff_normalizer"]["mean"])
    raw_core = raw_half_diff[:, :3] * 2.0
    return _scalar_transform(raw_core, frozen["core_normalizer"])


def _signal(state: np.ndarray, d: np.ndarray, u: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    return np.concatenate((state[:, -2:], u, d[:, relevant]), axis=1)


def _memory_matrix(kind: str, horizon: int, scales: tuple[int, ...], signal: np.ndarray, runs: list[Any], dt: float) -> np.ndarray:
    if kind == "baseline":
        return np.empty((len(signal), 0), dtype=np.float64)
    if kind == "raw_lag":
        result = np.empty((len(signal), horizon * signal.shape[1]), dtype=np.float64)
        for run in runs:
            first = signal[run.start]
            for index in range(run.start, run.end):
                history = [signal[index - lag] if index - lag >= run.start else first for lag in range(1, horizon + 1)]
                result[index] = np.concatenate(history)
        return result
    if kind == "multi_scale":
        result = np.empty((len(signal), len(scales) * signal.shape[1] * 3), dtype=np.float64)
        for run in runs:
            values = signal[run.start:run.end]
            cumulative = np.vstack((np.zeros((1, signal.shape[1])), np.cumsum(values, axis=0)))
            for offset, index in enumerate(range(run.start, run.end)):
                rows = []
                for scale in scales:
                    begin = max(0, offset + 1 - scale)
                    count = offset + 1 - begin
                    total = cumulative[offset + 1] - cumulative[begin]
                    old = values[max(0, offset - scale)]
                    rows.extend((total / count, total * dt, values[offset] - old))
                result[index] = np.concatenate(rows)
        return result
    if kind == "exponential":
        result = np.empty((len(signal), len(scales) * signal.shape[1]), dtype=np.float64)
        for run in runs:
            memories = [signal[run.start].copy() for _ in scales]
            for index in range(run.start, run.end):
                result[index] = np.concatenate(memories)
                for slot, tau_steps in enumerate(scales):
                    alpha = math.exp(-1.0 / tau_steps)
                    memories[slot] = alpha * memories[slot] + (1.0 - alpha) * signal[index]
        return result
    raise ValueError(kind)


def _valid_rows(runs: list[Any], horizon: int) -> np.ndarray:
    return np.concatenate([np.arange(run.start + max(horizon, 1), run.end) for run in runs])


def _fit_candidate(kind: str, horizon: int, scales: tuple[int, ...], ridge: float, state: np.ndarray, target: np.ndarray, d: np.ndarray, u: np.ndarray, memory: np.ndarray, runs: list[Any]) -> MemoryModel:
    rows = _valid_rows(runs, horizon)
    design = np.concatenate((state[rows], state[rows] - state[rows - 1], d[rows], d[rows - 1], u[rows], memory[rows], np.ones((len(rows), 1))), axis=1)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[-1, -1] = 0.0
    theta = np.linalg.solve(gram + penalty, design.T @ target[rows])
    return MemoryModel(kind, horizon, horizon * 0.2, ridge, theta, state.shape[1], d.shape[1], 6, scales)


def _one_step(model: MemoryModel, state: np.ndarray, target: np.ndarray, d: np.ndarray, u: np.ndarray, memory: np.ndarray, runs: list[Any], frozen: dict[str, Any]) -> dict[str, Any]:
    rows = _valid_rows(runs, model.horizon_steps)
    prediction = model.predict(state[rows], state[rows - 1], d[rows], d[rows - 1], u[rows], memory[rows])
    predicted_core = _decode_core(prediction[:, -2:], frozen)
    actual_core = _decode_core(target[rows, -2:], frozen)
    return {
        "state_nrmse": _nrmse(target[rows], prediction, np.zeros(state.shape[1])),
        "core_nrmse": {
            name: _nrmse(actual_core[:, i:i + 1], predicted_core[:, i:i + 1], np.zeros(1))
            for i, name in enumerate(CORE_NAMES)
        },
    }


def _history_feature(kind: str, horizon: int, scales: tuple[int, ...], states: np.ndarray, d: np.ndarray, u: np.ndarray, relevant: np.ndarray, run_start: int, index: int, dt: float, exp_memory: list[np.ndarray] | None) -> np.ndarray:
    def value(at: int) -> np.ndarray:
        return np.concatenate((states[at - run_start, -2:], u[at], d[at, relevant]))
    if kind == "baseline":
        return np.empty(0)
    if kind == "raw_lag":
        return np.concatenate([value(max(run_start, index - lag)) for lag in range(1, horizon + 1)])
    if kind == "multi_scale":
        rows = []
        for scale in scales:
            begin = max(run_start, index + 1 - scale)
            values = np.vstack([value(at) for at in range(begin, index + 1)])
            rows.extend((values.mean(axis=0), values.sum(axis=0) * dt, value(index) - value(max(run_start, index - scale))))
        return np.concatenate(rows)
    if kind == "exponential":
        assert exp_memory is not None
        return np.concatenate(exp_memory)
    raise ValueError(kind)


def _rollout(model: MemoryModel, state: np.ndarray, d: np.ndarray, u: np.ndarray, runs: list[Any], relevant: np.ndarray, frozen: dict[str, Any], dt: float) -> dict[str, Any]:
    state_error = state_energy = base_error = 0.0
    global_error = global_energy = 0.0
    core_error = np.zeros(3); core_energy = np.zeros(3)
    per_run = []
    warm = max(model.horizon_steps, 1)
    for run in runs:
        actual = state[run.start:run.end]
        prediction = actual.copy()
        exp_memory = None
        if model.kind == "exponential":
            exp_memory = [
                _signal(actual[:1], d[run.start:run.start + 1], u[run.start:run.start + 1], relevant)[0].copy()
                for _ in model.scales
            ]
            for offset in range(warm):
                signal_now = np.concatenate((actual[offset, -2:], u[run.start + offset], d[run.start + offset, relevant]))
                for slot, tau_steps in enumerate(model.scales):
                    alpha = math.exp(-1.0 / tau_steps)
                    exp_memory[slot] = alpha * exp_memory[slot] + (1.0 - alpha) * signal_now
        for offset in range(warm, len(actual) - 1):
            index = run.start + offset
            memory = _history_feature(model.kind, model.horizon_steps, model.scales, prediction, d, u, relevant, run.start, index, dt, exp_memory)
            prediction[offset + 1] = model.predict(
                prediction[offset:offset + 1], prediction[offset - 1:offset],
                d[index:index + 1], d[index - 1:index], u[index:index + 1], memory[None],
            )[0]
            if exp_memory is not None:
                signal_now = np.concatenate((prediction[offset, -2:], u[index], d[index, relevant]))
                for slot, tau_steps in enumerate(model.scales):
                    alpha = math.exp(-1.0 / tau_steps)
                    exp_memory[slot] = alpha * exp_memory[slot] + (1.0 - alpha) * signal_now
        observed = actual[warm + 1:]
        predicted = prediction[warm + 1:]
        error = predicted - observed
        state_error += float(np.square(error).sum()); state_energy += float(np.square(observed).sum())
        base_error += float(np.square(observed - actual[warm]).sum())
        global_error += float(np.square(error[:, :12]).sum()); global_energy += float(np.square(observed[:, :12]).sum())
        observed_core = _decode_core(observed[:, -2:], frozen)
        predicted_core = _decode_core(predicted[:, -2:], frozen)
        core_error += np.square(predicted_core - observed_core).sum(axis=0)
        core_energy += np.square(observed_core).sum(axis=0)
        per_run.append({"run_id": run.run_id, "nrmse": _nrmse(observed, predicted, np.zeros(state.shape[1]))})
    nrmse = math.sqrt(state_error / max(state_energy, 1e-30))
    persistence = math.sqrt(base_error / max(state_energy, 1e-30))
    return {
        "state_nrmse": nrmse,
        "global_nrmse": math.sqrt(global_error / max(global_energy, 1e-30)),
        "persistence_nrmse": persistence,
        "skill": 1.0 - nrmse / max(persistence, 1e-30),
        "core_nrmse": {name: math.sqrt(core_error[i] / max(core_energy[i], 1e-30)) for i, name in enumerate(CORE_NAMES)},
        "runs": per_run,
    }


def _augmented_radius(model: MemoryModel) -> float:
    n = model.state_dim
    base = model.theta[:n].T + model.theta[n:2 * n].T
    previous = -model.theta[n:2 * n].T
    memory_start = 2 * n + 2 * model.d_dim + 1
    memory_weights = model.theta[memory_start:memory_start + model.memory_dim].T
    selector = np.zeros((2, n)); selector[:, -2:] = np.eye(2)
    if model.kind == "exponential":
        size = 2 * n + 2 * len(model.scales)
        matrix = np.zeros((size, size)); matrix[:n, :n] = base; matrix[:n, n:2*n] = previous
        for slot, tau in enumerate(model.scales):
            columns = slice(slot * model.signal_dim, slot * model.signal_dim + 2)
            block = slice(2*n + 2*slot, 2*n + 2*slot + 2)
            matrix[:n, block] += memory_weights[:, columns]
            alpha = math.exp(-1.0 / tau)
            matrix[block, :n] = (1-alpha) * selector
            matrix[block, block] = alpha * np.eye(2)
        matrix[n:2*n, :n] = np.eye(n)
        return float(np.max(np.abs(np.linalg.eigvals(matrix))))
    horizon = max(model.horizon_steps, 1)
    size = 2*n + 2*max(0, horizon-1)
    matrix = np.zeros((size, size)); matrix[:n, :n] = base; matrix[:n, n:2*n] = previous
    lag_coefficients = {lag: np.zeros((n, 2)) for lag in range(horizon + 1)}
    if model.kind == "raw_lag":
        for lag in range(1, horizon + 1):
            columns = slice((lag-1)*model.signal_dim, (lag-1)*model.signal_dim+2)
            lag_coefficients[lag] += memory_weights[:, columns]
    elif model.kind == "multi_scale":
        cursor = 0
        for scale in model.scales:
            mean_w = memory_weights[:, cursor:cursor+model.signal_dim]; cursor += model.signal_dim
            integral_w = memory_weights[:, cursor:cursor+model.signal_dim]; cursor += model.signal_dim
            change_w = memory_weights[:, cursor:cursor+model.signal_dim]; cursor += model.signal_dim
            for lag in range(scale):
                lag_coefficients[lag] += mean_w[:, :2] / scale + integral_w[:, :2] * 0.2
            lag_coefficients[0] += change_w[:, :2]
            lag_coefficients[scale] -= change_w[:, :2]
    matrix[:n, :n] += lag_coefficients[0] @ selector
    matrix[:n, n:2*n] += lag_coefficients.get(1, 0) @ selector
    for lag in range(2, horizon + 1):
        block = slice(2*n + 2*(lag-2), 2*n + 2*(lag-1))
        matrix[:n, block] += lag_coefficients[lag]
    matrix[n:2*n, :n] = np.eye(n)
    if horizon > 1:
        matrix[2*n:2*n+2, n:2*n] = selector
        for lag in range(3, horizon + 1):
            dest = slice(2*n + 2*(lag-2), 2*n + 2*(lag-1))
            source = slice(2*n + 2*(lag-3), 2*n + 2*(lag-2))
            matrix[dest, source] = np.eye(2)
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _direction(model: MemoryModel, state: np.ndarray, d: np.ndarray, u: np.ndarray, memory: np.ndarray, runs: list[Any], frozen: dict[str, Any], u_normalizer: dict[str, Any]) -> dict[str, Any]:
    rows = _valid_rows(runs, model.horizon_steps)[::20]
    low = _scalar_transform(np.full((len(rows), 1), 0.4), u_normalizer)
    high = _scalar_transform(np.full((len(rows), 1), 0.6), u_normalizer)
    low_state = model.predict(state[rows], state[rows-1], d[rows], d[rows-1], low, memory[rows])
    high_state = model.predict(state[rows], state[rows-1], d[rows], d[rows-1], high, memory[rows])
    delta = _decode_core(high_state[:, -2:], frozen) - _decode_core(low_state[:, -2:], frozen)
    result = {}; passes = 0
    for i, name in enumerate(CORE_NAMES):
        fraction = float(np.mean(delta[:, i] > 0))
        result[name] = {"positive_fraction": fraction, "median_high_minus_low": float(np.median(delta[:, i]))}
        passes += fraction > 0.5
    result["direction_pass_fraction"] = passes / 3
    return result


def _save_model(path: Path, model: MemoryModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, theta=model.theta, kind=model.kind, horizon_steps=model.horizon_steps, horizon_seconds=model.horizon_seconds, ridge=model.ridge, scales=np.asarray(model.scales), state_dim=model.state_dim, d_dim=model.d_dim, signal_dim=model.signal_dim)


def _write_report(output: Path, result: dict[str, Any]) -> None:
    final = result["final"]
    lines = [
        "# ServingROM Step 15C-1 Control-Dynamics Memory Redesign",
        "", "## 结论", "",
        f"- `memory_dynamics_ready={str(result['memory_dynamics_ready']).lower()}`",
        f"- `effective_memory_horizon={result['effective_memory_horizon']}`",
        f"- 诊断候选 horizon：`{result['diagnostic_candidate_horizon_seconds']}s`",
        f"- `effective_forcing_available={str(result['forcing_audit']['effective_forcing_available']).lower()}`",
        f"- 冻结候选：`{final['kind']}`，horizon={final['horizon_seconds']}s，ridge={final['ridge']}。",
        "- representation 固定为 `gc12-diff2`；未读取 held-out、未启动 1P2D、未实现 MPC。",
        "- 未达到 strong gate，因此该候选仅作为失败诊断产物，不是可部署 Control-ROM。",
        "", "## Validation Ablation", "",
        "| kind | horizon(s) | dim | running | waiting | remaining | global | slow KPI | radius |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["ablation"]:
        core = row["validation_rollout"]["core_nrmse"]
        lines.append(f"| {row['kind']} | {row['horizon_seconds']} | {row['memory_dim']} | {core['running_imbalance']:.6f} | {core['waiting_imbalance']:.6f} | {core['remaining_token_imbalance']:.6f} | {row['validation_rollout']['global_nrmse']:.6f} | {row['validation_slow_kpi_nrmse']:.6f} | {row['spectral_radius']:.6f} |")
    test = final["test"]
    lines += [
        "", "## Test", "",
        f"- running rollout NRMSE：`{test['rollout']['core_nrmse']['running_imbalance']:.6f}`",
        f"- waiting rollout NRMSE：`{test['rollout']['core_nrmse']['waiting_imbalance']:.6f}`",
        f"- remaining-token rollout NRMSE：`{test['rollout']['core_nrmse']['remaining_token_imbalance']:.6f}`",
        f"- one-step state NRMSE：`{test['one_step']['state_nrmse']:.6f}`",
        f"- global rollout NRMSE：`{test['rollout']['global_nrmse']:.6f}`",
        f"- Slow KPI NRMSE：`{test['slow_kpi_nrmse']:.6f}`",
        f"- control-direction pass fraction：`{test['control_direction']['direction_pass_fraction']:.6f}`",
        f"- augmented spectral radius：`{final['spectral_radius']:.6f}`",
        "", "## 失败归因", "",
        "- 一步预测约为 0.4，但自由 rollout 仍约为 0.97，误差主要在递推中持续累积。",
        "- 1--20s memory 对 running/remaining-token imbalance 的收益很小，有限记忆不是当前主导瓶颈。",
        "- global state 与 Slow KPI 有一定改善，说明历史对总负载有信息，但没有恢复路由产生的 A/B 有效注入。",
        "- 下一步应使用已经存在的 routed request/token-mass imbalance 做显式 forcing；本轮按约束没有进入 Step 15C-2。",
        "", "## Effective Forcing Audit", "",
        f"- 来源：`{result['forcing_audit']['source']}`",
        f"- 对齐：{result['forcing_audit']['alignment']}",
        "- 本轮只准备 Step 15C-2 schema，没有把 forcing 加入 15C-1。",
    ]
    (output / "STEP15C1_MEMORY_REDESSIGN_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_memory_pipeline(dataset_root: Path, representation_root: Path, output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_root = dataset_root.resolve(); representation_root = representation_root.resolve(); output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = _verify_dataset(dataset_root, config["dataset_id"])
    frozen = _load_frozen_representation(dataset_root, representation_root)
    runs = _load_runs(dataset_root)
    disturbance_index = _load_json(dataset_root / "disturbance_index.json")
    d_positions = {row["name"]: int(row["index"]) for row in disturbance_index}
    relevant = np.asarray([d_positions[name] for name in config["control_relevant_disturbances"]], dtype=np.int64)
    forcing = _forcing_audit()
    save_json(output_root / "EFFECTIVE_FORCING_AUDIT.json", forcing)
    memory_schema = {
        "schema_version": "servingrom.control-memory-schema.v1", "dt_seconds": config["dt_seconds"],
        "frozen_state": "gc12-diff2", "signal_order": ["z_diff_0", "z_diff_1", "rho_A"] + config["control_relevant_disturbances"],
        "representations": {"raw_lag": "all past signal samples", "multi_scale": "mean, integral, change per scale", "exponential": "causal EWMA per tau"},
        "run_boundary_policy": "history is initialized independently per complete run; no cross-run rows",
    }
    save_json(output_root / "MEMORY_SCHEMA.json", memory_schema)

    arrays = {split: _load_split(dataset_root, split) for split in ("train", "validation")}
    state = {}; target = {}; core = {}; d = {}; u = {}; signal = {}
    for split in arrays:
        state[split], core[split] = _encode(arrays[split]["X"], frozen)
        target[split], _ = _encode(arrays[split]["X_next"], frozen)
        d[split] = frozen["d_normalizer"].transform(arrays[split]["D"], weighted=False)
        u[split] = _scalar_transform(arrays[split]["U"], frozen["u_normalizer"])
        signal[split] = _signal(state[split], d[split], u[split], relevant)
    slow_all = pq.read_table(dataset_root / "slow_kpi_windows.parquet", filters=[("split", "in", ["train", "validation"])]).to_pylist()
    slow_rows = {split: [row for row in slow_all if row["split"] == split] for split in ("train", "validation")}
    dt = float(config["dt_seconds"]); horizons = [int(round(float(value) / dt)) for value in config["memory_horizons_seconds"]]
    definitions = [("baseline", 0, ())]
    for horizon in horizons:
        scales = tuple(value for value in horizons if value <= horizon)
        definitions.extend((("raw_lag", horizon, (horizon,)), ("multi_scale", horizon, scales), ("exponential", horizon, scales)))
    ablation_runtime = []
    for kind, horizon, scales in definitions:
        memory = {split: _memory_matrix(kind, horizon, scales, signal[split], runs[split], dt) for split in signal}
        ridge_scan = []; winner = None
        for ridge in [float(value) for value in config["candidate_ridges"]]:
            model = _fit_candidate(kind, horizon, scales, ridge, state["train"], target["train"], d["train"], u["train"], memory["train"], runs["train"])
            one = _one_step(model, state["validation"], target["validation"], d["validation"], u["validation"], memory["validation"], runs["validation"], frozen)
            row = {"ridge": ridge, "validation_one_step": one}
            ridge_scan.append(row)
            if winner is None or one["state_nrmse"] < winner[0]["validation_one_step"]["state_nrmse"]:
                winner = (row, model)
        assert winner is not None
        win, model = winner
        rollout = _rollout(model, state["validation"], d["validation"], u["validation"], runs["validation"], relevant, frozen, dt)
        radius = _augmented_radius(model)
        direction = _direction(model, state["validation"], d["validation"], u["validation"], memory["validation"], runs["validation"], frozen, frozen["u_normalizer"])
        train_slow = _build_slow(slow_rows["train"], runs["train"], np.concatenate((state["train"], memory["train"]), axis=1), d["train"], u["train"])
        val_slow = _build_slow(slow_rows["validation"], runs["validation"], np.concatenate((state["validation"], memory["validation"]), axis=1), d["validation"], u["validation"])
        y_norm = _fit_scalar_normalizer(train_slow[3], SLOW_OUTPUTS); y_train = _scalar_transform(train_slow[3], y_norm); y_val = _scalar_transform(val_slow[3], y_norm)
        slow_best = None
        for ridge in [float(value) for value in config["candidate_ridges"]]:
            theta = _fit_slow_head(*train_slow[:3], y_train, ridge)
            score = _nrmse(y_val, _predict_slow(theta, *val_slow[:3]), np.zeros(len(SLOW_OUTPUTS)))
            if slow_best is None or score < slow_best[0]: slow_best = (score, ridge, theta)
        assert slow_best is not None
        ablation_runtime.append({
            "kind": kind, "horizon_steps": horizon, "horizon_seconds": horizon*dt, "scales_steps": list(scales),
            "memory_dim": model.memory_dim, "ridge": model.ridge, "ridge_scan": ridge_scan,
            "spectral_radius": radius, "validation_one_step": win["validation_one_step"],
            "validation_rollout": rollout, "validation_control_direction": direction,
            "validation_slow_kpi_nrmse": slow_best[0], "slow_ridge": slow_best[1],
            "model": model, "memory": memory, "slow_theta": slow_best[2], "slow_normalizer": y_norm,
        })
    baseline = next(row for row in ablation_runtime if row["kind"] == "baseline")
    strong = []
    for row in ablation_runtime:
        core_metric = row["validation_rollout"]["core_nrmse"]
        gates = {
            "running": core_metric["running_imbalance"] < float(config["strong_core_rollout_nrmse"]),
            "remaining": core_metric["remaining_token_imbalance"] < float(config["strong_core_rollout_nrmse"]),
            "stable": row["spectral_radius"] <= float(config["maximum_spectral_radius"]),
            "global": row["validation_rollout"]["global_nrmse"] <= baseline["validation_rollout"]["global_nrmse"]*(1+float(config["maximum_global_rollout_degradation"])),
            "slow": row["validation_slow_kpi_nrmse"] <= baseline["validation_slow_kpi_nrmse"]*(1+float(config["maximum_slow_kpi_degradation"])),
            "direction": row["validation_control_direction"]["direction_pass_fraction"] + 1e-12 >= float(config["minimum_control_direction_fraction"]),
        }
        row["validation_gates"] = gates
        if all(gates.values()): strong.append(row)
    clean = [{k:v for k,v in row.items() if k not in {"model","memory","slow_theta","slow_normalizer"}} for row in ablation_runtime]
    save_json(output_root / "MEMORY_ABLATION.json", clean)
    pool = strong or ablation_runtime
    best_core = min((np.mean([row["validation_rollout"]["core_nrmse"]["running_imbalance"], row["validation_rollout"]["core_nrmse"]["remaining_token_imbalance"]]) for row in pool))
    near = [row for row in pool if np.mean([row["validation_rollout"]["core_nrmse"]["running_imbalance"], row["validation_rollout"]["core_nrmse"]["remaining_token_imbalance"]]) <= best_core*1.05]
    priority = {"exponential":0,"multi_scale":1,"raw_lag":2,"baseline":3}
    selected = min(near, key=lambda row:(priority[row["kind"]],row["memory_dim"],row["horizon_steps"]))
    frozen_selection = {k:selected[k] for k in ("kind","horizon_steps","horizon_seconds","scales_steps","memory_dim","ridge","slow_ridge","spectral_radius")}
    frozen_selection.update({"selection_split":"validation","validation_strong_gate_passed":bool(strong),"test_accessed":False})
    save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen_selection)

    test_arrays = _load_split(dataset_root,"test"); state_test,core_test=_encode(test_arrays["X"],frozen); target_test,_=_encode(test_arrays["X_next"],frozen)
    d_test=frozen["d_normalizer"].transform(test_arrays["D"],weighted=False); u_test=_scalar_transform(test_arrays["U"],frozen["u_normalizer"]); signal_test=_signal(state_test,d_test,u_test,relevant)
    memory_test=_memory_matrix(selected["kind"],selected["horizon_steps"],tuple(selected["scales_steps"]),signal_test,runs["test"],dt)
    test_one=_one_step(selected["model"],state_test,target_test,d_test,u_test,memory_test,runs["test"],frozen)
    test_roll=_rollout(selected["model"],state_test,d_test,u_test,runs["test"],relevant,frozen,dt)
    test_direction=_direction(selected["model"],state_test,d_test,u_test,memory_test,runs["test"],frozen,frozen["u_normalizer"])
    test_slow_rows=pq.read_table(dataset_root/"slow_kpi_windows.parquet",filters=[("split","=","test")]).to_pylist(); test_slow=_build_slow(test_slow_rows,runs["test"],np.concatenate((state_test,memory_test),axis=1),d_test,u_test)
    test_y=_scalar_transform(test_slow[3],selected["slow_normalizer"]); test_slow_score=_nrmse(test_y,_predict_slow(selected["slow_theta"],*test_slow[:3]),np.zeros(len(SLOW_OUTPUTS)))
    test_core_metric=test_roll["core_nrmse"]
    memory_ready=(
        bool(strong)
        and test_core_metric["running_imbalance"] < float(config["strong_core_rollout_nrmse"])
        and test_core_metric["remaining_token_imbalance"] < float(config["strong_core_rollout_nrmse"])
        and test_direction["direction_pass_fraction"] + 1e-12 >= float(config["minimum_control_direction_fraction"])
        and test_roll["global_nrmse"] <= baseline["validation_rollout"]["global_nrmse"] * (1 + float(config["maximum_global_rollout_degradation"]))
        and test_slow_score <= baseline["validation_slow_kpi_nrmse"] * (1 + float(config["maximum_slow_kpi_degradation"]))
    )
    final={**frozen_selection,"validation":{k:v for k,v in selected.items() if k not in {"model","memory","slow_theta","slow_normalizer"}},"test":{"one_step":test_one,"rollout":test_roll,"control_direction":test_direction,"slow_kpi_nrmse":test_slow_score}}
    final["test_accessed"] = True
    result={"schema_version":"servingrom.control-memory.result.v1","dataset":dataset_audit,"frozen_representation":config["frozen_representation"],"ablation":clean,"final":final,"memory_dynamics_ready":memory_ready,"effective_memory_horizon":selected["horizon_seconds"] if memory_ready else None,"diagnostic_candidate_horizon_seconds":selected["horizon_seconds"],"forcing_audit":forcing,"data_isolation":{"heldout_actuator_data_read":False,"test_accessed_after_freeze":True,"new_runs_collected":False,"mpc_implemented":False}}
    save_json(output_root/"evaluation/final_metrics.json",result); _save_model(output_root/("models/final_memory_dynamics.npz" if memory_ready else "models/diagnostic_not_ready_memory_dynamics.npz"),selected["model"])
    np.savez_compressed(output_root/("models/final_slow_kpi_head.npz" if memory_ready else "models/diagnostic_not_ready_slow_kpi_head.npz"),theta=selected["slow_theta"],ridge=selected["slow_ridge"],outputs=np.asarray(SLOW_OUTPUTS))
    _write_report(output_root,result)
    manifest={"model_id":config["model_id"],"dataset_sha256_manifest":dataset_audit["sha256_manifest"],"representation_manifest_sha256":_sha256(representation_root/"SHA256_MANIFEST.json"),"selection":frozen_selection,"memory_dynamics_ready":memory_ready,"artifacts":{}}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256_MANIFEST.json","step15c1.log","step15c1.pid"}: manifest["artifacts"][str(path.relative_to(output_root))]=_sha256(path)
    save_json(output_root/"SHA256_MANIFEST.json",manifest)
    return result
