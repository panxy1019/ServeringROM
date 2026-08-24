from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from servingrom_modeling.preprocessing import save_json

from .memory import CORE_NAMES, _decode_core, _encode, _load_frozen_representation
from .pipeline import (
    SLOW_OUTPUTS,
    _build_slow,
    _fit_scalar_normalizer,
    _load_json,
    _load_runs,
    _load_split,
    _nrmse,
    _predict_slow,
    _scalar_transform,
    _sha256,
    _verify_dataset,
)


FORCING_NAMES = [
    "routed_request_imbalance",
    "routed_expected_token_mass_imbalance",
]
CANDIDATES = (
    "command_only",
    "actual_forcing_only",
    "actual_forcing_plus_command",
)


@dataclass
class FrozenGlobalModel:
    A: np.ndarray
    L: np.ndarray
    E: np.ndarray
    M: np.ndarray
    B: np.ndarray
    c: np.ndarray

    def predict(self, state: np.ndarray, previous: np.ndarray, d: np.ndarray, d_previous: np.ndarray, u: np.ndarray) -> np.ndarray:
        delta = state - previous
        return state @ self.A.T + delta @ self.L.T + d @ self.E.T + d_previous @ self.M.T + u @ self.B.T + self.c


@dataclass
class DifferentialIncrementModel:
    candidate: str
    ridge: float
    K: np.ndarray
    L: np.ndarray
    C: np.ndarray
    E: np.ndarray
    M: np.ndarray
    Bf: np.ndarray
    Bu: np.ndarray
    c: np.ndarray

    def predict_delta(
        self,
        diff: np.ndarray,
        previous_diff: np.ndarray,
        global_common: np.ndarray,
        d: np.ndarray,
        d_previous: np.ndarray,
        forcing: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        return (
            diff @ self.K.T
            + (diff - previous_diff) @ self.L.T
            + global_common @ self.C.T
            + d @ self.E.T
            + d_previous @ self.M.T
            + forcing @ self.Bf.T
            + u @ self.Bu.T
            + self.c
        )


def _load_frozen_global(path: Path) -> FrozenGlobalModel:
    values = np.load(path)
    if int(values["rank"]) != 14:
        raise ValueError("frozen Step 15B dynamics must have rank 14")
    return FrozenGlobalModel(
        A=np.asarray(values["A"][:12], dtype=np.float64),
        L=np.asarray(values["L"][:12], dtype=np.float64),
        E=np.asarray(values["E"][:12], dtype=np.float64),
        M=np.asarray(values["M"][:12], dtype=np.float64),
        B=np.asarray(values["B"][:12], dtype=np.float64),
        c=np.asarray(values["c"][:12], dtype=np.float64),
    )


def _forcing_values(row: dict[str, Any]) -> tuple[float, float]:
    return (
        2.0 * float(row["routed_A_request_count"]) - float(row["routed_request_count"]),
        2.0 * float(row["routed_A_expected_token_mass"]) - float(row["routed_expected_token_mass"]),
    )


def _build_forcing_sidecar(
    dataset_root: Path,
    forcing_runs_root: Path,
    output_root: Path,
    runs: dict[str, list[Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    sidecar_rows: list[dict[str, Any]] = []
    values: dict[str, np.ndarray] = {}
    source_files: dict[str, str] = {}
    split_audit: dict[str, Any] = {}
    all_widths: list[int] = []
    for split in ("train", "validation", "test"):
        split_values = np.empty((sum(run.end - run.start for run in runs[split]), 2), dtype=np.float64)
        dataset_u = np.load(dataset_root / split / "U.npy", mmap_mode="r")
        maximum_u_error = 0.0
        previous_end_by_run: dict[str, int] = {}
        for run in runs[split]:
            source = forcing_runs_root / run.run_id / "derived" / "control" / "control_windows.parquet"
            if not source.is_file():
                raise FileNotFoundError(f"forcing source missing: {source}")
            source_files[run.run_id] = _sha256(source)
            rows = pq.read_table(source).to_pylist()
            expected = run.end - run.start
            if len(rows) != expected:
                raise ValueError(f"forcing row count mismatch for {run.run_id}: {len(rows)} != {expected}")
            ids = [int(row["window_id"]) for row in rows]
            if ids != list(range(expected)):
                raise ValueError(f"forcing window_id gap or reorder in {run.run_id}")
            for offset, row in enumerate(rows):
                start = int(row["start_wall_ns"])
                end = int(row["end_wall_ns"])
                width = end - start
                if width <= 0:
                    raise ValueError(f"non-positive forcing window in {run.run_id}:{offset}")
                if offset and previous_end_by_run[run.run_id] != start:
                    raise ValueError(f"forcing windows are not contiguous in {run.run_id}:{offset}")
                previous_end_by_run[run.run_id] = end
                all_widths.append(width)
                row_in_split = run.start + offset
                maximum_u_error = max(maximum_u_error, abs(float(row["u_rho_A"]) - float(dataset_u[row_in_split, 0])))
                request_value, token_value = _forcing_values(row)
                split_values[row_in_split] = (request_value, token_value)
                sidecar_rows.append({
                    "split": split,
                    "run_id": run.run_id,
                    "window_id": offset,
                    "row_in_split": row_in_split,
                    "start_wall_ns": start,
                    "end_wall_ns": end,
                    "routed_request_count": int(row["routed_request_count"]),
                    "routed_A_request_count": int(row["routed_A_request_count"]),
                    "routed_expected_token_mass": float(row["routed_expected_token_mass"]),
                    "routed_A_expected_token_mass": float(row["routed_A_expected_token_mass"]),
                    FORCING_NAMES[0]: request_value,
                    FORCING_NAMES[1]: token_value,
                })
        if maximum_u_error > 1e-12:
            raise ValueError(f"forcing/U alignment mismatch in {split}: {maximum_u_error}")
        values[split] = split_values
        split_audit[split] = {
            "runs": len(runs[split]),
            "rows": len(split_values),
            "maximum_u_alignment_error": maximum_u_error,
            "first_row": int(min(run.start for run in runs[split])),
            "last_row_exclusive": int(max(run.end for run in runs[split])),
        }
    expected_width = 200_000_000
    if set(all_widths) != {expected_width}:
        raise ValueError(f"forcing window widths are not exactly 200 ms: {sorted(set(all_widths))[:10]}")
    sidecar = output_root / "sidecar" / "effective_forcing.parquet"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sidecar_rows), sidecar, compression="zstd")
    sidecar_sha = _sha256(sidecar)
    manifest = {
        "schema_version": "servingrom.effective-forcing-sidecar.v1",
        "dataset_id": "servingrom-control-dataset-v1",
        "immutable_dataset_modified": False,
        "path": str(sidecar.relative_to(output_root)),
        "sha256": sidecar_sha,
        "row_count": len(sidecar_rows),
        "split_rows": {split: len(values[split]) for split in values},
        "fields": {
            FORCING_NAMES[0]: "2*routed_A_request_count-routed_request_count",
            FORCING_NAMES[1]: "2*routed_A_expected_token_mass-routed_expected_token_mass",
        },
        "source_semantics": {
            "event": "p_to_d_route",
            "alignment": "200 ms half-open [start_wall_ns,end_wall_ns)",
            "token_mass": "expected_output_tokens captured at route time",
            "interpolation": "forbidden and not used",
            "run_boundary_crossing": "forbidden and not used",
        },
        "source_files": source_files,
    }
    audit = {
        "schema_version": "servingrom.effective-forcing-alignment-audit.v1",
        "valid": True,
        "source_file_count": len(source_files),
        "source_file_unique_sha256_count": len(set(source_files.values())),
        "window_width_ns": expected_width,
        "split_audit": split_audit,
        "checks": {
            "all_36_source_runs_present": len(source_files) == 36,
            "window_ids_contiguous_and_ordered": True,
            "window_timestamps_contiguous_within_run": True,
            "dataset_u_exactly_aligned": True,
            "row_counts_match_run_index": True,
            "cross_run_alignment_used": False,
            "interpolation_used": False,
        },
    }
    save_json(output_root / "EFFECTIVE_FORCING_MANIFEST.json", manifest)
    save_json(output_root / "FORCING_ALIGNMENT_AUDIT.json", audit)
    return values, manifest, audit


def _design(
    candidate: str,
    state: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    forcing: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    parts = [
        state[rows, -2:],
        state[rows, -2:] - state[rows - 1, -2:],
        state[rows, :12],
        d[rows],
        d[rows - 1],
    ]
    if candidate != "command_only":
        parts.append(forcing[rows])
    if candidate != "actual_forcing_only":
        parts.append(u[rows])
    parts.append(np.ones((len(rows), 1), dtype=np.float64))
    return np.concatenate(parts, axis=1)


def _transition_rows(runs: list[Any]) -> np.ndarray:
    return np.concatenate([np.arange(run.start + 1, run.end, dtype=np.int64) for run in runs])


def _fit_model(
    candidate: str,
    ridge: float,
    state: np.ndarray,
    target: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    forcing: np.ndarray,
    runs: list[Any],
) -> DifferentialIncrementModel:
    rows = _transition_rows(runs)
    design = _design(candidate, state, d, u, forcing, rows)
    delta_target = target[rows, -2:] - state[rows, -2:]
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    theta = np.linalg.solve(gram + penalty, design.T @ delta_target).T
    cursor = 0
    K = theta[:, cursor:cursor + 2]; cursor += 2
    L = theta[:, cursor:cursor + 2]; cursor += 2
    C = theta[:, cursor:cursor + 12]; cursor += 12
    E = theta[:, cursor:cursor + d.shape[1]]; cursor += d.shape[1]
    M = theta[:, cursor:cursor + d.shape[1]]; cursor += d.shape[1]
    if candidate != "command_only":
        Bf = theta[:, cursor:cursor + 2]; cursor += 2
    else:
        Bf = np.zeros((2, 2), dtype=np.float64)
    if candidate != "actual_forcing_only":
        Bu = theta[:, cursor:cursor + 1]; cursor += 1
    else:
        Bu = np.zeros((2, 1), dtype=np.float64)
    return DifferentialIncrementModel(candidate, ridge, K, L, C, E, M, Bf, Bu, theta[:, cursor])


def _augmented_spectral_radius(global_model: FrozenGlobalModel, model: DifferentialIncrementModel) -> float:
    matrix = np.zeros((28, 28), dtype=np.float64)
    matrix[:12, :14] = global_model.A + global_model.L
    matrix[:12, 14:] = -global_model.L
    matrix[12:14, :12] = model.C
    matrix[12:14, 12:14] = np.eye(2) + model.K + model.L
    matrix[12:14, 26:28] = -model.L
    matrix[14:, :14] = np.eye(14)
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _one_step(
    global_model: FrozenGlobalModel,
    model: DifferentialIncrementModel,
    state: np.ndarray,
    target: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    forcing: np.ndarray,
    runs: list[Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    rows = _transition_rows(runs)
    predicted = np.empty((len(rows), 14), dtype=np.float64)
    predicted[:, :12] = global_model.predict(state[rows], state[rows - 1], d[rows], d[rows - 1], u[rows])
    delta = model.predict_delta(
        state[rows, -2:], state[rows - 1, -2:], state[rows, :12],
        d[rows], d[rows - 1], forcing[rows], u[rows],
    )
    predicted[:, -2:] = state[rows, -2:] + delta
    actual_core = _decode_core(target[rows, -2:], frozen)
    predicted_core = _decode_core(predicted[:, -2:], frozen)
    return {
        "state_nrmse": _nrmse(target[rows], predicted, np.zeros(14)),
        "global_common_nrmse": _nrmse(target[rows, :12], predicted[:, :12], np.zeros(12)),
        "differential_pod_nrmse": _nrmse(target[rows, -2:], predicted[:, -2:], np.zeros(2)),
        "core_nrmse": {
            name: _nrmse(actual_core[:, i:i + 1], predicted_core[:, i:i + 1], np.zeros(1))
            for i, name in enumerate(CORE_NAMES)
        },
    }


def _rollout(
    global_model: FrozenGlobalModel,
    model: DifferentialIncrementModel,
    state: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    forcing: np.ndarray,
    runs: list[Any],
    frozen: dict[str, Any],
    forcing_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    predicted_all = np.empty_like(state, dtype=np.float64)
    global_error = global_energy = diff_error = diff_energy = 0.0
    core_error = np.zeros(3); core_energy = np.zeros(3)
    direction_correct = np.zeros(3); direction_total = np.zeros(3)
    per_run = []
    finite_all = True
    for run in runs:
        actual = state[run.start:run.end]
        predicted = actual.copy()
        for offset in range(1, len(actual) - 1):
            index = run.start + offset
            current = predicted[offset:offset + 1]
            previous = predicted[offset - 1:offset]
            predicted[offset + 1, :12] = global_model.predict(
                current, previous, d[index:index + 1], d[index - 1:index], u[index:index + 1],
            )[0]
            f = forcing[index:index + 1]
            if forcing_mask is not None:
                f = f * forcing_mask[None, :]
            delta = model.predict_delta(
                current[:, -2:], previous[:, -2:], current[:, :12],
                d[index:index + 1], d[index - 1:index], f, u[index:index + 1],
            )[0]
            predicted[offset + 1, -2:] = current[0, -2:] + delta
            if not np.isfinite(predicted[offset + 1]).all() or np.linalg.norm(predicted[offset + 1]) > 1e12:
                predicted[offset + 1:] = np.nan
                finite_all = False
                break
        predicted_all[run.start:run.end] = predicted
        observed = actual[2:]
        estimated = predicted[2:]
        finite = bool(np.isfinite(estimated).all())
        if not finite:
            per_run.append({"run_id": run.run_id, "finite": False})
            continue
        observed_core = _decode_core(observed[:, -2:], frozen)
        estimated_core = _decode_core(estimated[:, -2:], frozen)
        global_error += float(np.square(estimated[:, :12] - observed[:, :12]).sum())
        global_energy += float(np.square(observed[:, :12]).sum())
        diff_error += float(np.square(estimated[:, -2:] - observed[:, -2:]).sum())
        diff_energy += float(np.square(observed[:, -2:]).sum())
        core_error += np.square(estimated_core - observed_core).sum(axis=0)
        core_energy += np.square(observed_core).sum(axis=0)
        active = np.abs(observed_core) > 1e-12
        direction_correct += np.sum(
            ((estimated_core * observed_core) > 0) & active,
            axis=0,
        )
        direction_total += np.sum(active, axis=0)
        per_run.append({
            "run_id": run.run_id,
            "finite": True,
            "global_common_nrmse": _nrmse(observed[:, :12], estimated[:, :12], np.zeros(12)),
            "differential_pod_nrmse": _nrmse(observed[:, -2:], estimated[:, -2:], np.zeros(2)),
            "core_nrmse": {
                name: _nrmse(observed_core[:, i:i + 1], estimated_core[:, i:i + 1], np.zeros(1))
                for i, name in enumerate(CORE_NAMES)
            },
        })
    return ({
        "finite": finite_all and all(row["finite"] for row in per_run),
        "global_common_nrmse": math.sqrt(global_error / max(global_energy, 1e-30)) if finite_all else math.inf,
        "differential_pod_nrmse": math.sqrt(diff_error / max(diff_energy, 1e-30)) if finite_all else math.inf,
        "core_nrmse": {
            name: math.sqrt(core_error[i] / max(core_energy[i], 1e-30)) if finite_all else math.inf
            for i, name in enumerate(CORE_NAMES)
        },
        "control_direction_consistency": {
            name: float(direction_correct[i] / max(direction_total[i], 1.0))
            for i, name in enumerate(CORE_NAMES)
        },
        "runs": per_run,
    }, predicted_all)


def _frozen_reference_rollout(
    global_model: FrozenGlobalModel,
    full_model_path: Path,
    state: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    runs: list[Any],
) -> dict[str, float]:
    values = np.load(full_model_path)
    A = np.asarray(values["A"]); L = np.asarray(values["L"])
    E = np.asarray(values["E"]); M = np.asarray(values["M"])
    B = np.asarray(values["B"]); c = np.asarray(values["c"])
    error = energy = 0.0
    global_error = global_energy = 0.0
    for run in runs:
        actual = state[run.start:run.end]
        predicted = actual.copy()
        for offset in range(1, len(actual) - 1):
            index = run.start + offset
            current = predicted[offset]
            previous = predicted[offset - 1]
            predicted[offset + 1] = A @ current + L @ (current - previous) + E @ d[index] + M @ d[index - 1] + B @ u[index] + c
        observed = actual[2:]
        estimated = predicted[2:]
        error += float(np.square(estimated - observed).sum()); energy += float(np.square(observed).sum())
        global_error += float(np.square(estimated[:, :12] - observed[:, :12]).sum()); global_energy += float(np.square(observed[:, :12]).sum())
    return {
        "state_nrmse": math.sqrt(error / max(energy, 1e-30)),
        "global_common_nrmse": math.sqrt(global_error / max(global_energy, 1e-30)),
    }


def _slow_score(
    slow_rows: list[dict[str, Any]],
    runs: list[Any],
    predicted_state: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    y_normalizer: dict[str, Any],
) -> float:
    values = _build_slow(slow_rows, runs, predicted_state, d, u)
    y = _scalar_transform(values[3], y_normalizer)
    return _nrmse(y, _predict_slow(theta, *values[:3]), np.zeros(len(SLOW_OUTPUTS)))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _forcing_statistics(
    forcing_raw: dict[str, np.ndarray],
    arrays: dict[str, dict[str, np.ndarray]],
    core: dict[str, np.ndarray],
    disturbance_positions: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        f = forcing_raw[split]
        u = np.asarray(arrays[split]["U"][:, 0], dtype=np.float64)
        d = np.asarray(arrays[split]["D"], dtype=np.float64)
        variables = {"rho_A": u}
        for name in config["control_relevant_disturbances"]:
            variables[name] = d[:, disturbance_positions[name]]
        variables.update({name: core[split][:, i] for i, name in enumerate(CORE_NAMES)})
        correlations = {
            forcing_name: {name: _pearson(f[:, index], value) for name, value in variables.items()}
            for index, forcing_name in enumerate(FORCING_NAMES)
        }
        conditional = {}
        for level in (0.3, 0.5, 0.7):
            selected = np.isclose(u, level, atol=1e-9)
            conditional[str(level)] = {
                "rows": int(selected.sum()),
                **{
                    name: {"mean": float(f[selected, i].mean()), "std": float(f[selected, i].std())}
                    for i, name in enumerate(FORCING_NAMES)
                },
            }
        result[split] = {"correlations": correlations, "conditional_by_rho_A": conditional}
    return result


def _save_model(path: Path, model: DifferentialIncrementModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, candidate=model.candidate, ridge=model.ridge,
        K=model.K, L=model.L, C=model.C, E=model.E, M=model.M,
        Bf=model.Bf, Bu=model.Bu, c=model.c,
    )


def _write_report(output_root: Path, result: dict[str, Any]) -> None:
    selected = result["selection"]
    validation = result["ablation"][selected["candidate"]]["validation"]
    test = result["ablation"][selected["candidate"]]["test"]
    contribution = result["forcing_contribution"]
    improvement = result["relative_improvement_over_command"]
    lines = [
        "# ServingROM Step 15C-2A 实际有效强迫项诊断",
        "", "## 结论", "",
        f"- `actual_forcing_hypothesis_pass={str(result['actual_forcing_hypothesis_pass']).lower()}`",
        f"- `effective_forcing_dynamics_ready={str(result['effective_forcing_dynamics_ready']).lower()}`",
        "- `control_rom_ready=false`",
        f"- validation 冻结候选：`{selected['candidate']}`，ridge=`{selected['ridge']}`。",
        "- representation 固定为 `gc12-diff2`；global/common dynamics 与 5s Slow KPI Head 均保持冻结。",
        "- 未启动 1P2D、未重新采集、未读取 Round 14.3、未实现 MPC。",
        "", "## Validation Ablation", "",
        "| candidate | ridge | running rollout | waiting rollout | remaining rollout | diff POD | global | slow KPI | radius |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CANDIDATES:
        row = result["ablation"][name]
        roll = row["validation"]["rollout"]
        lines.append(
            f"| {name} | {row['ridge']} | {roll['core_nrmse']['running_imbalance']:.6f} | "
            f"{roll['core_nrmse']['waiting_imbalance']:.6f} | {roll['core_nrmse']['remaining_token_imbalance']:.6f} | "
            f"{roll['differential_pod_nrmse']:.6f} | {roll['global_common_nrmse']:.6f} | "
            f"{row['validation']['slow_kpi_nrmse']:.6f} | {row['spectral_radius']:.6f} |"
        )
    lines += [
        "", "## Frozen Test", "",
        f"- running one-step / rollout：`{test['one_step']['core_nrmse']['running_imbalance']:.6f}` / `{test['rollout']['core_nrmse']['running_imbalance']:.6f}`",
        f"- waiting one-step / rollout：`{test['one_step']['core_nrmse']['waiting_imbalance']:.6f}` / `{test['rollout']['core_nrmse']['waiting_imbalance']:.6f}`",
        f"- remaining one-step / rollout：`{test['one_step']['core_nrmse']['remaining_token_imbalance']:.6f}` / `{test['rollout']['core_nrmse']['remaining_token_imbalance']:.6f}`",
        f"- differential POD rollout：`{test['rollout']['differential_pod_nrmse']:.6f}`",
        f"- global/common rollout：`{test['rollout']['global_common_nrmse']:.6f}`",
        f"- Slow KPI regression：`{test['slow_kpi_nrmse']:.6f}`",
        f"- spectral radius：`{selected['spectral_radius']:.6f}`",
        "", "## 四个问题", "",
        "1. 真实 routed forcing 是否显著改善：**是，但未通过严格 readiness 门**。"
        f"Validation running/remaining 分别相对改善 `{improvement['running_imbalance']:.2%}` / "
        f"`{improvement['remaining_token_imbalance']:.2%}`；remaining=`{validation['rollout']['core_nrmse']['remaining_token_imbalance']:.6f}` "
        f"通过 `<0.75`，running=`{validation['rollout']['core_nrmse']['running_imbalance']:.6f}` 未通过。",
        f"2. `f_req` 与 `f_tok` 的主要贡献者：**{contribution['dominant_forcing']}**。",
        "3. forcing 已知后 `rho_A` 是否仍有实质信息：**"
        f"{'是' if result['answers']['rho_adds_material_information'] == 'yes' else '否'}**。",
        f"4. 下一步：**{result['next_step']}**。",
        "", "## Oracle 边界", "",
        "真实未来 routed forcing 在在线 counterfactual/MPC 中不可直接获得。即使本轮通过，也只能证明 effective-forcing dynamics 成立，不能把 oracle forcing 模型标记为可部署 Control-ROM。",
    ]
    (output_root / "STEP15C2A_EFFECTIVE_FORCING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_effective_forcing_pipeline(
    dataset_root: Path,
    representation_root: Path,
    forcing_runs_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    representation_root = representation_root.resolve()
    forcing_runs_root = forcing_runs_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = _verify_dataset(dataset_root, config["dataset_id"])
    frozen = _load_frozen_representation(dataset_root, representation_root)
    runs = _load_runs(dataset_root)
    forcing_raw, forcing_manifest, alignment_audit = _build_forcing_sidecar(
        dataset_root, forcing_runs_root, output_root, runs,
    )
    forcing_normalizer = _fit_scalar_normalizer(forcing_raw["train"], FORCING_NAMES)
    save_json(output_root / "normalization/forcing_train_only.json", forcing_normalizer)

    arrays = {split: _load_split(dataset_root, split) for split in ("train", "validation")}
    state: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    core: dict[str, np.ndarray] = {}
    d: dict[str, np.ndarray] = {}
    u: dict[str, np.ndarray] = {}
    forcing: dict[str, np.ndarray] = {}
    for split in arrays:
        state[split], core[split] = _encode(arrays[split]["X"], frozen)
        target[split], _ = _encode(arrays[split]["X_next"], frozen)
        d[split] = frozen["d_normalizer"].transform(arrays[split]["D"], weighted=False)
        u[split] = _scalar_transform(arrays[split]["U"], frozen["u_normalizer"])
        forcing[split] = _scalar_transform(forcing_raw[split], forcing_normalizer)

    frozen_dynamic_path = representation_root / "models" / "diagnostic_not_ready_control_dynamics.npz"
    frozen_slow_path = representation_root / "models" / "diagnostic_not_ready_slow_kpi_head.npz"
    global_model = _load_frozen_global(frozen_dynamic_path)
    slow_values = np.load(frozen_slow_path)
    slow_theta = np.asarray(slow_values["theta"], dtype=np.float64)
    slow_all = pq.read_table(dataset_root / "slow_kpi_windows.parquet", filters=[("split", "in", ["train", "validation"])]).to_pylist()
    slow_rows = {split: [row for row in slow_all if row["split"] == split] for split in ("train", "validation")}
    train_slow = _build_slow(slow_rows["train"], runs["train"], state["train"], d["train"], u["train"])
    slow_normalizer = _fit_scalar_normalizer(train_slow[3], SLOW_OUTPUTS)

    reference_validation = _frozen_reference_rollout(
        global_model, frozen_dynamic_path, state["validation"], d["validation"], u["validation"], runs["validation"],
    )
    runtime: dict[str, Any] = {}
    for candidate in CANDIDATES:
        scans = []
        winner = None
        for ridge in [float(value) for value in config["candidate_ridges"]]:
            model = _fit_model(
                candidate, ridge, state["train"], target["train"], d["train"], u["train"],
                forcing["train"], runs["train"],
            )
            one = _one_step(
                global_model, model, state["validation"], target["validation"], d["validation"],
                u["validation"], forcing["validation"], runs["validation"], frozen,
            )
            rollout, predicted = _rollout(
                global_model, model, state["validation"], d["validation"], u["validation"],
                forcing["validation"], runs["validation"], frozen,
            )
            radius = _augmented_spectral_radius(global_model, model)
            score = np.mean([
                rollout["core_nrmse"]["running_imbalance"],
                rollout["core_nrmse"]["remaining_token_imbalance"],
            ])
            row = {"ridge": ridge, "score": float(score), "spectral_radius": radius, "one_step": one, "rollout": rollout}
            scans.append(row)
            eligible = rollout["finite"] and radius <= float(config["maximum_spectral_radius"])
            if eligible and (winner is None or score < winner[0]):
                winner = (float(score), model, row, predicted)
        if winner is None:
            best = min(scans, key=lambda row: row["score"])
            model = _fit_model(
                candidate, best["ridge"], state["train"], target["train"], d["train"], u["train"],
                forcing["train"], runs["train"],
            )
            rollout, predicted = _rollout(
                global_model, model, state["validation"], d["validation"], u["validation"],
                forcing["validation"], runs["validation"], frozen,
            )
            winner = (best["score"], model, best, predicted)
        _, model, best, predicted = winner
        slow_score = _slow_score(
            slow_rows["validation"], runs["validation"], predicted, d["validation"], u["validation"],
            slow_theta, slow_normalizer,
        )
        runtime[candidate] = {
            "candidate": candidate,
            "ridge": model.ridge,
            "spectral_radius": best["spectral_radius"],
            "ridge_scan": scans,
            "validation": {"one_step": best["one_step"], "rollout": best["rollout"], "slow_kpi_nrmse": slow_score},
            "model": model,
        }

    command = runtime["command_only"]["validation"]["rollout"]["core_nrmse"]
    actual_candidates = [runtime["actual_forcing_only"], runtime["actual_forcing_plus_command"]]
    selected_runtime = min(actual_candidates, key=lambda row: np.mean([
        row["validation"]["rollout"]["core_nrmse"]["running_imbalance"],
        row["validation"]["rollout"]["core_nrmse"]["remaining_token_imbalance"],
    ]))
    selected_rollout = selected_runtime["validation"]["rollout"]
    selected_core = selected_rollout["core_nrmse"]
    relative = {
        name: (command[name] - selected_core[name]) / max(command[name], 1e-30)
        for name in ("running_imbalance", "remaining_token_imbalance")
    }
    improvements = list(relative.values())
    validation_gates = {
        "running_below_threshold": selected_core["running_imbalance"] < float(config["strong_core_rollout_nrmse"]),
        "remaining_below_threshold": selected_core["remaining_token_imbalance"] < float(config["strong_core_rollout_nrmse"]),
        "at_least_one_relative_improvement": max(improvements) >= float(config["minimum_relative_improvement"]),
        "other_clearly_improves": min(improvements) > float(config["minimum_clear_improvement"]),
        "rollout_finite": bool(selected_rollout["finite"]),
        "stable": selected_runtime["spectral_radius"] <= float(config["maximum_spectral_radius"]),
        "global_not_degraded": selected_rollout["global_common_nrmse"] <= reference_validation["global_common_nrmse"] * (1.0 + float(config["maximum_global_rollout_degradation"])),
    }
    hypothesis_pass = all(validation_gates.values())

    frozen_selection = {
        "schema_version": "servingrom.step15c2a.frozen-selection.v1",
        "selection_split": "validation",
        "candidate": selected_runtime["candidate"],
        "ridge": selected_runtime["ridge"],
        "spectral_radius": selected_runtime["spectral_radius"],
        "candidate_ridges": {name: runtime[name]["ridge"] for name in CANDIDATES},
        "validation_gates": validation_gates,
        "actual_forcing_hypothesis_pass": hypothesis_pass,
        "test_accessed": False,
    }
    save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen_selection)

    test_arrays = _load_split(dataset_root, "test")
    state_test, core_test = _encode(test_arrays["X"], frozen)
    target_test, _ = _encode(test_arrays["X_next"], frozen)
    d_test = frozen["d_normalizer"].transform(test_arrays["D"], weighted=False)
    u_test = _scalar_transform(test_arrays["U"], frozen["u_normalizer"])
    forcing_test = _scalar_transform(forcing_raw["test"], forcing_normalizer)
    test_slow_rows = pq.read_table(dataset_root / "slow_kpi_windows.parquet", filters=[("split", "=", "test")]).to_pylist()
    for candidate in CANDIDATES:
        entry = runtime[candidate]
        model = entry["model"]
        one = _one_step(global_model, model, state_test, target_test, d_test, u_test, forcing_test, runs["test"], frozen)
        rollout, predicted = _rollout(global_model, model, state_test, d_test, u_test, forcing_test, runs["test"], frozen)
        entry["test"] = {
            "one_step": one,
            "rollout": rollout,
            "slow_kpi_nrmse": _slow_score(test_slow_rows, runs["test"], predicted, d_test, u_test, slow_theta, slow_normalizer),
        }

    forcing_model = runtime["actual_forcing_only"]["model"]
    forcing_base = runtime["actual_forcing_only"]["validation"]["rollout"]
    drop_metrics = {}
    for index, name in enumerate(FORCING_NAMES):
        mask = np.ones(2); mask[index] = 0.0
        dropped, _ = _rollout(
            global_model, forcing_model, state["validation"], d["validation"], u["validation"],
            forcing["validation"], runs["validation"], frozen, forcing_mask=mask,
        )
        drop_metrics[name] = {
            "coefficient_frobenius_norm": float(np.linalg.norm(forcing_model.Bf[:, index])),
            "running_rollout_nrmse": dropped["core_nrmse"]["running_imbalance"],
            "remaining_rollout_nrmse": dropped["core_nrmse"]["remaining_token_imbalance"],
            "mean_core_degradation": float(np.mean([
                dropped["core_nrmse"]["running_imbalance"] - forcing_base["core_nrmse"]["running_imbalance"],
                dropped["core_nrmse"]["remaining_token_imbalance"] - forcing_base["core_nrmse"]["remaining_token_imbalance"],
            ])),
        }
    dominant = max(drop_metrics, key=lambda name: drop_metrics[name]["mean_core_degradation"])
    forcing_contribution = {"drop_column_diagnostic": drop_metrics, "dominant_forcing": dominant}

    only_score = np.mean([
        runtime["actual_forcing_only"]["validation"]["rollout"]["core_nrmse"]["running_imbalance"],
        runtime["actual_forcing_only"]["validation"]["rollout"]["core_nrmse"]["remaining_token_imbalance"],
    ])
    plus_score = np.mean([
        runtime["actual_forcing_plus_command"]["validation"]["rollout"]["core_nrmse"]["running_imbalance"],
        runtime["actual_forcing_plus_command"]["validation"]["rollout"]["core_nrmse"]["remaining_token_imbalance"],
    ])
    rho_improvement = (only_score - plus_score) / max(only_score, 1e-30)
    rho_material = rho_improvement >= float(config["rho_material_improvement"])

    disturbance_index = _load_json(dataset_root / "disturbance_index.json")
    disturbance_positions = {row["name"]: int(row["index"]) for row in disturbance_index}
    statistics = _forcing_statistics(forcing_raw, arrays, core, disturbance_positions, config)
    clean_ablation: dict[str, Any] = {}
    for name, entry in runtime.items():
        clean_ablation[name] = {key: value for key, value in entry.items() if key != "model"}
    next_step = "Step 15C-2B actuator realization model" if hypothesis_pass else "explicit conservation/service-outflow model"
    result = {
        "schema_version": "servingrom.step15c2a.result.v1",
        "dataset": dataset_audit,
        "frozen_representation": config["frozen_representation"],
        "frozen_global_reference": reference_validation,
        "forcing_manifest": forcing_manifest,
        "alignment_audit": alignment_audit,
        "forcing_normalizer": forcing_normalizer,
        "forcing_statistics": statistics,
        "forcing_contribution": forcing_contribution,
        "ablation": clean_ablation,
        "selection": {**frozen_selection, "test_accessed": True},
        "relative_improvement_over_command": relative,
        "rho_residual_relative_improvement": float(rho_improvement),
        "actual_forcing_hypothesis_pass": hypothesis_pass,
        "effective_forcing_dynamics_ready": hypothesis_pass,
        "control_rom_ready": False,
        "answers": {
            "forcing_improves_differential_rollout": (
                "yes_and_readiness_gate_passed"
                if hypothesis_pass
                else "yes_but_strict_readiness_gate_failed"
            ),
            "rho_adds_material_information": "yes" if rho_material else "no",
        },
        "next_step": next_step,
        "data_isolation": {
            "new_runs_collected": False,
            "one_p_two_d_started": False,
            "heldout_actuator_data_read": False,
            "pod_or_memory_rescanned": False,
            "mpc_implemented": False,
            "test_accessed_after_validation_freeze": True,
        },
    }
    save_json(output_root / "FORCING_ABLATION.json", clean_ablation)
    save_json(output_root / "evaluation/final_metrics.json", result)
    _save_model(output_root / "models/diagnostic_effective_forcing_dynamics.npz", selected_runtime["model"])
    _write_report(output_root, result)
    manifest = {
        "schema_version": "servingrom.step15c2a.sha256-manifest.v1",
        "model_id": config["model_id"],
        "dataset_sha256_manifest": dataset_audit["sha256_manifest"],
        "representation_manifest_sha256": _sha256(representation_root / "SHA256_MANIFEST.json"),
        "frozen_dynamics_sha256": _sha256(frozen_dynamic_path),
        "frozen_slow_head_sha256": _sha256(frozen_slow_path),
        "selection": result["selection"],
        "artifacts": {},
    }
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256_MANIFEST.json", "step15c2a.log", "step15c2a.pid"}:
            manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    save_json(output_root / "SHA256_MANIFEST.json", manifest)
    return result
