from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from servingrom_modeling.pod import fit_pod, mode_block_contributions, reconstruction_scan
from servingrom_modeling.preprocessing import Normalizer, fit_normalizer, save_json


IMBALANCE_FEATURES = {
    "waiting_imbalance": ("decode_d1_waiting_count", "decode_d2_waiting_count"),
    "running_imbalance": ("decode_d1_running_count", "decode_d2_running_count"),
    "remaining_token_imbalance": (
        "decode_d1_expected_remaining_tokens",
        "decode_d2_expected_remaining_tokens",
    ),
}

SLOW_OUTPUTS = [
    "throughput_output_tokens_per_second",
    "goodput_output_tokens_per_second",
    "completed_requests",
    "completed_output_tokens",
    "ttft_total_ms",
    "tpot_total_ms",
    "prefill_waiting_integral_request_seconds",
    "decode_A_waiting_integral_request_seconds",
    "decode_B_waiting_integral_request_seconds",
    "decode_A_running_integral_request_seconds",
    "decode_B_running_integral_request_seconds",
    "decode_A_remaining_integral_token_seconds",
    "decode_B_remaining_integral_token_seconds",
    "decode_waiting_imbalance_integral_request_seconds",
    "decode_running_imbalance_integral_request_seconds",
    "decode_remaining_imbalance_integral_token_seconds",
    "kv_transfer_completed_bytes",
    "prefill_scheduled_tokens",
    "decode_scheduled_tokens",
]


@dataclass(frozen=True)
class RunRange:
    run_id: str
    split: str
    start: int
    end: int


@dataclass
class DynamicModel:
    rank: int
    ridge: float
    model_type: str
    A: np.ndarray
    L: np.ndarray
    E: np.ndarray
    M: np.ndarray
    B: np.ndarray
    c: np.ndarray
    N: np.ndarray | None = None

    def predict(
        self,
        z: np.ndarray,
        z_prev: np.ndarray,
        d: np.ndarray,
        d_prev: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        result = z @ self.A.T + (z - z_prev) @ self.L.T
        result += d @ self.E.T + d_prev @ self.M.T + u @ self.B.T + self.c
        if self.N is not None:
            result += (z * u[..., :1]) @ self.N.T
        return result

    def spectral_radius(self, u_nominal: float = 0.0) -> float:
        effective_a = self.A
        if self.N is not None:
            effective_a = effective_a + u_nominal * self.N
        top = np.concatenate((effective_a + self.L, -self.L), axis=1)
        bottom = np.concatenate((np.eye(self.rank), np.zeros((self.rank, self.rank))), axis=1)
        return float(np.max(np.abs(np.linalg.eigvals(np.concatenate((top, bottom), axis=0)))))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_dataset(root: Path, expected_id: str) -> dict[str, Any]:
    if "heldout" in str(root).lower():
        raise ValueError("Step 15 refuses any held-out actuator path")
    manifest = _load_json(root / "dataset_manifest.json")
    if manifest.get("dataset_id") != expected_id or manifest.get("immutable") is not True:
        raise ValueError("dataset identity or immutable seal is invalid")
    if manifest.get("dimensions") != {"D": 31, "U": 1, "X": 1804, "X_next": 1804}:
        raise ValueError("unexpected frozen dataset dimensions")
    quality = _load_json(root / "quality_summary.json")
    for name in ("control_dataset_ready", "control_identifiability_ready"):
        if quality.get(name) is not True:
            raise ValueError(f"frozen quality gate is false: {name}")
    sums = _load_json(root / "SHA256SUMS.json")
    entries = sums.get("files", sums)
    if isinstance(entries, list):
        expected = {row["path"]: row["sha256"] for row in entries}
    else:
        expected = {str(key): value for key, value in entries.items()}
    failures = []
    for relative, wanted in expected.items():
        path = root / relative
        if not path.is_file() or _sha256(path) != wanted:
            failures.append(relative)
    if failures:
        raise ValueError(f"dataset checksum failures: {failures[:5]}")
    return {
        "manifest": manifest,
        "quality": quality,
        "sha256_manifest": _sha256(root / "SHA256SUMS.json"),
        "verified_file_count": len(expected),
    }


def _load_runs(root: Path) -> dict[str, list[RunRange]]:
    rows = pq.read_table(root / "run_index.parquet").to_pylist()
    result: dict[str, list[RunRange]] = {name: [] for name in ("train", "validation", "test")}
    seen: set[str] = set()
    for row in rows:
        split = str(row["split"])
        run_id = str(row["run_id"])
        if split not in result or run_id in seen:
            raise ValueError("run split isolation is invalid")
        seen.add(run_id)
        start = int(row.get("row_start", row.get("start_row", row.get("fast_row_start", -1))))
        end = int(row.get("row_stop", row.get("row_end", row.get("end_row", row.get("fast_row_end", -1)))))
        if start < 0 or end <= start:
            raise ValueError(f"invalid row range for {run_id}: {start}:{end}")
        result[split].append(RunRange(run_id, split, start, end))
    for split, runs in result.items():
        runs.sort(key=lambda row: row.start)
        if len(runs) != 12 or runs[0].start != 0:
            raise ValueError(f"expected 12 complete runs in {split}")
        if any(left.end != right.start for left, right in zip(runs, runs[1:])):
            raise ValueError(f"non-contiguous run ranges in {split}")
    return result


def _load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(split)
    return {
        name: np.load(root / split / f"{name}.npy", mmap_mode="r")
        for name in ("X", "X_next", "D", "U")
    }


def _fit_scalar_normalizer(values: np.ndarray, names: list[str]) -> dict[str, Any]:
    value = np.asarray(values, dtype=np.float64)
    mean = value.mean(axis=0)
    scale = value.std(axis=0)
    active = scale > np.maximum(1e-12, np.abs(mean) * 1e-10)
    scale = np.where(active, scale, 1.0)
    return {
        "schema_version": "servingrom.scalar-normalizer.v1",
        "fit_split": "train",
        "names": names,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "active": active.tolist(),
    }


def _scalar_transform(values: np.ndarray, normalizer: dict[str, Any]) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - np.asarray(normalizer["mean"])) / np.asarray(normalizer["scale"])


def _project(array: np.ndarray, normalizer: Normalizer, basis: np.ndarray, chunk: int) -> np.ndarray:
    output = np.empty((array.shape[0], basis.shape[1]), dtype=np.float64)
    for start in range(0, array.shape[0], chunk):
        end = min(start + chunk, array.shape[0])
        output[start:end] = normalizer.transform(array[start:end]) @ basis
    return output


def _nrmse(actual: np.ndarray, predicted: np.ndarray, reference_mean: np.ndarray | None = None) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    center = actual.mean(axis=0) if reference_mean is None else reference_mean
    numerator = np.square(predicted - actual).sum()
    denominator = np.square(actual - center).sum()
    return float(np.sqrt(numerator / max(float(denominator), 1e-30)))


def _transition_rows(runs: list[RunRange]) -> np.ndarray:
    return np.concatenate([np.arange(run.start + 1, run.end, dtype=np.int64) for run in runs])


def _design(
    z: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    rows: np.ndarray,
    *,
    bilinear: bool,
) -> np.ndarray:
    z_now = z[rows]
    parts = [z_now, z_now - z[rows - 1], d[rows], d[rows - 1], u[rows]]
    if bilinear:
        parts.append(z_now * u[rows, :1])
    parts.append(np.ones((len(rows), 1), dtype=np.float64))
    return np.concatenate(parts, axis=1)


def _fit_dynamic(
    z: np.ndarray,
    z_next: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
    runs: list[RunRange],
    ridge: float,
    *,
    bilinear: bool,
) -> DynamicModel:
    rows = _transition_rows(runs)
    design = _design(z, d, u, rows, bilinear=bilinear)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[-1, -1] = 0.0
    theta = np.linalg.solve(gram + penalty, design.T @ z_next[rows]).T
    rank = z.shape[1]
    d_dim = d.shape[1]
    cursor = 0
    A = theta[:, cursor:cursor + rank]; cursor += rank
    L = theta[:, cursor:cursor + rank]; cursor += rank
    E = theta[:, cursor:cursor + d_dim]; cursor += d_dim
    M = theta[:, cursor:cursor + d_dim]; cursor += d_dim
    B = theta[:, cursor:cursor + 1]; cursor += 1
    N = theta[:, cursor:cursor + rank] if bilinear else None
    if bilinear:
        cursor += rank
    return DynamicModel(rank, ridge, "bilinear" if bilinear else "linear", A, L, E, M, B, theta[:, cursor], N)


def _one_step(model: DynamicModel, z: np.ndarray, target: np.ndarray, d: np.ndarray, u: np.ndarray, runs: list[RunRange]) -> float:
    rows = _transition_rows(runs)
    predicted = model.predict(z[rows], z[rows - 1], d[rows], d[rows - 1], u[rows])
    return _nrmse(target[rows], predicted, np.zeros(target.shape[1]))


def _rollout(model: DynamicModel, z: np.ndarray, d: np.ndarray, u: np.ndarray, runs: list[RunRange]) -> dict[str, Any]:
    squared = baseline_squared = denominator = 0.0
    run_rows = []
    for run in runs:
        actual = z[run.start:run.end]
        predicted = np.empty_like(actual)
        predicted[0] = actual[0]
        previous = actual[0].copy()
        for offset in range(len(actual) - 1):
            current = predicted[offset]
            index = run.start + offset
            d_previous = d[index - 1] if offset else d[index]
            predicted[offset + 1] = model.predict(
                current[None, :], previous[None, :], d[index:index + 1],
                d_previous[None, :], u[index:index + 1],
            )[0]
            previous = current
            if not np.isfinite(predicted[offset + 1]).all() or np.linalg.norm(predicted[offset + 1]) > 1e12:
                predicted[offset + 1:] = np.nan
                break
        finite = bool(np.isfinite(predicted).all())
        if finite:
            error = float(np.square(predicted - actual).sum())
            base = float(np.square(actual[0] - actual).sum())
            denom = float(np.square(actual).sum())
            squared += error; baseline_squared += base; denominator += denom
            nrmse = math.sqrt(error / max(denom, 1e-30))
        else:
            error = math.inf; nrmse = math.inf
        run_rows.append({"run_id": run.run_id, "finite": finite, "nrmse": nrmse})
    value = math.sqrt(squared / max(denominator, 1e-30)) if all(row["finite"] for row in run_rows) else math.inf
    baseline = math.sqrt(baseline_squared / max(denominator, 1e-30))
    return {
        "state_nrmse": value,
        "persistence_nrmse": baseline,
        "skill": 1.0 - value / max(baseline, 1e-30),
        "runs": run_rows,
    }


def _imbalance_reconstruction(
    array: np.ndarray,
    normalizer: Normalizer,
    basis: np.ndarray,
    feature_index: dict[str, int],
    train_means: dict[str, float],
    chunk: int,
) -> dict[str, float]:
    residual = {name: 0.0 for name in IMBALANCE_FEATURES}
    denominator = {name: 0.0 for name in IMBALANCE_FEATURES}
    for start in range(0, array.shape[0], chunk):
        end = min(start + chunk, array.shape[0])
        raw = np.asarray(array[start:end], dtype=np.float64)
        normalized = normalizer.transform(raw)
        reconstructed = normalizer.inverse((normalized @ basis) @ basis.T)
        for name, (left, right) in IMBALANCE_FEATURES.items():
            actual = raw[:, feature_index[left]] - raw[:, feature_index[right]]
            predicted = reconstructed[:, feature_index[left]] - reconstructed[:, feature_index[right]]
            residual[name] += float(np.square(predicted - actual).sum())
            denominator[name] += float(np.square(actual - train_means[name]).sum())
    return {name: math.sqrt(residual[name] / max(denominator[name], 1e-30)) for name in residual}


def _control_direction(
    model: DynamicModel,
    z: np.ndarray,
    d: np.ndarray,
    u_normalizer: dict[str, Any],
    normalizer: Normalizer,
    basis: np.ndarray,
    feature_index: dict[str, int],
) -> dict[str, Any]:
    rows = np.arange(1, len(z), 20, dtype=np.int64)
    low = np.full((len(rows), 1), 0.4)
    high = np.full((len(rows), 1), 0.6)
    low = _scalar_transform(low, u_normalizer)
    high = _scalar_transform(high, u_normalizer)
    common = (z[rows], z[rows - 1], d[rows], d[rows - 1])
    low_state = normalizer.inverse(model.predict(*common, low) @ basis.T)
    high_state = normalizer.inverse(model.predict(*common, high) @ basis.T)
    result = {}
    passes = 0
    for name, (left, right) in IMBALANCE_FEATURES.items():
        delta = (
            high_state[:, feature_index[left]] - high_state[:, feature_index[right]]
            - low_state[:, feature_index[left]] + low_state[:, feature_index[right]]
        )
        fraction = float(np.mean(delta > 0.0))
        result[name] = {"positive_fraction": fraction, "median_high_minus_low": float(np.median(delta))}
        passes += fraction > 0.5
    result["direction_pass_fraction"] = passes / len(IMBALANCE_FEATURES)
    return result


def _slow_value(row: dict[str, Any], name: str) -> float:
    if name == "ttft_total_ms":
        return float(row.get("ttft_mean_ms") or 0.0) * float(row.get("completed_requests") or 0.0)
    if name == "tpot_total_ms":
        return float(row.get("tpot_mean_ms") or 0.0) * float(row.get("completed_requests") or 0.0)
    return float(row.get(name) or 0.0)


def _build_slow(
    slow_rows: list[dict[str, Any]],
    runs: list[RunRange],
    z: np.ndarray,
    d: np.ndarray,
    u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in slow_rows:
        by_run.setdefault(str(row["run_id"]), []).append(row)
    zs, ds, us, ys = [], [], [], []
    for run in runs:
        rows = sorted(by_run.get(run.run_id, []), key=lambda value: int(value.get("slow_window_id", 0)))
        if len(rows) != 120 or run.end - run.start != 3000:
            raise ValueError(f"slow/fast alignment failed for {run.run_id}")
        for offset, row in enumerate(rows):
            start = run.start + offset * 25
            end = start + 25
            zs.append(z[start:end].mean(axis=0))
            ds.append(d[start:end].mean(axis=0))
            us.append(u[start:end].mean(axis=0))
            ys.append([_slow_value(row, name) for name in SLOW_OUTPUTS])
    return tuple(np.asarray(value, dtype=np.float64) for value in (zs, ds, us, ys))  # type: ignore[return-value]


def _fit_slow_head(z: np.ndarray, d: np.ndarray, u: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    design = np.concatenate((z, d, u, np.ones((len(z), 1))), axis=1)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _predict_slow(theta: np.ndarray, z: np.ndarray, d: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.concatenate((z, d, u, np.ones((len(z), 1))), axis=1) @ theta


def _save_model(path: Path, model: DynamicModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, A=model.A, L=model.L, E=model.E, M=model.M, B=model.B,
        c=model.c, N=np.asarray([]) if model.N is None else model.N,
        rank=model.rank, ridge=model.ridge, model_type=model.model_type,
    )


def _copy_configured_report(output: Path, result: dict[str, Any]) -> None:
    selected = result["selection"]
    metrics = result["metrics"]
    direction = result["control_direction"]
    slow = result["slow_head"]
    readiness = result["readiness"]
    lines = [
        "# ServingROM Step 15 Control-aware ROM Identification",
        "",
        "## 结论",
        "",
        f"- `control_rom_ready={str(readiness['control_rom_ready']).lower()}`",
        f"- 最终模型：`{selected['model_type']}`，POD rank={selected['rank']}，ridge={selected['ridge']}",
        f"- 增广谱半径：`{selected['spectral_radius']:.6f}`",
        "- 只使用冻结的 `servingrom-control-dataset-v1`；Round 14.3 held-out 数据未读取。",
        "- train/validation 用于拟合和选择；test 仅在模型冻结后执行一次最终评估。",
        f"- rank=16 控制状态重构充分性：`{result['pod']['rank16_control_imbalance_sufficient']}`。",
        "",
        "## Fast State 指标",
        "",
        "| split | one-step NRMSE | rollout NRMSE | persistence NRMSE | rollout skill |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        row = metrics[split]
        lines.append(
            f"| {split} | {row['one_step_nrmse']:.6f} | {row['rollout']['state_nrmse']:.6f} | "
            f"{row['rollout']['persistence_nrmse']:.6f} | {row['rollout']['skill']:.6f} |"
        )
    lines += [
        "",
        "## POD 控制状态重构",
        "",
        "validation NRMSE 小于 1 才表示重构优于 train-mean 基线；该门只使用 validation，不参与 test 后调参。",
        "",
        "| state | rank 16 validation NRMSE | selected-rank validation NRMSE |",
        "|---|---:|---:|",
    ]
    for name in IMBALANCE_FEATURES:
        lines.append(
            f"| {name} | {result['pod']['rank16_validation_control_imbalance'][name]:.6f} | "
            f"{result['pod']['selected_rank_validation_control_imbalance'][name]:.6f} |"
        )
    lines += [
        "",
        "## 控制方向",
        "",
        "以下反事实比较保持状态和扰动不变，只将 `rho_A` 从 0.4 提高到 0.6；正号表示模型预测 A-B 状态随控制方向正确增加。",
        "",
        "| state | positive fraction | median delta |",
        "|---|---:|---:|",
    ]
    for name in IMBALANCE_FEATURES:
        row = direction[name]
        lines.append(f"| {name} | {row['positive_fraction']:.6f} | {row['median_high_minus_low']:.6f} |")
    lines += [
        "",
        "## Slow KPI Head",
        "",
        f"- Slow Head ridge：`{slow['ridge']}`",
        f"- validation aggregate NRMSE：`{slow['validation_nrmse']:.6f}`",
        f"- test aggregate NRMSE：`{slow['test_nrmse']:.6f}`",
        "- TTFT/TPOT 使用 5 秒窗口内守恒总量（mean × completed requests）建模，避免无 completion 窗口的 null 均值被伪造成观测值。",
        "",
        "## Bilinear 候选",
        "",
        f"- validation rollout 相对改善：`{selected['bilinear_validation_relative_improvement']:.6f}`",
        f"- 是否保留：`{selected['model_type'] == 'bilinear'}`",
        "",
        "## Readiness Gates",
        "",
    ]
    for name, value in readiness["gates"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines += [
        "",
        "## 边界",
        "",
        "本轮未读取 held-out actuator benchmark，未重新调参，未实现 actuator 或 MPC。若 readiness 为 false，应停在 Step 15 分析 POD 对控制状态的表达能力或动力学输出头的结构性缺口。",
    ]
    (output / "STEP15_CONTROL_ROM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_control_rom_pipeline(dataset_root: Path, output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    audit = _verify_dataset(dataset_root, config["dataset_id"])
    runs = _load_runs(dataset_root)
    state_index = _load_json(dataset_root / "state_index.json")
    disturbance_index = _load_json(dataset_root / "disturbance_index.json")
    control_index = _load_json(dataset_root / "control_index.json")
    if [row["name"] for row in control_index] != ["rho_A"]:
        raise ValueError("U is not the frozen rho_A actuator")
    feature_index = {row["name"]: int(row["index"]) for row in state_index}
    chunk = int(config["chunk_size"])

    # Selection phase: test files are deliberately unopened until final_model.json is written.
    train = _load_split(dataset_root, "train")
    validation = _load_split(dataset_root, "validation")
    x_normalizer, x_audit = fit_normalizer(train["X"], state_index, chunk)
    d_normalizer, d_audit = fit_normalizer(train["D"], disturbance_index, chunk)
    u_normalizer = _fit_scalar_normalizer(train["U"], ["rho_A"])
    save_json(output_root / "preprocessing/x_normalizer.json", x_normalizer.to_json())
    save_json(output_root / "preprocessing/d_normalizer.json", d_normalizer.to_json())
    save_json(output_root / "preprocessing/u_normalizer.json", u_normalizer)
    save_json(output_root / "preprocessing/train_audit.json", {"X": x_audit, "D": d_audit})

    ranks = [int(value) for value in config["candidate_ranks"]]
    pod = fit_pod(train["X"], x_normalizer, max(ranks), chunk)
    basis_max = pod["basis"]
    np.save(output_root / "pod_basis_candidates.npy", basis_max)
    np.save(output_root / "pod_eigenvalues.npy", pod["eigenvalues"])
    spectrum = {
        "singular_values": pod["singular_values"].tolist(),
        "cumulative_energy": (np.cumsum(pod["eigenvalues"]) / max(float(pod["eigenvalues"].sum()), 1e-30)).tolist(),
        "mode_block_contributions": mode_block_contributions(basis_max, x_normalizer, modes=max(ranks)),
        "reconstruction_nrmse": {
            "train": reconstruction_scan(train["X"], x_normalizer, basis_max, ranks, chunk),
            "validation": reconstruction_scan(validation["X"], x_normalizer, basis_max, ranks, chunk),
        },
    }
    save_json(output_root / "pod_spectrum.json", spectrum)

    train_imbalance_means = {}
    for name, (left, right) in IMBALANCE_FEATURES.items():
        train_imbalance_means[name] = float(np.mean(train["X"][:, feature_index[left]] - train["X"][:, feature_index[right]]))
    imbalance_reconstruction = {"train": {}, "validation": {}}
    for rank in ranks:
        basis = basis_max[:, :rank]
        imbalance_reconstruction["train"][rank] = _imbalance_reconstruction(
            train["X"], x_normalizer, basis, feature_index, train_imbalance_means, chunk,
        )
        imbalance_reconstruction["validation"][rank] = _imbalance_reconstruction(
            validation["X"], x_normalizer, basis, feature_index, train_imbalance_means, chunk,
        )
    save_json(output_root / "pod_control_imbalance_reconstruction.json", imbalance_reconstruction)

    d_train = d_normalizer.transform(train["D"], weighted=False)
    d_validation = d_normalizer.transform(validation["D"], weighted=False)
    u_train = _scalar_transform(train["U"], u_normalizer)
    u_validation = _scalar_transform(validation["U"], u_normalizer)
    z_train_max = _project(train["X"], x_normalizer, basis_max, chunk)
    z_train_next_max = _project(train["X_next"], x_normalizer, basis_max, chunk)
    z_validation_max = _project(validation["X"], x_normalizer, basis_max, chunk)
    z_validation_next_max = _project(validation["X_next"], x_normalizer, basis_max, chunk)

    candidates = []
    rank_winners: list[tuple[dict[str, Any], DynamicModel]] = []
    maximum_radius = float(config["maximum_spectral_radius"])
    for rank in ranks:
        best: tuple[dict[str, Any], DynamicModel] | None = None
        for ridge in config["candidate_ridges"]:
            model = _fit_dynamic(
                z_train_max[:, :rank], z_train_next_max[:, :rank], d_train, u_train,
                runs["train"], float(ridge), bilinear=False,
            )
            radius = model.spectral_radius()
            row = {
                "rank": rank, "ridge": float(ridge), "model_type": "linear",
                "spectral_radius": radius,
                "train_one_step_nrmse": _one_step(model, z_train_max[:, :rank], z_train_next_max[:, :rank], d_train, u_train, runs["train"]),
                "validation_one_step_nrmse": _one_step(model, z_validation_max[:, :rank], z_validation_next_max[:, :rank], d_validation, u_validation, runs["validation"]),
                "stable": radius <= maximum_radius,
            }
            candidates.append(row)
            if row["stable"] and (best is None or row["validation_one_step_nrmse"] < best[0]["validation_one_step_nrmse"]):
                best = (row, model)
        if best is None:
            continue
        rollout = _rollout(best[1], z_validation_max[:, :rank], d_validation, u_validation, runs["validation"])
        best[0]["validation_rollout"] = rollout
        rank_winners.append(best)
    if not rank_winners:
        raise RuntimeError("no stable linear candidate")
    linear_row, linear_model = min(rank_winners, key=lambda value: value[0]["validation_rollout"]["state_nrmse"])

    # Compare one minimal bilinear family after the linear rank is frozen by validation.
    rank = linear_model.rank
    bilinear_winners = []
    u_nominal = float(_scalar_transform(np.asarray([[0.5]]), u_normalizer)[0, 0])
    for ridge in config["candidate_ridges"]:
        model = _fit_dynamic(
            z_train_max[:, :rank], z_train_next_max[:, :rank], d_train, u_train,
            runs["train"], float(ridge), bilinear=True,
        )
        radius = model.spectral_radius(u_nominal)
        rollout = _rollout(model, z_validation_max[:, :rank], d_validation, u_validation, runs["validation"])
        row = {
            "rank": rank, "ridge": float(ridge), "model_type": "bilinear",
            "spectral_radius": radius,
            "validation_one_step_nrmse": _one_step(model, z_validation_max[:, :rank], z_validation_next_max[:, :rank], d_validation, u_validation, runs["validation"]),
            "validation_rollout": rollout,
            "stable": radius <= maximum_radius,
        }
        candidates.append(row)
        if row["stable"]:
            bilinear_winners.append((row, model))
    bilinear_row, bilinear_model = min(
        bilinear_winners,
        key=lambda value: value[0]["validation_rollout"]["state_nrmse"],
        default=({"validation_rollout": {"state_nrmse": math.inf}, "spectral_radius": math.inf}, None),
    )
    linear_val = linear_row["validation_rollout"]["state_nrmse"]
    bilinear_val = bilinear_row["validation_rollout"]["state_nrmse"]
    improvement = (linear_val - bilinear_val) / max(linear_val, 1e-30)
    retain_bilinear = bool(
        bilinear_model is not None
        and improvement >= float(config["bilinear_min_validation_improvement"])
        and bilinear_row["spectral_radius"] <= linear_row["spectral_radius"] + float(config["maximum_bilinear_radius_increase"])
    )
    final_model = bilinear_model if retain_bilinear else linear_model
    final_row = bilinear_row if retain_bilinear else linear_row
    selection = {
        "selection_split": "validation",
        "test_accessed": False,
        "rank": final_model.rank,
        "ridge": final_model.ridge,
        "model_type": final_model.model_type,
        "spectral_radius": final_model.spectral_radius(u_nominal if final_model.N is not None else 0.0),
        "linear_validation_rollout_nrmse": linear_val,
        "bilinear_validation_rollout_nrmse": bilinear_val,
        "bilinear_validation_relative_improvement": improvement,
        "bilinear_retained": retain_bilinear,
    }
    save_json(output_root / "evaluation/candidates.json", candidates)
    save_json(output_root / "evaluation/frozen_selection_before_test.json", selection)
    basis = basis_max[:, :final_model.rank]
    np.save(output_root / "pod_basis.npy", basis)
    _save_model(output_root / "models/final_fast_model.npz", final_model)

    # Final evaluation phase: test is opened only after all choices are durable on disk.
    test = _load_split(dataset_root, "test")
    d_test = d_normalizer.transform(test["D"], weighted=False)
    u_test = _scalar_transform(test["U"], u_normalizer)
    z_test = _project(test["X"], x_normalizer, basis, chunk)
    z_test_next = _project(test["X_next"], x_normalizer, basis, chunk)
    z_train = z_train_max[:, :final_model.rank]
    z_train_next = z_train_next_max[:, :final_model.rank]
    z_validation = z_validation_max[:, :final_model.rank]
    z_validation_next = z_validation_next_max[:, :final_model.rank]
    metrics = {}
    for split, z, z_next, d, u in (
        ("train", z_train, z_train_next, d_train, u_train),
        ("validation", z_validation, z_validation_next, d_validation, u_validation),
        ("test", z_test, z_test_next, d_test, u_test),
    ):
        metrics[split] = {
            "one_step_nrmse": _one_step(final_model, z, z_next, d, u, runs[split]),
            "rollout": _rollout(final_model, z, d, u, runs[split]),
        }
    control_direction = _control_direction(
        final_model, z_validation, d_validation, u_normalizer,
        x_normalizer, basis, feature_index,
    )
    test_reconstruction = {
        "weighted_state_nrmse": reconstruction_scan(test["X"], x_normalizer, basis, [final_model.rank], chunk)[final_model.rank],
        "control_imbalance_nrmse": _imbalance_reconstruction(
            test["X"], x_normalizer, basis, feature_index, train_imbalance_means, chunk,
        ),
    }

    slow_rows = pq.read_table(dataset_root / "slow_kpi_windows.parquet").to_pylist()
    slow = {}
    for split, z, d, u in (
        ("train", z_train, d_train, u_train),
        ("validation", z_validation, d_validation, u_validation),
        ("test", z_test, d_test, u_test),
    ):
        split_rows = [row for row in slow_rows if str(row["split"]) == split]
        slow[split] = _build_slow(split_rows, runs[split], z, d, u)
    y_normalizer = _fit_scalar_normalizer(slow["train"][3], SLOW_OUTPUTS)
    save_json(output_root / "preprocessing/slow_y_normalizer.json", y_normalizer)
    slow_y = {split: _scalar_transform(values[3], y_normalizer) for split, values in slow.items()}
    slow_candidates = []
    best_slow = None
    for ridge in config["candidate_ridges"]:
        theta = _fit_slow_head(*slow["train"][:3], slow_y["train"], float(ridge))
        prediction = _predict_slow(theta, *slow["validation"][:3])
        score = _nrmse(slow_y["validation"], prediction, np.zeros(len(SLOW_OUTPUTS)))
        row = {"ridge": float(ridge), "validation_nrmse": score}
        slow_candidates.append(row)
        if best_slow is None or score < best_slow[0]["validation_nrmse"]:
            best_slow = (row, theta)
    assert best_slow is not None
    slow_row, slow_theta = best_slow
    slow_metrics = {"ridge": slow_row["ridge"], "outputs": SLOW_OUTPUTS, "candidates": slow_candidates}
    for split in ("train", "validation", "test"):
        prediction = _predict_slow(slow_theta, *slow[split][:3])
        slow_metrics[f"{split}_nrmse"] = _nrmse(slow_y[split], prediction, np.zeros(len(SLOW_OUTPUTS)))
        slow_metrics[f"{split}_per_output_nrmse"] = {
            name: _nrmse(slow_y[split][:, index:index + 1], prediction[:, index:index + 1], np.zeros(1))
            for index, name in enumerate(SLOW_OUTPUTS)
        }
    np.savez_compressed(output_root / "models/slow_kpi_head.npz", theta=slow_theta, ridge=slow_row["ridge"], outputs=np.asarray(SLOW_OUTPUTS))

    gates_config = config["readiness_gates"]
    imbalance_limit = float(gates_config["maximum_validation_control_imbalance_nrmse"])
    selected_imbalance = imbalance_reconstruction["validation"][final_model.rank]
    rank16_imbalance = imbalance_reconstruction["validation"][16]
    gates = {
        "spectral_radius": selection["spectral_radius"] <= maximum_radius,
        "validation_rollout_skill_positive": metrics["validation"]["rollout"]["skill"] > 0.0,
        "test_rollout_skill_positive": metrics["test"]["rollout"]["skill"] > 0.0,
        "validation_slow_kpi_beats_mean": slow_metrics["validation_nrmse"] <= float(gates_config["maximum_validation_slow_kpi_nrmse"]),
        "test_slow_kpi_beats_mean": slow_metrics["test_nrmse"] <= float(gates_config["maximum_test_slow_kpi_nrmse"]),
        "control_direction": control_direction["direction_pass_fraction"] >= float(gates_config["minimum_control_direction_fraction"]),
        "pod_control_imbalance_reconstruction": all(value < imbalance_limit for value in selected_imbalance.values()),
        "all_metrics_finite": bool(all(math.isfinite(metrics[split]["rollout"]["state_nrmse"]) for split in metrics)),
    }
    readiness = {"control_rom_ready": all(gates.values()), "gates": gates}
    selection["test_accessed"] = True
    result = {
        "schema_version": "servingrom.control-rom.result.v1",
        "dataset": audit,
        "data_isolation": {
            "selection_splits": ["train", "validation"],
            "test_access_phase": "after_frozen_selection_before_test.json",
            "heldout_actuator_data_read": False,
            "mpc_implemented": False,
        },
        "selection": selection,
        "pod": {
            "test_reconstruction": test_reconstruction,
            "rank16_validation_control_imbalance": rank16_imbalance,
            "rank16_control_imbalance_sufficient": all(value < imbalance_limit for value in rank16_imbalance.values()),
            "selected_rank_validation_control_imbalance": selected_imbalance,
        },
        "metrics": metrics,
        "control_direction": control_direction,
        "slow_head": slow_metrics,
        "readiness": readiness,
    }
    save_json(output_root / "evaluation/final_metrics.json", result)
    manifest = {
        "model_id": config["model_id"],
        "dataset_id": config["dataset_id"],
        "dataset_sha256_manifest": audit["sha256_manifest"],
        "train_runs": [run.run_id for run in runs["train"]],
        "validation_runs": [run.run_id for run in runs["validation"]],
        "test_runs": [run.run_id for run in runs["test"]],
        "selection": selection,
        "readiness": readiness,
        "artifacts": {},
    }
    _copy_configured_report(output_root, result)
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"MODEL_MANIFEST.json", "step15.log", "step15.pid"}:
            manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    save_json(output_root / "MODEL_MANIFEST.json", manifest)
    return result
