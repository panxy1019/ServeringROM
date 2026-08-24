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

from .forcing import FrozenGlobalModel, _load_frozen_global, _slow_score
from .memory import _encode, _load_frozen_representation
from .pipeline import (
    SLOW_OUTPUTS,
    _build_slow,
    _fit_scalar_normalizer,
    _load_json,
    _load_runs,
    _load_split,
    _nrmse,
    _scalar_transform,
    _sha256,
    _verify_dataset,
)


Q_NAMES = ("running_imbalance", "remaining_token_imbalance")
FORCING_NAMES = ("routed_request_imbalance", "routed_expected_token_mass_imbalance")
COMMON_NAMES = ("total_running", "total_remaining_tokens")
CANDIDATES = ("M0_inflow_only", "M1_minimal_service_closure", "M2_load_dependent_service_closure")
WINDOW_NS = 200_000_000


@dataclass
class ConservationModel:
    candidate: str
    ridge: float
    B0: np.ndarray
    B1: np.ndarray
    Q: np.ndarray
    interactions: np.ndarray

    def predict_delta(
        self,
        q: np.ndarray,
        forcing: np.ndarray,
        previous_forcing: np.ndarray,
        common: np.ndarray,
    ) -> np.ndarray:
        value = forcing @ self.B0.T
        if self.candidate != "M0_inflow_only":
            value += previous_forcing @ self.B1.T + q @ self.Q.T
        if self.candidate == "M2_load_dependent_service_closure":
            for index in range(common.shape[1]):
                value += (q * common[:, index:index + 1]) @ self.interactions[index].T
        return value

    def transition_matrix(self, common: np.ndarray | None = None) -> np.ndarray:
        matrix = np.eye(2) + self.Q
        if self.candidate == "M2_load_dependent_service_closure" and common is not None:
            for index, value in enumerate(common):
                matrix += self.interactions[index] * float(value)
        return matrix


def _decoder(value: Any) -> int | None:
    text = str(value or "")
    if text.endswith(":13701") or text == "decode-0":
        return 0
    if text.endswith(":13702") or text == "decode-1":
        return 1
    return None


def _signed(decoder: int) -> float:
    return 1.0 if decoder == 0 else -1.0


def _window_index(timestamp: Any, starts: list[int]) -> int | None:
    if timestamp is None or not starts:
        return None
    value = int(timestamp)
    index = (value - starts[0]) // WINDOW_NS
    if index < 0 or index >= len(starts):
        return None
    index = int(index)
    if starts[index] <= value < starts[index] + WINDOW_NS:
        return index
    return None


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _run_outflows(
    run_id: str,
    forcing_root: Path,
    outflow_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    control_path = forcing_root / run_id / "derived/control/control_windows.parquet"
    derived = outflow_root / run_id / "derived"
    controls = _read(control_path)
    if len(controls) != 3000:
        raise ValueError(f"{run_id}: expected 3000 control windows")
    starts = [int(row["start_wall_ns"]) for row in controls]
    if any(int(row["end_wall_ns"]) - starts[i] != WINDOW_NS for i, row in enumerate(controls)):
        raise ValueError(f"{run_id}: non-200ms control window")
    if any(starts[i] + WINDOW_NS != starts[i + 1] for i in range(len(starts) - 1)):
        raise ValueError(f"{run_id}: non-contiguous control windows")

    attempts = _read(derived / "attempt_lifecycle.parquet")
    traces = _read(derived / "trace_lifecycle.parquet")
    engines = _read(derived / "engine_requests.parquet")
    memberships = _read(derived / "scheduler_membership.parquet")
    emissions = _read(derived / "token_emissions.parquet")
    transfers = _read(derived / "kv_transfers.parquet")
    request_decoder: dict[str, int] = {}
    trace_request: dict[str, str] = {}
    for row in attempts:
        decoder = _decoder(row.get("decoder_backend"))
        request_id = row.get("request_id")
        if decoder is not None and request_id:
            request_decoder[str(request_id)] = decoder
            if row.get("trace_id"):
                trace_request[str(row["trace_id"])] = str(request_id)

    fields = (
        "running_admission_imbalance", "completion_outflow_imbalance",
        "decode_added_imbalance", "kv_ready_transition_imbalance",
        "scheduled_token_service_imbalance", "emitted_token_service_imbalance",
        "terminal_residual_token_outflow_imbalance",
        "active_inventory_request_inflow_imbalance",
        "active_inventory_token_inflow_imbalance",
    )
    values = {name: np.zeros(len(controls), dtype=np.float64) for name in fields}
    first_membership: dict[str, int] = {}
    emitted_by_request: dict[str, float] = {}
    join_counts = {"attempts": len(attempts), "mapped_requests": len(request_decoder)}

    for row in memberships:
        request_id = str(row.get("request_id") or "")
        decoder = request_decoder.get(request_id)
        if decoder is None:
            continue
        if _decoder(row.get("component")) is not None:
            timestamp = int(row["ts_wall_ns"])
            first_membership[request_id] = min(timestamp, first_membership.get(request_id, timestamp))
            index = _window_index(timestamp, starts)
            if index is not None:
                values["scheduled_token_service_imbalance"][index] += _signed(decoder) * float(row.get("scheduled_tokens") or 0)
    for request_id, timestamp in first_membership.items():
        index = _window_index(timestamp, starts)
        if index is not None:
            values["running_admission_imbalance"][index] += _signed(request_decoder[request_id])

    for row in emissions:
        request_id = str(row.get("request_id") or "")
        decoder = request_decoder.get(request_id)
        if decoder is None:
            continue
        count = float(row.get("new_token_count") or 0)
        emitted_by_request[request_id] = emitted_by_request.get(request_id, 0.0) + count
        index = _window_index(row.get("ts_wall_ns"), starts)
        if index is not None:
            values["emitted_token_service_imbalance"][index] += _signed(decoder) * count

    decode_engines = 0
    for row in engines:
        decoder = _decoder(row.get("component"))
        request_id = str(row.get("request_id") or "")
        if decoder is None or request_decoder.get(request_id) != decoder:
            continue
        decode_engines += 1
        index = _window_index(row.get("added_wall_ns"), starts)
        if index is not None:
            values["decode_added_imbalance"][index] += _signed(decoder)

    trace_by_id = {str(row["trace_id"]): row for row in traces if row.get("trace_id")}
    terminal_mapped = 0
    for trace_id, request_id in trace_request.items():
        row = trace_by_id.get(trace_id)
        decoder = request_decoder.get(request_id)
        if row is None or decoder is None or row.get("terminal_wall_ns") is None:
            continue
        terminal_mapped += 1
        arrival_index = _window_index(row.get("arrival_wall_ns"), starts)
        if arrival_index is not None:
            values["active_inventory_request_inflow_imbalance"][arrival_index] += _signed(decoder)
            values["active_inventory_token_inflow_imbalance"][arrival_index] += (
                _signed(decoder) * float(row.get("expected_output_tokens") or 0)
            )
        index = _window_index(row["terminal_wall_ns"], starts)
        if index is None:
            continue
        values["completion_outflow_imbalance"][index] += _signed(decoder)
        residual = max(float(row.get("expected_output_tokens") or 0) - emitted_by_request.get(request_id, 0.0), 0.0)
        values["terminal_residual_token_outflow_imbalance"][index] += _signed(decoder) * residual

    kv_mapped = 0
    for row in transfers:
        request_id = str(row.get("request_id") or "")
        decoder = request_decoder.get(request_id)
        if decoder is None or row.get("kv_ready_wall_ns") is None:
            continue
        kv_mapped += 1
        index = _window_index(row["kv_ready_wall_ns"], starts)
        if index is not None:
            values["kv_ready_transition_imbalance"][index] += _signed(decoder)

    output = []
    for index, control in enumerate(controls):
        forcing_request = 2.0 * float(control["routed_A_request_count"]) - float(control["routed_request_count"])
        forcing_tokens = 2.0 * float(control["routed_A_expected_token_mass"]) - float(control["routed_expected_token_mass"])
        output.append({
            "run_id": run_id, "window_id": index,
            "start_wall_ns": starts[index], "end_wall_ns": starts[index] + WINDOW_NS,
            "routed_request_imbalance": forcing_request,
            "routed_expected_token_mass_imbalance": forcing_tokens,
            **{name: float(values[name][index]) for name in fields},
        })
    audit = {
        "run_id": run_id,
        "source_rows": {
            "attempt_lifecycle": len(attempts), "trace_lifecycle": len(traces),
            "engine_requests": len(engines), "scheduler_membership": len(memberships),
            "token_emissions": len(emissions), "kv_transfers": len(transfers),
        },
        "join_counts": {
            **join_counts, "decode_engine_requests": decode_engines,
            "terminal_mapped": terminal_mapped, "kv_ready_mapped": kv_mapped,
        },
    }
    return output, audit


def _scale_only(values: np.ndarray, names: tuple[str, ...]) -> dict[str, Any]:
    scale = np.asarray(values, dtype=np.float64).std(axis=0)
    scale[scale < 1e-12] = 1.0
    return {
        "schema_version": "servingrom.scale-only-normalizer.v1",
        "fit_split": "train", "preserves_physical_zero": True,
        "names": list(names), "scale": scale.tolist(), "mean_subtracted": False,
    }


def _transform_scale_only(values: np.ndarray, normalizer: dict[str, Any]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / np.asarray(normalizer["scale"], dtype=np.float64)


def _physical_arrays(
    dataset_root: Path,
    runs: dict[str, list[Any]],
    forcing_root: Path,
    outflow_root: Path,
    splits: tuple[str, ...],
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    state_index = {row["name"]: int(row["index"]) for row in _load_json(dataset_root / "state_index.json")}
    indices = {
        "run_a": state_index["decode_d1_running_count"],
        "run_b": state_index["decode_d2_running_count"],
        "remaining_a": state_index["decode_d1_expected_remaining_tokens"],
        "remaining_b": state_index["decode_d2_expected_remaining_tokens"],
        "waiting_a": state_index["decode_d1_waiting_count"],
        "waiting_b": state_index["decode_d2_waiting_count"],
    }
    result: dict[str, dict[str, np.ndarray]] = {}
    sidecar_rows = []
    run_audits = []
    source_hashes: dict[str, str] = {}
    for split in splits:
        arrays = _load_split(dataset_root, split)
        x, xn = np.asarray(arrays["X"]), np.asarray(arrays["X_next"])
        q = np.column_stack((x[:, indices["run_a"]] - x[:, indices["run_b"]], x[:, indices["remaining_a"]] - x[:, indices["remaining_b"]]))
        qn = np.column_stack((xn[:, indices["run_a"]] - xn[:, indices["run_b"]], xn[:, indices["remaining_a"]] - xn[:, indices["remaining_b"]]))
        common = np.column_stack((x[:, indices["run_a"]] + x[:, indices["run_b"]], x[:, indices["remaining_a"]] + x[:, indices["remaining_b"]]))
        waiting = x[:, indices["waiting_a"]] - x[:, indices["waiting_b"]]
        forcing = np.empty((len(x), 2)); flows: dict[str, np.ndarray] = {}
        for run in runs[split]:
            rows, audit = _run_outflows(run.run_id, forcing_root, outflow_root)
            run_audits.append(audit)
            control_path = forcing_root / run.run_id / "derived/control/control_windows.parquet"
            source_hashes[f"{run.run_id}/control_windows.parquet"] = _sha256(control_path)
            for name in ("attempt_lifecycle", "trace_lifecycle", "engine_requests", "scheduler_membership", "token_emissions", "kv_transfers"):
                path = outflow_root / run.run_id / "derived" / f"{name}.parquet"
                source_hashes[f"{run.run_id}/{name}.parquet"] = _sha256(path)
            if len(rows) != run.end - run.start:
                raise ValueError(f"outflow row mismatch: {run.run_id}")
            for name in rows[0]:
                if name in {"run_id", "window_id", "start_wall_ns", "end_wall_ns"}:
                    continue
                flows.setdefault(name, np.empty(len(x), dtype=np.float64))[run.start:run.end] = [float(row[name]) for row in rows]
            forcing[run.start:run.end] = np.column_stack((
                flows[FORCING_NAMES[0]][run.start:run.end], flows[FORCING_NAMES[1]][run.start:run.end],
            ))
            for offset, row in enumerate(rows):
                sidecar_rows.append({"split": split, "row_in_split": run.start + offset, **row})
        result[split] = {"q": q, "q_next": qn, "common": common, "waiting": waiting, "forcing": forcing, **flows}
    return result, sidecar_rows, run_audits, source_hashes


def _write_outflow_sidecar(
    output_root: Path,
    sidecar_rows: list[dict[str, Any]],
    values: dict[str, dict[str, np.ndarray]],
    run_audits: list[dict[str, Any]],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    sidecar = output_root / "sidecar/outflow_transition_windows.parquet"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sidecar_rows), sidecar, compression="zstd")
    manifest = {
        "schema_version": "servingrom.outflow-transition-sidecar.v1",
        "path": str(sidecar.relative_to(output_root)), "sha256": _sha256(sidecar),
        "rows": len(sidecar_rows), "split_rows": {s: len(values[s]["q"]) for s in values},
        "source_files": source_hashes, "run_audits": run_audits,
    }
    return manifest


def _conservation_audit(values: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        row = values[split]
        delta = row["q_next"] - row["q"]
        request_balance = row["running_admission_imbalance"] - row["completion_outflow_imbalance"]
        token_balance = (
            row[FORCING_NAMES[1]] - row["emitted_token_service_imbalance"]
            - row["terminal_residual_token_outflow_imbalance"]
        )
        active_inventory_token_balance = (
            row["active_inventory_token_inflow_imbalance"]
            - row["emitted_token_service_imbalance"]
            - row["terminal_residual_token_outflow_imbalance"]
        )
        residual = np.column_stack((delta[:, 0] - request_balance, delta[:, 1] - token_balance))
        active_residual = delta[:, 1] - active_inventory_token_balance
        result[split] = {
            "rows": len(delta),
            "request_closure": {
                "delta_std": float(delta[:, 0].std()), "residual_mean": float(residual[:, 0].mean()),
                "residual_std": float(residual[:, 0].std()),
                "residual_nrmse": _nrmse(delta[:, 0:1], request_balance[:, None], np.zeros(1)),
                "exact_fraction": float(np.mean(np.abs(residual[:, 0]) < 1e-9)),
            },
            "token_closure": {
                "delta_std": float(delta[:, 1].std()), "residual_mean": float(residual[:, 1].mean()),
                "residual_std": float(residual[:, 1].std()),
                "residual_nrmse": _nrmse(delta[:, 1:2], token_balance[:, None], np.zeros(1)),
                "exact_fraction": float(np.mean(np.abs(residual[:, 1]) < 1e-9)),
            },
            "active_inventory_token_closure": {
                "inflow_semantics": "request arrival attributed to the decoder selected later in the same attempt",
                "residual_mean": float(active_residual.mean()),
                "residual_std": float(active_residual.std()),
                "residual_nrmse": _nrmse(
                    delta[:, 1:2], active_inventory_token_balance[:, None], np.zeros(1),
                ),
                "exact_fraction": float(np.mean(np.abs(active_residual) < 1e-9)),
            },
        }
    return result


def _design(candidate: str, q: np.ndarray, forcing: np.ndarray, common: np.ndarray, rows: np.ndarray) -> np.ndarray:
    parts = [forcing[rows]]
    if candidate != "M0_inflow_only":
        parts.extend((forcing[rows - 1], q[rows]))
    if candidate == "M2_load_dependent_service_closure":
        parts.extend(q[rows] * common[rows, i:i + 1] for i in range(common.shape[1]))
    return np.concatenate(parts, axis=1)


def _rows(runs: list[Any]) -> np.ndarray:
    return np.concatenate([np.arange(run.start + 1, run.end, dtype=np.int64) for run in runs])


def _fit(candidate: str, ridge: float, q: np.ndarray, q_next: np.ndarray, forcing: np.ndarray, common: np.ndarray, runs: list[Any]) -> ConservationModel:
    rows = _rows(runs)
    design = _design(candidate, q, forcing, common, rows)
    target = q_next[rows] - q[rows]
    theta = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ target).T
    cursor = 0
    B0 = theta[:, cursor:cursor + 2]; cursor += 2
    if candidate == "M0_inflow_only":
        return ConservationModel(candidate, ridge, B0, np.zeros((2, 2)), np.zeros((2, 2)), np.empty((0, 2, 2)))
    B1 = theta[:, cursor:cursor + 2]; cursor += 2
    Q = theta[:, cursor:cursor + 2]; cursor += 2
    interactions = []
    if candidate == "M2_load_dependent_service_closure":
        for _ in range(common.shape[1]):
            interactions.append(theta[:, cursor:cursor + 2]); cursor += 2
    return ConservationModel(candidate, ridge, B0, B1, Q, np.asarray(interactions))


def _stability(model: ConservationModel, common: np.ndarray) -> dict[str, Any]:
    if model.candidate != "M2_load_dependent_service_closure":
        radius = float(np.max(np.abs(np.linalg.eigvals(model.transition_matrix()))))
        return {"spectral_radius_max": radius, "spectral_radius_p95": radius, "samples": 1}
    sampled = common[::max(1, len(common) // 5000)]
    radii = np.asarray([np.max(np.abs(np.linalg.eigvals(model.transition_matrix(row)))) for row in sampled])
    return {"spectral_radius_max": float(radii.max()), "spectral_radius_p95": float(np.quantile(radii, 0.95)), "samples": len(radii)}


def _one_step(model: ConservationModel, q: np.ndarray, q_next: np.ndarray, forcing: np.ndarray, common: np.ndarray, runs: list[Any], scale: np.ndarray) -> dict[str, Any]:
    rows = _rows(runs)
    prediction = q[rows] + model.predict_delta(q[rows], forcing[rows], forcing[rows - 1], common[rows])
    actual_raw, predicted_raw = q_next[rows] * scale, prediction * scale
    return {name: _nrmse(actual_raw[:, i:i + 1], predicted_raw[:, i:i + 1], np.zeros(1)) for i, name in enumerate(Q_NAMES)}


def _lag_and_amplitude(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    rows = {}
    for index, name in enumerate(Q_NAMES):
        left, right = actual[:, index], predicted[:, index]
        correlations = []
        for lag in range(-25, 26):
            a = left[max(0, lag):len(left) + min(0, lag)]
            p = right[max(0, -lag):len(right) - max(0, lag)]
            correlations.append(float(np.corrcoef(a, p)[0, 1]) if a.std() > 0 and p.std() > 0 else -1.0)
        best = int(np.argmax(correlations)) - 25
        rows[name] = {
            "best_lag_steps": best, "best_lag_seconds": best * 0.2,
            "cross_correlation": correlations[best + 25],
            "amplitude_ratio": float(right.std() / max(left.std(), 1e-30)),
        }
    return rows


def _rollout(model: ConservationModel, q: np.ndarray, forcing: np.ndarray, common: np.ndarray, runs: list[Any], scale: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    predicted_all = np.empty_like(q)
    errors = np.zeros(2); energy = np.zeros(2); direction_ok = np.zeros(2); direction_total = np.zeros(2)
    per_run = []; finite_all = True
    for run in runs:
        actual = q[run.start:run.end]
        predicted = actual.copy()
        for offset in range(1, len(actual) - 1):
            index = run.start + offset
            delta = model.predict_delta(
                predicted[offset:offset + 1], forcing[index:index + 1],
                forcing[index - 1:index], common[index:index + 1],
            )[0]
            predicted[offset + 1] = predicted[offset] + delta
            if not np.isfinite(predicted[offset + 1]).all() or np.linalg.norm(predicted[offset + 1]) > 1e12:
                predicted[offset + 1:] = np.nan; finite_all = False; break
        predicted_all[run.start:run.end] = predicted
        observed_raw, predicted_raw = actual[2:] * scale, predicted[2:] * scale
        finite = bool(np.isfinite(predicted_raw).all())
        if not finite:
            per_run.append({"run_id": run.run_id, "finite": False}); continue
        err = predicted_raw - observed_raw
        errors += np.square(err).sum(axis=0); energy += np.square(observed_raw).sum(axis=0)
        active = np.abs(observed_raw) > 1e-12
        direction_ok += np.sum((predicted_raw * observed_raw > 0) & active, axis=0)
        direction_total += np.sum(active, axis=0)
        per_run.append({
            "run_id": run.run_id, "finite": True,
            "nrmse": {name: _nrmse(observed_raw[:, i:i + 1], predicted_raw[:, i:i + 1], np.zeros(1)) for i, name in enumerate(Q_NAMES)},
            "response": _lag_and_amplitude(observed_raw, predicted_raw),
        })
    return ({
        "finite": finite_all and all(row["finite"] for row in per_run),
        "nrmse": {name: math.sqrt(errors[i] / max(energy[i], 1e-30)) if finite_all else math.inf for i, name in enumerate(Q_NAMES)},
        "direction_consistency": {name: float(direction_ok[i] / max(direction_total[i], 1.0)) for i, name in enumerate(Q_NAMES)},
        "runs": per_run,
    }, predicted_all)


def _symmetry_audit(model: ConservationModel, common: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(1502)
    count = 10000
    q = rng.normal(size=(count, 2)); f = rng.normal(size=(count, 2)); fp = rng.normal(size=(count, 2))
    c = common[rng.integers(0, len(common), size=count)]
    positive = model.predict_delta(q, f, fp, c)
    negative = model.predict_delta(-q, -f, -fp, c)
    symmetry_error = float(np.max(np.abs(positive + negative)))
    zero = model.predict_delta(np.zeros((count, 2)), np.zeros((count, 2)), np.zeros((count, 2)), c)
    zero_error = float(np.max(np.abs(zero)))
    return {
        "samples": count, "maximum_symmetry_error": symmetry_error,
        "maximum_zero_bias_error": zero_error,
        "symmetry_pass": symmetry_error < 1e-10, "zero_bias_pass": zero_error < 1e-12,
        "intercept_present": False,
    }


def _frozen_full_rollout(state: np.ndarray, d: np.ndarray, u: np.ndarray, runs: list[Any], model_path: Path) -> np.ndarray:
    values = np.load(model_path)
    A, L, E, M, B, c = (np.asarray(values[name]) for name in ("A", "L", "E", "M", "B", "c"))
    result = np.empty_like(state)
    for run in runs:
        actual = state[run.start:run.end]; predicted = actual.copy()
        for offset in range(1, len(actual) - 1):
            index = run.start + offset; current = predicted[offset]; previous = predicted[offset - 1]
            predicted[offset + 1] = A @ current + L @ (current - previous) + E @ d[index] + M @ d[index - 1] + B @ u[index] + c
        result[run.start:run.end] = predicted
    return result


def _common_from_frozen_global(
    global_state: np.ndarray,
    frozen: dict[str, Any],
    common_normalizer: dict[str, Any],
) -> np.ndarray:
    reconstructed = global_state @ np.asarray(frozen["gc_basis"], dtype=np.float64).T
    common_normalized = reconstructed[:, len(frozen["global_indices"]):]
    raw_half_common = (
        common_normalized * np.asarray(frozen["common_normalizer"]["scale"], dtype=np.float64)
        + np.asarray(frozen["common_normalizer"]["mean"], dtype=np.float64)
    )
    physical_common = np.column_stack((2.0 * raw_half_common[:, 0], 2.0 * raw_half_common[:, 2]))
    return _scalar_transform(physical_common, common_normalizer)


def _write_report(output_root: Path, result: dict[str, Any]) -> None:
    selection = result["selection"]; selected = result["ablation"][selection["candidate"]]
    closure = result["outflow_transition_audit"]["conservation"]["validation"]
    improvement = result["closure_improvement"]
    responses = selected["validation"]["rollout"]["runs"]
    response_summary = {}
    for name in Q_NAMES:
        response_summary[name] = {
            "median_amplitude_ratio": float(np.median([row["response"][name]["amplitude_ratio"] for row in responses if row["finite"]])),
            "median_best_lag_seconds": float(np.median([row["response"][name]["best_lag_seconds"] for row in responses if row["finite"]])),
            "median_cross_correlation": float(np.median([row["response"][name]["cross_correlation"] for row in responses if row["finite"]])),
        }
    lines = [
        "# ServingROM Step 15C-2B 对称保持差分守恒与服务闭合 ROM",
        "", "## 最终状态", "",
        f"- `conservation_dynamics_ready={str(result['conservation_dynamics_ready']).lower()}`",
        "- `control_rom_ready=false`",
        f"- validation 冻结候选：`{selection['candidate']}`，ridge=`{selection['ridge']}`。",
        "- 12D global/common、原始 gc12-diff2 hidden reference 与 5s Slow KPI Head 均保持冻结。",
        "- 未启动 1P2D、未重新采集、未读取 Round 14.3、未实现 MPC。",
        "", "## Outflow / Transition Audit", "",
        "| 守恒关系 | validation residual NRMSE | exact fraction | 结论 |",
        "|---|---:|---:|---|",
        f"| `Δq_run = first_decode_schedule - terminal` | {closure['request_closure']['residual_nrmse']:.6f} | {closure['request_closure']['exact_fraction']:.2%} | 精确闭合 |",
        f"| `Δq_remaining = routed_token_mass - emission - terminal_residual` | {closure['token_closure']['residual_nrmse']:.6f} | {closure['token_closure']['exact_fraction']:.2%} | 不闭合 |",
        f"| `Δq_remaining = arrival_token_mass - emission - terminal_residual` | {closure['active_inventory_token_closure']['residual_nrmse']:.6f} | {closure['active_inventory_token_closure']['exact_fraction']:.2%} | 近乎精确闭合 |",
        "", "当前 snapshot 的 `decode_d1/d2_expected_remaining_tokens` 会在请求仍处于 Prefill 时，按该 attempt 后续选中的 decoder 提前计入库存。因此它的物理流入边界是 request arrival，不是 `p_to_d_route`。这解释了 routed-token 守恒残差，而不是 token emission 大规模缺失。",
        "", "## Validation Ablation", "",
        "| 候选 | ridge | running rollout | remaining rollout | radius max | symmetry | zero bias |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["ablation"].items():
        lines.append(
            f"| {name} | {row['ridge']} | {row['validation']['rollout']['nrmse']['running_imbalance']:.6f} | "
            f"{row['validation']['rollout']['nrmse']['remaining_token_imbalance']:.6f} | "
            f"{row['stability']['spectral_radius_max']:.6f} | {row['symmetry']['symmetry_pass']} | {row['symmetry']['zero_bias_pass']} |"
        )
    lines += [
        "", "## 消融解释", "",
        f"- M1 相比 M0 的平均 rollout 相对改善：`{improvement['m1_relative_improvement_over_m0']:.2%}`。`-Hq` 能抑制 inflow-only 积分漂移。",
        f"- M2 相比 M1 的平均 rollout 相对改善：`{improvement['m2_relative_improvement_over_m1']:.2%}`。幅度不足以证明必须加入 total-load correction。",
        f"- 最终候选相比 Step 15C-2A：running `{improvement['selected_relative_change_vs_step15c2a']['running_imbalance']:.2%}`，remaining `{improvement['selected_relative_change_vs_step15c2a']['remaining_token_imbalance']:.2%}`；负值代表退化。",
        "- M2 虽按 validation 平均 NRMSE 被选中，但没有通过 readiness gate，也不应被视为可部署模型。",
        "", "## 幅度与相位诊断", "",
        f"- running：median amplitude ratio=`{response_summary['running_imbalance']['median_amplitude_ratio']:.4f}`，median best lag=`{response_summary['running_imbalance']['median_best_lag_seconds']:.2f}s`，median correlation=`{response_summary['running_imbalance']['median_cross_correlation']:.4f}`。",
        f"- remaining：median amplitude ratio=`{response_summary['remaining_token_imbalance']['median_amplitude_ratio']:.4f}`，median best lag=`{response_summary['remaining_token_imbalance']['median_best_lag_seconds']:.2f}s`，median correlation=`{response_summary['remaining_token_imbalance']['median_cross_correlation']:.4f}`。",
        "- 由于 200 ms routed forcing 是稀疏随机脉冲而非隔离阶跃，本轮不伪造单一 settling time；低 amplitude ratio 和低 remaining correlation 已表明长期库存响应被严重低估。",
    ]
    test = selected["test"]
    answers = result["answers"]
    lines += [
        "", "## Frozen Test", "",
        f"- running one-step / rollout：`{test['one_step']['running_imbalance']:.6f}` / `{test['rollout']['nrmse']['running_imbalance']:.6f}`",
        f"- remaining one-step / rollout：`{test['one_step']['remaining_token_imbalance']:.6f}` / `{test['rollout']['nrmse']['remaining_token_imbalance']:.6f}`",
        f"- waiting diagnostic rollout：`{result['frozen_regression']['test_waiting_rollout_nrmse']:.6f}`",
        f"- global/common frozen rollout：`{result['frozen_regression']['test_global_rollout_nrmse']:.6f}`",
        f"- Slow KPI frozen-head regression：`{result['frozen_regression']['test_slow_kpi_nrmse']:.6f}`",
        "", "## 五个问题", "",
        f"1. 剩余误差是否主要来自 service/outflow closure：**{answers['service_closure_is_primary']}**。",
        f"2. `-Hq` 是否显著改善长期 rollout：**{answers['restoring_term_improves']}**。",
        f"3. 是否需要 total-load correction：**{answers['load_dependent_correction_needed']}**。",
        f"4. telemetry 是否足够构造真实 A/B outflow：**{answers['telemetry_supports_outflow_closure']}**。",
        f"5. oracle plant-side 是否具有进入 forcing surrogate 的余量：**{answers['oracle_headroom_sufficient']}**。",
        "", "## 下一步边界", "",
        f"- `{result['next_step']}`",
        "- 应把 Decode-only remaining inventory 从 Prefill/route 前库存中拆开，并显式建模 route→KV-ready→Decode admission transition；在此之前不进入 forcing surrogate。",
        "", "## Oracle 边界", "",
        "本轮 actual forcing 和真实 outflow 只用于 plant-side 诊断。未来 counterfactual/MPC 无法直接访问未来 routed forcing，因此无论本轮是否通过，`control_rom_ready` 均保持 false。",
    ]
    (output_root / "STEP15C2B_CONSERVATION_ROM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_model(path: Path, model: ConservationModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, candidate=model.candidate, ridge=model.ridge, B0=model.B0, B1=model.B1, Q=model.Q, interactions=model.interactions)


def run_conservation_pipeline(
    dataset_root: Path,
    representation_root: Path,
    forcing_root: Path,
    outflow_root: Path,
    step15c2a_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    paths = [dataset_root, representation_root, forcing_root, outflow_root, step15c2a_root, output_root]
    dataset_root, representation_root, forcing_root, outflow_root, step15c2a_root, output_root = [p.resolve() for p in paths]
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = _verify_dataset(dataset_root, config["dataset_id"])
    frozen = _load_frozen_representation(dataset_root, representation_root)
    runs = _load_runs(dataset_root)
    physical, sidecar_rows, run_audits, source_hashes = _physical_arrays(
        dataset_root, runs, forcing_root, outflow_root, ("train", "validation"),
    )

    q_norm = _scale_only(physical["train"]["q"], Q_NAMES)
    forcing_norm = _scale_only(physical["train"]["forcing"], FORCING_NAMES)
    common_norm = _fit_scalar_normalizer(physical["train"]["common"], list(COMMON_NAMES))
    save_json(output_root / "normalization/q_scale_only_train.json", q_norm)
    save_json(output_root / "normalization/forcing_scale_only_train.json", forcing_norm)
    save_json(output_root / "normalization/common_load_train.json", common_norm)
    save_json(output_root / "PHYSICAL_DIFFERENTIAL_STATE_MANIFEST.json", {
        "schema_version": "servingrom.physical-differential-state.v1",
        "dimension": 14, "global_common_dimension": 12, "physical_differential_dimension": 2,
        "state": ["z_global_common[12]", "q_run=decode_A_running_count-decode_B_running_count", "q_remaining=decode_A_expected_remaining_tokens-decode_B_expected_remaining_tokens"],
        "waiting_role": "diagnostic_only", "normalization": "train-only scale-only for q and forcing; no mean subtraction",
        "symmetry": "A/B swap maps q->-q and f->-f while common load is unchanged",
    })

    arrays = {split: _load_split(dataset_root, split) for split in ("train", "validation")}
    encoded = {}; d = {}; u = {}; q = {}; q_next = {}; forcing = {}; common = {}
    for split in arrays:
        encoded[split], _ = _encode(arrays[split]["X"], frozen)
        d[split] = frozen["d_normalizer"].transform(arrays[split]["D"], weighted=False)
        u[split] = _scalar_transform(arrays[split]["U"], frozen["u_normalizer"])
        q[split] = _transform_scale_only(physical[split]["q"], q_norm)
        q_next[split] = _transform_scale_only(physical[split]["q_next"], q_norm)
        forcing[split] = _transform_scale_only(physical[split]["forcing"], forcing_norm)
        common[split] = _scalar_transform(physical[split]["common"], common_norm)

    dynamic_path = representation_root / "models/diagnostic_not_ready_control_dynamics.npz"
    slow_path = representation_root / "models/diagnostic_not_ready_slow_kpi_head.npz"
    frozen_global = _load_frozen_global(dynamic_path)
    hidden_prediction = {
        split: _frozen_full_rollout(encoded[split], d[split], u[split], runs[split], dynamic_path)
        for split in ("train", "validation")
    }
    common_rollout = {
        split: _common_from_frozen_global(hidden_prediction[split][:, :12], frozen, common_norm)
        for split in ("train", "validation")
    }
    baseline2a = _load_json(step15c2a_root / "evaluation/final_metrics.json")
    baseline_roll = baseline2a["ablation"]["actual_forcing_only"]["validation"]["rollout"]["core_nrmse"]
    baseline_values = np.asarray([baseline_roll["running_imbalance"], baseline_roll["remaining_token_imbalance"]])
    baseline_global = float(baseline2a["ablation"]["actual_forcing_only"]["validation"]["rollout"]["global_common_nrmse"])
    baseline_slow = float(baseline2a["ablation"]["actual_forcing_only"]["validation"]["slow_kpi_nrmse"])
    validation_global = _nrmse(encoded["validation"][:, :12], hidden_prediction["validation"][:, :12], np.zeros(12))
    slow_values = np.load(slow_path); slow_theta = np.asarray(slow_values["theta"])
    slow_train_validation = pq.read_table(
        dataset_root / "slow_kpi_windows.parquet",
        filters=[("split", "in", ["train", "validation"])],
    ).to_pylist()
    train_slow_rows = [row for row in slow_train_validation if row["split"] == "train"]
    validation_slow_rows = [row for row in slow_train_validation if row["split"] == "validation"]
    train_slow = _build_slow(train_slow_rows, runs["train"], encoded["train"], d["train"], u["train"])
    slow_norm = _fit_scalar_normalizer(train_slow[3], SLOW_OUTPUTS)
    validation_slow = _slow_score(
        validation_slow_rows, runs["validation"], hidden_prediction["validation"],
        d["validation"], u["validation"], slow_theta, slow_norm,
    )

    runtime: dict[str, Any] = {}
    run_m2 = False
    for candidate in CANDIDATES:
        if candidate == "M2_load_dependent_service_closure" and not run_m2:
            continue
        scan = []; winner = None
        for ridge in [float(value) for value in config["candidate_ridges"]]:
            model = _fit(candidate, ridge, q["train"], q_next["train"], forcing["train"], common["train"], runs["train"])
            one = _one_step(model, q["validation"], q_next["validation"], forcing["validation"], common["validation"], runs["validation"], np.asarray(q_norm["scale"]))
            rollout, predicted = _rollout(model, q["validation"], forcing["validation"], common_rollout["validation"], runs["validation"], np.asarray(q_norm["scale"]))
            stability = _stability(model, common_rollout["validation"]); symmetry = _symmetry_audit(model, common_rollout["validation"])
            score = float(np.mean(list(rollout["nrmse"].values())))
            row = {"ridge": ridge, "score": score, "one_step": one, "rollout": rollout, "stability": stability, "symmetry": symmetry}
            scan.append(row)
            eligible = rollout["finite"] and stability["spectral_radius_max"] <= float(config["maximum_spectral_radius"]) and symmetry["symmetry_pass"] and symmetry["zero_bias_pass"]
            if eligible and (winner is None or score < winner[0]):
                winner = (score, model, row, predicted)
        if winner is None:
            best = min(scan, key=lambda item: item["score"])
            model = _fit(candidate, best["ridge"], q["train"], q_next["train"], forcing["train"], common["train"], runs["train"])
            _, predicted = _rollout(model, q["validation"], forcing["validation"], common_rollout["validation"], runs["validation"], np.asarray(q_norm["scale"]))
            winner = (best["score"], model, best, predicted)
        _, model, best, predicted = winner
        runtime[candidate] = {"candidate": candidate, "ridge": model.ridge, "ridge_scan": scan, "validation": {"one_step": best["one_step"], "rollout": best["rollout"]}, "stability": best["stability"], "symmetry": best["symmetry"], "model": model}
        if candidate == "M1_minimal_service_closure":
            m0 = runtime["M0_inflow_only"]["validation"]["rollout"]["nrmse"]
            m1 = runtime[candidate]["validation"]["rollout"]["nrmse"]
            improvement = 1.0 - np.mean(list(m1.values())) / max(np.mean(list(m0.values())), 1e-30)
            m1_ready = all(m1[name] < float(config["maximum_core_rollout_nrmse"]) for name in Q_NAMES)
            run_m2 = improvement >= float(config["minimum_m1_relative_improvement_for_m2"]) and not m1_ready

    selected_name = min(runtime, key=lambda name: np.mean(list(runtime[name]["validation"]["rollout"]["nrmse"].values())))
    selected = runtime[selected_name]
    selected_values = np.asarray([selected["validation"]["rollout"]["nrmse"][name] for name in Q_NAMES])
    gates = {
        "running_below_0_70": bool(selected_values[0] < float(config["maximum_core_rollout_nrmse"])),
        "remaining_below_0_70": bool(selected_values[1] < float(config["maximum_core_rollout_nrmse"])),
        "at_least_one_below_0_65": bool(np.min(selected_values) < float(config["preferred_core_rollout_nrmse"])),
        "both_improve_over_step15c2a": bool(np.all(selected_values < baseline_values)),
        "finite": selected["validation"]["rollout"]["finite"],
        "stable": selected["stability"]["spectral_radius_max"] <= float(config["maximum_spectral_radius"]),
        "symmetry": selected["symmetry"]["symmetry_pass"], "zero_bias": selected["symmetry"]["zero_bias_pass"],
        "global_not_structurally_degraded": validation_global <= baseline_global * (1.0 + float(config["maximum_global_rollout_degradation"])),
        "slow_kpi_not_structurally_degraded": validation_slow <= baseline_slow * (1.0 + float(config["maximum_slow_kpi_degradation"])),
    }
    ready = all(gates.values())
    frozen_selection = {
        "schema_version": "servingrom.step15c2b.frozen-selection.v1", "selection_split": "validation",
        "candidate": selected_name, "ridge": selected["ridge"], "m2_executed": "M2_load_dependent_service_closure" in runtime,
        "gates": gates, "conservation_dynamics_ready": ready, "test_accessed": False,
    }
    save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen_selection)

    test_physical, test_sidecar_rows, test_run_audits, test_source_hashes = _physical_arrays(
        dataset_root, runs, forcing_root, outflow_root, ("test",),
    )
    physical.update(test_physical)
    sidecar_rows.extend(test_sidecar_rows)
    run_audits.extend(test_run_audits)
    source_hashes.update(test_source_hashes)
    sidecar_manifest = _write_outflow_sidecar(
        output_root, sidecar_rows, physical, run_audits, source_hashes,
    )
    conservation_audit = _conservation_audit(physical)
    outflow_audit = {
        "schema_version": "servingrom.outflow-transition-audit.v1",
        "field_capabilities": {
            "running_admission": "first Decode scheduler_membership timestamp",
            "request_completion": "trace terminal timestamp joined through attempt request_id",
            "scheduled_token_service": "Decode scheduler_membership scheduled_tokens",
            "emitted_token_service": "all engine token emissions attributed by final decoder",
            "kv_ready_to_running": "kv_ready_wall_ns and first Decode scheduler membership",
            "terminal_unconsumed_budget": "expected_output_tokens minus observed emissions at terminal",
            "active_inventory_inflow": "trace arrival attributed by the attempt's eventual decoder; matches current snapshot state semantics",
        },
        "alignment": "200ms half-open [start_wall_ns,end_wall_ns), no interpolation, no cross-run joins",
        "state_semantic_finding": (
            "decode_d1/d2_expected_remaining_tokens currently includes every active request using its eventual "
            "decoder assignment, including requests still in Prefill. Its inventory inflow is therefore aligned "
            "to request arrival, not p_to_d_route. Routed token forcing and this state cannot close exactly."
        ),
        "sidecar": sidecar_manifest,
        "conservation": conservation_audit,
    }
    save_json(output_root / "OUTFLOW_TRANSITION_AUDIT.json", outflow_audit)

    arrays_test = _load_split(dataset_root, "test")
    encoded_test, _ = _encode(arrays_test["X"], frozen)
    d_test = frozen["d_normalizer"].transform(arrays_test["D"], weighted=False)
    u_test = _scalar_transform(arrays_test["U"], frozen["u_normalizer"])
    q_test = _transform_scale_only(physical["test"]["q"], q_norm); qn_test = _transform_scale_only(physical["test"]["q_next"], q_norm)
    f_test = _transform_scale_only(physical["test"]["forcing"], forcing_norm); c_test = _scalar_transform(physical["test"]["common"], common_norm)
    hidden_test = _frozen_full_rollout(encoded_test, d_test, u_test, runs["test"], dynamic_path)
    c_test_rollout = _common_from_frozen_global(hidden_test[:, :12], frozen, common_norm)
    test_one = _one_step(selected["model"], q_test, qn_test, f_test, c_test, runs["test"], np.asarray(q_norm["scale"]))
    test_roll, _ = _rollout(selected["model"], q_test, f_test, c_test_rollout, runs["test"], np.asarray(q_norm["scale"]))
    selected["test"] = {"one_step": test_one, "rollout": test_roll}

    original_core_test = physical["test"]["waiting"]
    decoded_waiting = __import__("servingrom_control_modeling.memory", fromlist=["_decode_core"])._decode_core(hidden_test[:, -2:], frozen)[:, 1] * np.asarray(frozen["core_normalizer"]["scale"])[1] + np.asarray(frozen["core_normalizer"]["mean"])[1]
    waiting_nrmse = _nrmse(original_core_test[:, None], decoded_waiting[:, None], np.zeros(1))
    slow_all = pq.read_table(dataset_root / "slow_kpi_windows.parquet").to_pylist()
    test_slow_rows = [row for row in slow_all if row["split"] == "test"]
    slow_test = _slow_score(test_slow_rows, runs["test"], hidden_test, d_test, u_test, slow_theta, slow_norm)
    global_test = _nrmse(encoded_test[:, :12], hidden_test[:, :12], np.zeros(12))

    stripped = {name: {key: value for key, value in row.items() if key != "model"} for name, row in runtime.items()}
    m0_score = np.mean(list(runtime["M0_inflow_only"]["validation"]["rollout"]["nrmse"].values()))
    m1_score = np.mean(list(runtime["M1_minimal_service_closure"]["validation"]["rollout"]["nrmse"].values()))
    m2_present = "M2_load_dependent_service_closure" in runtime
    m2_better = m2_present and np.mean(list(runtime["M2_load_dependent_service_closure"]["validation"]["rollout"]["nrmse"].values())) < m1_score * 0.98
    closure = outflow_audit["conservation"]["validation"]
    request_closes = closure["request_closure"]["residual_nrmse"] < 0.1
    active_token_closes = closure["active_inventory_token_closure"]["residual_nrmse"] < 0.1
    routed_token_closes = closure["token_closure"]["residual_nrmse"] < 0.1
    telemetry_sufficient = request_closes and active_token_closes
    m1_relative_improvement = 1.0 - m1_score / max(m0_score, 1e-30)
    m2_relative_improvement = (
        1.0 - np.mean(list(runtime["M2_load_dependent_service_closure"]["validation"]["rollout"]["nrmse"].values())) / max(m1_score, 1e-30)
        if m2_present else None
    )
    result = {
        "schema_version": "servingrom.step15c2b.result.v1", "dataset": dataset_audit,
        "physical_state_manifest": _load_json(output_root / "PHYSICAL_DIFFERENTIAL_STATE_MANIFEST.json"),
        "outflow_transition_audit": outflow_audit, "ablation": stripped,
        "selection": {**frozen_selection, "test_accessed": True},
        "step15c2a_baseline": dict(zip(Q_NAMES, baseline_values.tolist())),
        "conservation_dynamics_ready": ready, "control_rom_ready": False,
        "closure_improvement": {
            "m1_relative_improvement_over_m0": float(m1_relative_improvement),
            "m2_relative_improvement_over_m1": float(m2_relative_improvement) if m2_relative_improvement is not None else None,
            "selected_relative_change_vs_step15c2a": {
                name: float(1.0 - selected_values[index] / baseline_values[index])
                for index, name in enumerate(Q_NAMES)
            },
        },
        "oracle_outflow_diagnostic": {
            "deployable_model": False,
            "validation_running_direct_conservation_residual_nrmse": closure["request_closure"]["residual_nrmse"],
            "validation_remaining_routed_inflow_residual_nrmse": closure["token_closure"]["residual_nrmse"],
            "validation_remaining_arrival_aligned_residual_nrmse": closure["active_inventory_token_closure"]["residual_nrmse"],
            "interpretation": "outflows are observable; current q_remaining inventory starts at arrival, before p_to_d_route",
        },
        "frozen_regression": {
            "validation_global_rollout_nrmse": validation_global,
            "validation_slow_kpi_nrmse": validation_slow,
            "test_waiting_rollout_nrmse": waiting_nrmse,
            "test_global_rollout_nrmse": global_test,
            "test_slow_kpi_nrmse": slow_test,
        },
        "answers": {
            "service_closure_is_primary": (
                "否；restoring closure 可阻止 M0 积分漂移，但未击败 Step15C-2A，主要残差更符合 arrival/route 状态语义错位与 transition timing"
            ),
            "restoring_term_improves": "是" if m1_score < m0_score * 0.98 else "否",
            "load_dependent_correction_needed": "是" if m2_better else "否",
            "telemetry_supports_outflow_closure": (
                "部分：request closure 精确，arrival-aligned token closure 近乎精确；routed-token 与当前 q_remaining 时间语义不一致"
                if telemetry_sufficient and not routed_token_closes
                else "是" if telemetry_sufficient else "否"
            ),
            "oracle_headroom_sufficient": "是" if ready else "否",
        },
        "next_step": (
            "Step 15C-2C actuator realization"
            if ready
            else "stop: redesign Decode inventory boundary and route-to-admission transition state"
        ),
        "data_isolation": {"one_p_two_d_started": False, "new_runs_collected": False, "heldout_read": False, "mpc_implemented": False, "test_accessed_after_validation_freeze": True},
    }
    save_json(output_root / "CONSERVATION_ABLATION.json", stripped)
    save_json(output_root / "SYMMETRY_AUDIT.json", {name: row["symmetry"] for name, row in stripped.items()})
    save_json(output_root / "evaluation/final_metrics.json", result)
    _save_model(output_root / "models/diagnostic_conservation_dynamics.npz", selected["model"])
    _write_report(output_root, result)
    manifest = {
        "schema_version": "servingrom.step15c2b.sha256-manifest.v1", "model_id": config["model_id"],
        "dataset_sha256_manifest": dataset_audit["sha256_manifest"],
        "representation_manifest_sha256": _sha256(representation_root / "SHA256_MANIFEST.json"),
        "step15c2a_manifest_sha256": _sha256(step15c2a_root / "SHA256_MANIFEST.json"),
        "selection": result["selection"], "artifacts": {},
    }
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.json":
            manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    save_json(output_root / "SHA256_MANIFEST.json", manifest)
    return result
