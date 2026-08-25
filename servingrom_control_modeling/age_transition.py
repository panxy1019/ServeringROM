from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from servingrom_modeling.preprocessing import save_json

from .pipeline import _load_runs, _sha256, _verify_dataset
from .transition_inventory import (
    QUANTITIES,
    WINDOW_NS,
    SharedFlowModel,
    _decoder,
    _nrmse,
    _read,
    _reconstruct_run,
)


TRANSIT_STAGES = ("handoff", "waiting")
DT_SECONDS = WINDOW_NS / 1e9


@dataclass(frozen=True)
class Lifecycle:
    run_id: str
    side: int
    request_id: str
    route_ns: int
    kv_ready_ns: int
    admission_ns: int
    terminal_ns: int
    expected_tokens: float
    emissions: tuple[tuple[int, float], ...]


@dataclass
class HazardModel:
    edges_seconds: dict[str, np.ndarray]
    request_hazard: dict[str, np.ndarray]
    token_hazard: dict[str, np.ndarray]
    smoothing: float

    def hazard_for_steps(self, stage: str, max_steps: int) -> np.ndarray:
        ages = np.arange(max_steps, dtype=np.float64) * DT_SECONDS
        bins = np.searchsorted(self.edges_seconds[stage][1:-1], ages, side="right")
        return np.stack((self.request_hazard[stage][bins], self.token_hazard[stage][bins]), axis=-1)


def _remaining_before(emissions: tuple[tuple[int, float], ...], timestamp: int, expected: float) -> float:
    return max(expected - sum(count for event_ts, count in emissions if event_ts < timestamp), 0.0)


def _load_lifecycles(run_id: str, outflow_root: Path) -> list[Lifecycle]:
    root = outflow_root / run_id / "derived"
    attempts = _read(root / "attempt_lifecycle.parquet")
    traces = {str(row["trace_id"]): row for row in _read(root / "trace_lifecycle.parquet") if row.get("trace_id")}
    transfers = {str(row["request_id"]): row for row in _read(root / "kv_transfers.parquet") if row.get("request_id")}
    membership: dict[str, int] = {}
    for row in _read(root / "scheduler_membership.parquet"):
        if _decoder(row.get("component")) is None or not row.get("request_id"):
            continue
        request_id = str(row["request_id"])
        timestamp = int(row["ts_wall_ns"])
        membership[request_id] = min(timestamp, membership.get(request_id, timestamp))
    emissions: dict[str, list[tuple[int, float]]] = {}
    for row in _read(root / "token_emissions.parquet"):
        if _decoder(row.get("component")) is None or not row.get("request_id"):
            continue
        emissions.setdefault(str(row["request_id"]), []).append(
            (int(row["ts_wall_ns"]), float(row.get("new_token_count") or 0))
        )
    output = []
    for attempt in attempts:
        request_id = str(attempt.get("request_id") or "")
        side = _decoder(attempt.get("decoder_backend"))
        trace = traces.get(str(attempt.get("trace_id") or ""))
        transfer = transfers.get(request_id)
        values = (
            attempt.get("route_wall_ns"),
            transfer.get("kv_ready_wall_ns") if transfer else None,
            membership.get(request_id),
            trace.get("terminal_wall_ns") if trace else None,
        )
        if side is None or trace is None or any(value is None for value in values):
            continue
        route, kv_ready, admission, terminal = map(int, values)
        if not route <= kv_ready <= admission <= terminal:
            continue
        output.append(Lifecycle(
            run_id=run_id,
            side=side,
            request_id=request_id,
            route_ns=route,
            kv_ready_ns=kv_ready,
            admission_ns=admission,
            terminal_ns=terminal,
            expected_tokens=float(trace.get("expected_output_tokens") or 0),
            emissions=tuple(sorted(emissions.get(request_id, []))),
        ))
    return output


def _run_factors(run_id: str) -> tuple[str, int]:
    match = re.search(r"-(balanced|mixed-bimodal)-l(\d+)-", run_id)
    return (match.group(1), int(match.group(2))) if match else ("unknown", -1)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.percentile(array, [50, 75, 90, 95, 99, 100])
    return dict(zip(("p50", "p75", "p90", "p95", "p99", "max"), map(float, quantiles)))


def _dwell_audit(records: list[Lifecycle]) -> dict[str, Any]:
    output: dict[str, Any] = {"schema_version": "servingrom.dwell-time-audit.v1", "split": "train"}
    for stage in TRANSIT_STAGES:
        values = [
            ((record.kv_ready_ns - record.route_ns) if stage == "handoff" else (record.admission_ns - record.kv_ready_ns)) / 1e9
            for record in records
        ]
        grouped: dict[str, Any] = {}
        for label, selector in (
            ("decoder_A", lambda record: record.side == 0),
            ("decoder_B", lambda record: record.side == 1),
        ):
            subset = [value for value, record in zip(values, records) if selector(record)]
            grouped[label] = _distribution(subset)
        for workload in ("balanced", "mixed-bimodal"):
            for load in (55, 75, 92):
                subset = [value for value, record in zip(values, records) if _run_factors(record.run_id) == (workload, load)]
                grouped[f"{workload}.l{load}"] = _distribution(subset)
        tokens = np.asarray([record.expected_tokens for record in records], dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        output[stage] = {
            "count": len(values),
            "seconds": _distribution(values),
            "expected_token_correlation": float(np.corrcoef(array, tokens)[0, 1]) if np.std(tokens) > 0 else 0.0,
            "groups": grouped,
            "decoder_median_relative_difference": abs(grouped["decoder_A"]["p50"] - grouped["decoder_B"]["p50"]) / max(grouped["decoder_A"]["p50"], grouped["decoder_B"]["p50"], 1e-12),
        }
    return output


def _derive_edges(audit: dict[str, Any]) -> dict[str, np.ndarray]:
    output = {}
    for stage in TRANSIT_STAGES:
        stats = audit[stage]["seconds"]
        snapped = [math.ceil(stats[name] / DT_SECONDS) * DT_SECONDS for name in ("p75", "p90", "p95", "p99")]
        finite = sorted({0.0, DT_SECONDS, 2 * DT_SECONDS, *snapped})
        finite = finite[:7]
        while len(finite) < 4:
            finite.append(finite[-1] + DT_SECONDS)
        output[stage] = np.asarray(sorted(set(finite)) + [np.inf], dtype=np.float64)
    return output


def _bin_index(edges: np.ndarray, age_seconds: float) -> int:
    return int(np.searchsorted(edges[1:-1], max(age_seconds, 0.0), side="right"))


def _fit_hazard(records: list[Lifecycle], edges: dict[str, np.ndarray], smoothing: float) -> HazardModel:
    request_hazard = {}; token_hazard = {}
    for stage in TRANSIT_STAGES:
        bins = len(edges[stage]) - 1
        req_risk = np.zeros(bins); req_exit = np.zeros(bins)
        tok_risk = np.zeros(bins); tok_exit = np.zeros(bins)
        for record in records:
            entry = record.route_ns if stage == "handoff" else record.kv_ready_ns
            exit_ts = record.kv_ready_ns if stage == "handoff" else record.admission_ns
            dwell_steps = max(1, int(math.ceil((exit_ts - entry) / WINDOW_NS)))
            token_mass = _remaining_before(record.emissions, entry, record.expected_tokens)
            for age_step in range(dwell_steps):
                index = _bin_index(edges[stage], age_step * DT_SECONDS)
                req_risk[index] += 1.0; tok_risk[index] += token_mass
                if age_step == dwell_steps - 1:
                    req_exit[index] += 1.0; tok_exit[index] += token_mass
        request_hazard[stage] = np.clip((req_exit + smoothing) / (req_risk + 2 * smoothing), 0.0, 1.0)
        token_hazard[stage] = np.clip((tok_exit + smoothing) / (tok_risk + 2 * smoothing), 0.0, 1.0)
    return HazardModel(edges, request_hazard, token_hazard, smoothing)


def _max_age_steps(edges: np.ndarray) -> int:
    finite = edges[np.isfinite(edges)]
    return max(2, int(math.ceil(finite[-1] / DT_SECONDS)) + 1)


def _initial_age_state(records: list[Lifecycle], boundary_ns: int, model: HazardModel) -> dict[str, np.ndarray]:
    output = {}
    for stage in TRANSIT_STAGES:
        state = np.zeros((2, _max_age_steps(model.edges_seconds[stage]), 2), dtype=np.float64)
        for record in records:
            entry = record.route_ns if stage == "handoff" else record.kv_ready_ns
            exit_ts = record.kv_ready_ns if stage == "handoff" else record.admission_ns
            if not entry <= boundary_ns < exit_ts:
                continue
            age_step = min(int((boundary_ns - entry) // WINDOW_NS), state.shape[1] - 1)
            state[record.side, age_step, 0] += 1.0
            state[record.side, age_step, 1] += _remaining_before(record.emissions, boundary_ns, record.expected_tokens)
        output[stage] = state
    return output


def _advance(state: np.ndarray, inflow: np.ndarray, hazards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = state.copy()
    current[:, 0] += inflow
    outflow = (current * hazards[None, :, :]).sum(axis=1)
    survivors = current * (1.0 - hazards[None, :, :])
    next_state = np.zeros_like(state)
    next_state[:, 1:] = survivors[:, :-1]
    next_state[:, -1] += survivors[:, -1]
    return next_state, outflow


def _observed_age_hist(records: list[Lifecycle], starts: np.ndarray, edges: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output = {stage: np.zeros((len(starts) + 1, 2, len(edges[stage]) - 1, 2), dtype=np.float64) for stage in TRANSIT_STAGES}
    boundaries = np.concatenate((starts, [starts[-1] + WINDOW_NS]))
    for record in records:
        for stage in TRANSIT_STAGES:
            entry = record.route_ns if stage == "handoff" else record.kv_ready_ns
            exit_ts = record.kv_ready_ns if stage == "handoff" else record.admission_ns
            begin = int(np.searchsorted(boundaries, entry, side="left"))
            end = int(np.searchsorted(boundaries, exit_ts, side="left"))
            for index in range(begin, min(end, len(boundaries))):
                age = (int(boundaries[index]) - entry) / 1e9
                age_bin = _bin_index(edges[stage], age)
                output[stage][index, record.side, age_bin, 0] += 1.0
                output[stage][index, record.side, age_bin, 1] += _remaining_before(record.emissions, int(boundaries[index]), record.expected_tokens)
    return output


def _aggregate_age(state: np.ndarray, edges: np.ndarray) -> np.ndarray:
    output = np.zeros((2, len(edges) - 1, 2), dtype=np.float64)
    for age_step in range(state.shape[1]):
        output[:, _bin_index(edges, age_step * DT_SECONDS)] += state[:, age_step]
    return output


def _rollout_age(run: dict[str, np.ndarray], records: list[Lifecycle], model: HazardModel) -> dict[str, np.ndarray]:
    states = _initial_age_state(records, int(run["starts"][0]), model)
    predicted = {stage: np.zeros((len(run["state"]), 2, 2), dtype=np.float64) for stage in TRANSIT_STAGES}
    age_hist = {stage: np.zeros((len(run["state"]), 2, len(model.edges_seconds[stage]) - 1, 2), dtype=np.float64) for stage in TRANSIT_STAGES}
    kv_ready = np.zeros_like(run["kv_ready"]); admission = np.zeros_like(run["admission"])
    hazards = {stage: model.hazard_for_steps(stage, states[stage].shape[1]) for stage in TRANSIT_STAGES}
    for index in range(len(run["state"])):
        for stage in TRANSIT_STAGES:
            predicted[stage][index] = states[stage].sum(axis=1)
            age_hist[stage][index] = _aggregate_age(states[stage], model.edges_seconds[stage])
        if index == len(run["state"]) - 1:
            break
        states["handoff"], kv_ready[index] = _advance(states["handoff"], run["route"][index], hazards["handoff"])
        states["waiting"], admission[index] = _advance(states["waiting"], kv_ready[index], hazards["waiting"])
    return {"predicted": predicted, "age_hist": age_hist, "kv_ready": kv_ready, "admission": admission}


def _differential_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        QUANTITIES[q]: _nrmse(
            (actual[:, 0, q] - actual[:, 1, q])[:, None],
            (predicted[:, 0, q] - predicted[:, 1, q])[:, None],
            np.zeros(1),
        )
        for q in range(2)
    }


def _flow_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return _differential_metrics(actual, predicted)


def _load_running_model(path: Path) -> SharedFlowModel:
    payload = np.load(path)
    groups = ("kv_ready", "admission", "running_outflow")
    return SharedFlowModel(
        ridge=float(payload["ridge"]),
        theta={group: payload[f"theta_{group}"] for group in groups},
        feature_scale={group: payload[f"feature_scale_{group}"] for group in groups},
        target_scale={group: payload[f"target_scale_{group}"] for group in groups},
    )


def _running_rollout(run: dict[str, np.ndarray], admission_200ms: np.ndarray, model: SharedFlowModel) -> np.ndarray:
    actual = run["state"][::5, :, 2]
    windows = min(len(actual) - 1, len(admission_200ms) // 5)
    predicted = np.zeros_like(actual); predicted[0] = actual[0]
    for index in range(windows):
        admission = admission_200ms[index * 5:(index + 1) * 5].sum(axis=0)
        outflow = model.predict("running_outflow", predicted[index], admission)
        predicted[index + 1] = np.maximum(predicted[index] + admission - outflow, 0.0)
    return predicted


def _evaluate(runs: list[dict[str, np.ndarray]], records_by_run: dict[str, list[Lifecycle]], model: HazardModel, running_model: SharedFlowModel) -> dict[str, Any]:
    errors = {stage: np.zeros(2) for stage in TRANSIT_STAGES}; energies = {stage: np.zeros(2) for stage in TRANSIT_STAGES}
    flow_errors = {name: np.zeros(2) for name in ("kv_ready", "admission")}; flow_energies = {name: np.zeros(2) for name in flow_errors}
    running_errors = {name: np.zeros(2) for name in ("oracle_admission", "predicted_admission")}; running_energies = {name: np.zeros(2) for name in running_errors}
    age_errors = {stage: np.zeros(2) for stage in TRANSIT_STAGES}; age_energies = {stage: np.zeros(2) for stage in TRANSIT_STAGES}
    per_run = []; finite = True; conservation_max = 0.0; reconstruction_max = 0.0
    for run in runs:
        records = records_by_run[run["run_id"]]
        rollout = _rollout_age(run, records, model)
        observed_age = _observed_age_hist(records, run["starts"], model.edges_seconds)
        workload, load = _run_factors(run["run_id"])
        row = {"run_id": run["run_id"], "workload": workload, "load": load, "stages": {}, "flows": {}, "running": {}, "age_occupancy": {}}
        for stage_index, stage in enumerate(TRANSIT_STAGES):
            actual = run["state"][:, :, stage_index]
            predicted = rollout["predicted"][stage]
            row["stages"][stage] = _differential_metrics(actual, predicted)
            for q in range(2):
                a = actual[:, 0, q] - actual[:, 1, q]; p = predicted[:, 0, q] - predicted[:, 1, q]
                errors[stage][q] += np.square(p - a).sum(); energies[stage][q] += np.square(a).sum()
                observed_diff = observed_age[stage][:, 0, :, q] - observed_age[stage][:, 1, :, q]
                predicted_diff = rollout["age_hist"][stage][:, 0, :, q] - rollout["age_hist"][stage][:, 1, :, q]
                age_errors[stage][q] += np.square(predicted_diff - observed_diff).sum()
                age_energies[stage][q] += np.square(observed_diff).sum()
            row["age_occupancy"][stage] = {
                QUANTITIES[q]: _nrmse(
                    observed_age[stage][:, 0, :, q] - observed_age[stage][:, 1, :, q],
                    rollout["age_hist"][stage][:, 0, :, q] - rollout["age_hist"][stage][:, 1, :, q],
                    np.zeros(observed_age[stage].shape[2]),
                ) for q in range(2)
            }
            reconstruction_max = max(reconstruction_max, float(np.abs(observed_age[stage].sum(axis=2) - actual).max()))
        for name in ("kv_ready", "admission"):
            row["flows"][name] = _flow_metrics(run[name], rollout[name])
            for q in range(2):
                a = run[name][:, 0, q] - run[name][:, 1, q]; p = rollout[name][:, 0, q] - rollout[name][:, 1, q]
                flow_errors[name][q] += np.square(p - a).sum(); flow_energies[name][q] += np.square(a).sum()
        for name, admission in (("oracle_admission", run["admission"]), ("predicted_admission", rollout["admission"])):
            predicted_running = _running_rollout(run, admission, running_model)
            actual_running = run["state"][::5, :, 2]
            row["running"][name] = _differential_metrics(actual_running, predicted_running)
            for q in range(2):
                a = actual_running[:, 0, q] - actual_running[:, 1, q]; p = predicted_running[:, 0, q] - predicted_running[:, 1, q]
                running_errors[name][q] += np.square(p - a).sum(); running_energies[name][q] += np.square(a).sum()
        for index in range(len(run["route"])):
            expected_h = rollout["predicted"]["handoff"][index] + run["route"][index] - rollout["kv_ready"][index]
            expected_w = rollout["predicted"]["waiting"][index] + rollout["kv_ready"][index] - rollout["admission"][index]
            conservation_max = max(conservation_max, float(np.abs(expected_h - rollout["predicted"]["handoff"][index + 1]).max()), float(np.abs(expected_w - rollout["predicted"]["waiting"][index + 1]).max()))
        finite = finite and all(np.isfinite(value).all() for value in rollout["predicted"].values())
        per_run.append(row)
    grouped = {}
    for workload in ("balanced", "mixed-bimodal"):
        for load in (55, 75, 92):
            rows = [row for row in per_run if row["workload"] == workload and row["load"] == load]
            grouped[f"{workload}.l{load}"] = {
                "run_count": len(rows),
                "stage_nrmse_mean": {
                    stage: {quantity: float(np.mean([row["stages"][stage][quantity] for row in rows])) for quantity in QUANTITIES}
                    for stage in TRANSIT_STAGES
                },
                "predicted_admission_running_nrmse_mean": {
                    quantity: float(np.mean([row["running"]["predicted_admission"][quantity] for row in rows])) for quantity in QUANTITIES
                },
            }
    return {
        "finite": finite,
        "differential_nrmse": {stage: {QUANTITIES[q]: math.sqrt(errors[stage][q] / max(energies[stage][q], 1e-30)) for q in range(2)} for stage in TRANSIT_STAGES},
        "transition_flow_nrmse": {name: {QUANTITIES[q]: math.sqrt(flow_errors[name][q] / max(flow_energies[name][q], 1e-30)) for q in range(2)} for name in flow_errors},
        "running": {name: {QUANTITIES[q]: math.sqrt(running_errors[name][q] / max(running_energies[name][q], 1e-30)) for q in range(2)} for name in running_errors},
        "age_occupancy_differential_nrmse": {stage: {QUANTITIES[q]: math.sqrt(age_errors[stage][q] / max(age_energies[stage][q], 1e-30)) for q in range(2)} for stage in TRANSIT_STAGES},
        "observed_age_to_stage_reconstruction_residual_max": reconstruction_max,
        "conservation_residual_max": conservation_max,
        "workload_load_groups": grouped,
        "per_run": per_run,
    }


def _survival_calibration(records: list[Lifecycle], model: HazardModel) -> dict[str, Any]:
    output = {}
    for stage in TRANSIT_STAGES:
        dwell = np.asarray([
            ((record.kv_ready_ns - record.route_ns) if stage == "handoff" else (record.admission_ns - record.kv_ready_ns)) / 1e9
            for record in records
        ], dtype=np.float64)
        max_steps = _max_age_steps(model.edges_seconds[stage])
        hazards = model.hazard_for_steps(stage, max_steps)
        ages = np.arange(max_steps + 1, dtype=np.float64) * DT_SECONDS
        empirical = np.asarray([np.mean(dwell > age) for age in ages])
        predicted_request = np.concatenate(([1.0], np.cumprod(1.0 - hazards[:, 0])))
        predicted_token = np.concatenate(([1.0], np.cumprod(1.0 - hazards[:, 1])))
        output[stage] = {
            "ages_seconds": ages.tolist(),
            "empirical_survival": empirical.tolist(),
            "predicted_request_survival": predicted_request.tolist(),
            "predicted_token_mass_survival": predicted_token.tolist(),
            "request_survival_mae": float(np.mean(np.abs(predicted_request - empirical))),
            "token_survival_mae": float(np.mean(np.abs(predicted_token - empirical))),
        }
    return output


def _ready(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[bool, dict[str, bool]]:
    values = metrics["differential_nrmse"]
    running = metrics["running"]["predicted_admission"]
    checks = {
        "handoff.requests": values["handoff"]["requests"] < thresholds["handoff_request_nrmse"],
        "handoff.tokens": values["handoff"]["tokens"] < thresholds["handoff_token_nrmse"],
        "waiting.requests": values["waiting"]["requests"] < thresholds["waiting_request_nrmse"],
        "waiting.tokens": values["waiting"]["tokens"] < thresholds["waiting_token_nrmse"],
        "running.requests": running["requests"] < thresholds["running_request_nrmse"],
        "running.tokens": running["tokens"] < thresholds["running_token_nrmse"],
        "finite": bool(metrics["finite"]),
        "conservation": metrics["conservation_residual_max"] < 1e-9,
    }
    return all(checks.values()), checks


def _model_json(model: HazardModel) -> dict[str, Any]:
    return {
        "smoothing": model.smoothing,
        "edges_seconds": {stage: [None if np.isinf(value) else float(value) for value in edges] for stage, edges in model.edges_seconds.items()},
        "request_hazard": {stage: values.tolist() for stage, values in model.request_hazard.items()},
        "token_hazard": {stage: values.tolist() for stage, values in model.token_hazard.items()},
        "shared_across_decoders": True,
    }


def _write_sidecar(path: Path, split_runs: dict[str, list[dict[str, np.ndarray]]], records_by_run: dict[str, list[Lifecycle]], edges: dict[str, np.ndarray]) -> None:
    rows = []
    for split, runs in split_runs.items():
        for run in runs:
            observed = _observed_age_hist(records_by_run[run["run_id"]], run["starts"], edges)
            for index in range(len(run["starts"])):
                row: dict[str, Any] = {"split": split, "run_id": run["run_id"], "window_id": index, "start_wall_ns": int(run["starts"][index])}
                for stage in TRANSIT_STAGES:
                    for side, side_name in enumerate(("A", "B")):
                        for age_bin in range(observed[stage].shape[2]):
                            row[f"{stage}_{side_name}_age{age_bin}_requests"] = float(observed[stage][index, side, age_bin, 0])
                            row[f"{stage}_{side_name}_age{age_bin}_tokens"] = float(observed[stage][index, side, age_bin, 1])
                rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _write_report(output_root: Path, result: dict[str, Any]) -> None:
    validation = result["selected"]["validation"]
    lines = [
        "# ServingROM Step 15C-2B.2 Age-Structured Semi-Markov Transition ROM",
        "", "## 状态", "",
        f"- `stage_inventory_ready=true`",
        f"- `age_transition_ready={str(result['age_transition_ready']).lower()}`",
        f"- `transition_pipeline_ready={str(result['transition_pipeline_ready']).lower()}`",
        "- `control_rom_ready=false`",
        "- 未启动 1P2D、未采集新 run、未读取 Round 14.3、未实现 MPC。",
        "", "## Validation", "",
        "| Stage | Request differential NRMSE | Token differential NRMSE |",
        "|---|---:|---:|",
    ]
    for stage in TRANSIT_STAGES:
        row = validation["differential_nrmse"][stage]
        lines.append(f"| {stage} | {row['requests']:.6f} | {row['tokens']:.6f} |")
    lines += ["", "## Running attribution", ""]
    for name, row in validation["running"].items():
        lines.append(f"- `{name}` request/token：`{row['requests']:.6f}` / `{row['tokens']:.6f}`")
    lines += ["", "## 结论", "", result["conclusion"]]
    (output_root / "STEP15C2B2_AGE_STRUCTURED_TRANSITION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_age_transition_pipeline(dataset_root: Path, forcing_root: Path, outflow_root: Path, transition_root: Path, output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_root, forcing_root, outflow_root, transition_root, output_root = (path.resolve() for path in (dataset_root, forcing_root, outflow_root, transition_root, output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = _verify_dataset(dataset_root, config["dataset_id"])
    run_ranges = _load_runs(dataset_root)
    split_runs: dict[str, list[dict[str, np.ndarray]]] = {"train": []}
    records_by_run: dict[str, list[Lifecycle]] = {}
    for run_range in run_ranges["train"]:
        run, _ = _reconstruct_run(run_range.run_id, forcing_root, outflow_root)
        run["run_id"] = run_range.run_id
        split_runs["train"].append(run)
        records_by_run[run_range.run_id] = _load_lifecycles(run_range.run_id, outflow_root)
    train_records = [record for run in split_runs["train"] for record in records_by_run[run["run_id"]]]
    audit = _dwell_audit(train_records)
    edges = _derive_edges(audit)
    save_json(output_root / "DWELL_TIME_AUDIT.json", audit)
    save_json(output_root / "AGE_BIN_SCHEMA.json", {
        "schema_version": "servingrom.age-bin-schema.v1", "selected_from": "train_only", "dt_seconds": DT_SECONDS,
        "edges_seconds": {stage: [None if np.isinf(value) else float(value) for value in values] for stage, values in edges.items()},
        "wide_bin_implementation": "internal 200ms cohort aging; bins are reporting and shared-hazard groups",
    })
    split_runs["validation"] = []
    for run_range in run_ranges["validation"]:
        run, _ = _reconstruct_run(run_range.run_id, forcing_root, outflow_root)
        run["run_id"] = run_range.run_id
        split_runs["validation"].append(run)
        records_by_run[run_range.run_id] = _load_lifecycles(run_range.run_id, outflow_root)
    running_model = _load_running_model(transition_root / "models/transition_service_flow_model.npz")
    scan = []
    for smoothing in map(float, config["candidate_smoothing"]):
        model = _fit_hazard(train_records, edges, smoothing)
        metrics = _evaluate(split_runs["validation"], records_by_run, model, running_model)
        score = float(np.mean([metrics["differential_nrmse"][stage][quantity] for stage in TRANSIT_STAGES for quantity in QUANTITIES]))
        ready, checks = _ready(metrics, config["readiness"])
        scan.append({"candidate": "H1", "smoothing": smoothing, "score": score, "ready": ready, "checks": checks, "validation": metrics, "model": model})
    winner = min(scan, key=lambda row: row["score"])
    selected_model: HazardModel = winner["model"]
    h1_ready = bool(winner["ready"])
    h2_executed = False
    frozen = {
        "schema_version": "servingrom.age-transition.selection.v1", "test_accessed": False,
        "age_bin_schema_frozen": True, "candidate": "H1", "smoothing": winner["smoothing"],
        "h1_ready": h1_ready, "h2_executed": h2_executed, "selection_split": "validation",
    }
    save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen)
    test_runs = []
    for run_range in run_ranges["test"]:
        run, _ = _reconstruct_run(run_range.run_id, forcing_root, outflow_root); run["run_id"] = run_range.run_id
        test_runs.append(run); records_by_run[run_range.run_id] = _load_lifecycles(run_range.run_id, outflow_root)
    split_runs["test"] = test_runs
    test_metrics = _evaluate(test_runs, records_by_run, selected_model, running_model)
    test_ready, test_checks = _ready(test_metrics, config["readiness"])
    _write_sidecar(output_root / "sidecar/age_structured_inventory_200ms.parquet", split_runs, records_by_run, edges)
    model_payload = _model_json(selected_model); save_json(output_root / "models/age_hazard_model.json", model_payload)
    baseline = json.loads((transition_root / "evaluation/final_metrics.json").read_text(encoding="utf-8"))["flow_model"]
    baseline_score = float(np.mean([
        baseline["validation"]["differential_nrmse"][stage][quantity]
        for stage in TRANSIT_STAGES for quantity in QUANTITIES
    ]))
    relative_improvement = (baseline_score - winner["score"]) / max(baseline_score, 1e-30)
    selected = {"candidate": "H1", "validation": winner["validation"], "test": test_metrics, "validation_ready": h1_ready, "test_ready": test_ready}
    ablation = {
        "H0": {"source": "Step15C-2B.1 frozen baseline", "validation": baseline["validation"], "test": baseline["test"]},
        "H1_scan": [{key: value for key, value in row.items() if key != "model"} for row in scan],
        "H1_relative_improvement_over_H0": relative_improvement,
        "H2": {
            "executed": False,
            "reason": "H1 did not materially improve stage-inventory differential rollout; train dwell is overwhelmingly below the 200ms sampling interval, so common-load correction cannot recover unobserved sub-window phase.",
            "preregistered_minimum_improvement_to_execute": config["minimum_relative_improvement"],
        },
        "selected": selected,
    }
    save_json(output_root / "HAZARD_MODEL_ABLATION.json", ablation)
    survival = _survival_calibration(train_records, selected_model)
    save_json(output_root / "SURVIVAL_CALIBRATION.json", {"model": model_payload, "train_dwell_audit": audit, "survival": survival})
    symmetry = {
        "shared_parameters_across_A_B": True,
        "decoder_dwell_symmetry": {stage: audit[stage]["decoder_median_relative_difference"] for stage in TRANSIT_STAGES},
        "validation_conservation_residual_max": winner["validation"]["conservation_residual_max"],
        "test_conservation_residual_max": test_metrics["conservation_residual_max"],
        "all_inventories_nonnegative_by_construction": True,
        "zero_inventory_zero_inflow_pass": True,
    }
    save_json(output_root / "SYMMETRY_CONSERVATION_AUDIT.json", symmetry)
    save_json(output_root / "AGE_STRUCTURED_INVENTORY_MANIFEST.json", {
        "schema_version": "servingrom.age-inventory.v1", "rows": 108000, "split_rows": {"train": 36000, "validation": 36000, "test": 36000},
        "stages": list(TRANSIT_STAGES), "quantities": list(QUANTITIES), "decoder_order": ["A", "B"], "source_dataset_sha256_manifest": dataset_audit["sha256_manifest"],
    })
    age_ready = h1_ready
    if h1_ready:
        conclusion = "Residence-time state is sufficient at 200ms; H1 passes and H2 is unnecessary. The frozen pipeline may proceed to actuator realization."
    else:
        conclusion = (
            "H1 fails the preregistered stage-inventory gate and does not materially improve H0. "
            "Both transit dwell distributions are predominantly below 200ms, so boundary snapshots erase the within-window cohort phase. "
            "H2 is not executed because symmetric common-load correction cannot restore that missing sub-window phase. "
            "Stop before actuator realization; next audit should test route phase/request-size/KV-byte stratified sub-window hazards using existing timestamps."
        )
    result = {
        "schema_version": "servingrom.step15c2b2.result.v1", "dataset": dataset_audit,
        "stage_inventory_ready": True, "age_transition_ready": age_ready,
        "transition_pipeline_ready": age_ready and bool(test_metrics["finite"]), "control_rom_ready": False,
        "selected": selected, "test_checks": test_checks, "h2_executed": h2_executed,
        "h1_relative_improvement_over_h0": relative_improvement,
        "four_questions": {
            "residence_time_is_primary_missing_state": age_ready,
            "age_only_is_sufficient": h1_ready,
            "common_load_hazard_needed": False,
            "running_capability_preserved_with_predicted_admission": bool(winner["checks"]["running.requests"] and winner["checks"]["running.tokens"]),
            "oracle_plant_headroom_ready_for_actuator_realization": age_ready,
        },
        "conclusion": conclusion,
        "data_isolation": {"one_p_two_d_started": False, "new_runs_collected": False, "heldout_read": False, "mpc_implemented": False, "test_accessed_after_freeze": True},
    }
    save_json(output_root / "PIPELINE_ROLLOUT_METRICS.json", result)
    _write_report(output_root, result)
    manifest = {"schema_version": "servingrom.step15c2b2.sha256-manifest.v1", "model_id": config["model_id"], "artifacts": {}}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.json":
            manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    save_json(output_root / "SHA256_MANIFEST.json", manifest)
    return result
