from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from servingrom_modeling.preprocessing import save_json

from .age_transition import _differential_metrics, _load_running_model, _run_factors, _running_rollout
from .pipeline import _load_runs, _sha256, _verify_dataset
from .transition_inventory import QUANTITIES, WINDOW_NS, _decoder, _read, _reconstruct_run, _window_index


@dataclass(frozen=True)
class PhaseRecord:
    run_id: str
    side: int
    request_id: str
    route_ns: int
    enqueue_ns: int
    kv_ready_ns: int
    admission_ns: int
    terminal_ns: int
    input_tokens: float
    expected_tokens: float
    kv_bytes: float
    block_count: float


@dataclass
class LogDelayModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    theta: np.ndarray
    ridge: float

    def predict_ns(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        design = np.column_stack((np.ones(len(normalized)), normalized))
        return np.maximum(np.expm1(design @ self.theta), 0.0) * 1e9


@dataclass
class PhaseKernel:
    handoff: Any
    waiting: Any
    feature_set: str
    ridge: float


@dataclass
class StratifiedDelayModel:
    stage: str
    medians_ns: dict[str, float]
    fallback_ns: float

    def predict_record_ns(self, record: PhaseRecord) -> float:
        workload, load = _run_factors(record.run_id)
        arrival = "burst" if "-on_off_burst-" in record.run_id else "poisson"
        input_bucket = int(record.input_tokens)
        block_bucket = int(record.block_count)
        keys = (
            f"{workload}|{load}|{arrival}|{input_bucket}|{block_bucket}",
            f"{workload}|{load}|{arrival}|{input_bucket}|*",
            f"{workload}|{load}|{arrival}|*|*",
            f"{workload}|{load}|*|*|*",
        )
        return next((self.medians_ns[key] for key in keys if key in self.medians_ns), self.fallback_ns)


@dataclass
class CompositeDelayModel:
    components: tuple[LogDelayModel, ...]

    def predict_ns(self, features: np.ndarray) -> np.ndarray:
        return sum((component.predict_ns(features) for component in self.components), start=np.zeros(len(features)))


def _load_records(run_id: str, outflow_root: Path) -> list[PhaseRecord]:
    root = outflow_root / run_id / "derived"
    attempts = _read(root / "attempt_lifecycle.parquet")
    traces = {str(row["trace_id"]): row for row in _read(root / "trace_lifecycle.parquet") if row.get("trace_id")}
    transfers = {str(row["request_id"]): row for row in _read(root / "kv_transfers.parquet") if row.get("request_id")}
    membership: dict[str, int] = {}
    for row in _read(root / "scheduler_membership.parquet"):
        if _decoder(row.get("component")) is None or not row.get("request_id"):
            continue
        request_id = str(row["request_id"]); timestamp = int(row["ts_wall_ns"])
        membership[request_id] = min(timestamp, membership.get(request_id, timestamp))
    output = []
    for attempt in attempts:
        request_id = str(attempt.get("request_id") or "")
        side = _decoder(attempt.get("decoder_backend")); trace = traces.get(str(attempt.get("trace_id") or "")); transfer = transfers.get(request_id)
        if side is None or trace is None or transfer is None:
            continue
        values = (attempt.get("route_wall_ns"), transfer.get("enqueue_wall_ns"), transfer.get("kv_ready_wall_ns"), membership.get(request_id), trace.get("terminal_wall_ns"))
        if any(value is None for value in values):
            continue
        route, enqueue, kv_ready, admission, terminal = map(int, values)
        if not route <= enqueue <= kv_ready <= admission <= terminal:
            continue
        output.append(PhaseRecord(
            run_id, side, request_id, route, enqueue, kv_ready, admission, terminal,
            float(trace.get("input_tokens") or 0), float(trace.get("expected_output_tokens") or 0),
            float(transfer.get("actual_total_bytes") or 0), float(transfer.get("block_count") or 0),
        ))
    return output


def _phase(timestamp: int) -> float:
    return float(timestamp % WINDOW_NS) / WINDOW_NS


def _features(record: PhaseRecord, run: dict[str, np.ndarray], timestamp: int, feature_set: str) -> np.ndarray:
    if feature_set == "constant":
        return np.empty(0, dtype=np.float64)
    index = _window_index(timestamp, run["starts"])
    running = run["state"][index, :, 2] if index is not None else np.zeros((2, 2))
    phase = _phase(timestamp)
    workload, load = _run_factors(record.run_id)
    burst = 1.0 if "-on_off_burst-" in record.run_id else 0.0
    return np.asarray([
        phase,
        math.sin(2 * math.pi * phase),
        math.cos(2 * math.pi * phase),
        math.log1p(record.input_tokens),
        math.log1p(record.expected_tokens),
        math.log1p(record.kv_bytes),
        math.log1p(record.block_count),
        float(load) / 100.0,
        1.0 if workload == "mixed-bimodal" else 0.0,
        burst,
        math.log1p(running[:, 0].sum()),
        math.log1p(running[:, 1].sum()),
    ], dtype=np.float64)


FEATURE_NAMES = (
    "phase", "phase_sin", "phase_cos", "log_input_tokens", "log_expected_tokens",
    "log_kv_bytes", "log_block_count", "load", "mixed_bimodal", "on_off_burst",
    "log_total_running_requests", "log_total_running_tokens",
)


def _fit_log_delay(records: list[PhaseRecord], runs: dict[str, dict[str, np.ndarray]], stage: str, feature_set: str, ridge: float) -> LogDelayModel:
    timestamps = [record.route_ns if stage != "waiting" else record.kv_ready_ns for record in records]
    x = np.stack([_features(record, runs[record.run_id], timestamp, feature_set) for record, timestamp in zip(records, timestamps)]) if feature_set != "constant" else np.empty((len(records), 0))
    delay_getters = {
        "handoff": lambda record: record.kv_ready_ns - record.route_ns,
        "route_to_enqueue": lambda record: record.enqueue_ns - record.route_ns,
        "transfer": lambda record: record.kv_ready_ns - record.enqueue_ns,
        "waiting": lambda record: record.admission_ns - record.kv_ready_ns,
    }
    delays = np.asarray([delay_getters[stage](record) for record in records], dtype=np.float64) / 1e9
    mean = x.mean(axis=0) if x.shape[1] else np.empty(0); scale = x.std(axis=0) if x.shape[1] else np.empty(0)
    scale[scale < 1e-12] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    target = np.log1p(delays)
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.0
    theta = np.linalg.solve(design.T @ design + ridge * penalty, design.T @ target)
    return LogDelayModel(tuple() if feature_set == "constant" else FEATURE_NAMES, mean, scale, theta, ridge)


def _fit_kernel(records: list[PhaseRecord], runs: dict[str, dict[str, np.ndarray]], feature_set: str, ridge: float) -> PhaseKernel:
    return PhaseKernel(
        _fit_log_delay(records, runs, "handoff", feature_set, ridge),
        _fit_log_delay(records, runs, "waiting", feature_set, ridge),
        feature_set,
        ridge,
    )


def _fit_stratified_delay(records: list[PhaseRecord], stage: str) -> StratifiedDelayModel:
    groups: dict[str, list[float]] = {}
    all_delays = []
    for record in records:
        delay = float((record.kv_ready_ns - record.route_ns) if stage == "handoff" else (record.admission_ns - record.kv_ready_ns))
        workload, load = _run_factors(record.run_id)
        arrival = "burst" if "-on_off_burst-" in record.run_id else "poisson"
        values = (int(record.input_tokens), int(record.block_count))
        keys = (
            f"{workload}|{load}|{arrival}|{values[0]}|{values[1]}",
            f"{workload}|{load}|{arrival}|{values[0]}|*",
            f"{workload}|{load}|{arrival}|*|*",
            f"{workload}|{load}|*|*|*",
        )
        for key in keys:
            groups.setdefault(key, []).append(delay)
        all_delays.append(delay)
    return StratifiedDelayModel(stage, {key: float(np.median(values)) for key, values in groups.items()}, float(np.median(all_delays)))


def _fit_stratified_kernel(records: list[PhaseRecord]) -> PhaseKernel:
    return PhaseKernel(_fit_stratified_delay(records, "handoff"), _fit_stratified_delay(records, "waiting"), "stratified_median", 0.0)


def _fit_decomposed_kernel(records: list[PhaseRecord], runs: dict[str, dict[str, np.ndarray]], ridge: float) -> PhaseKernel:
    handoff = CompositeDelayModel((
        _fit_log_delay(records, runs, "route_to_enqueue", "phase_size_load", ridge),
        _fit_log_delay(records, runs, "transfer", "phase_size_load", ridge),
    ))
    waiting = _fit_log_delay(records, runs, "waiting", "phase_size_load", ridge)
    return PhaseKernel(handoff, waiting, "decomposed_phase_size_load", ridge)


def _predict_delay_ns(model: Any, record: PhaseRecord, run: dict[str, np.ndarray], timestamp: int, feature_set: str) -> float:
    if isinstance(model, StratifiedDelayModel):
        return model.predict_record_ns(record)
    effective_feature_set = "phase_size_load" if feature_set == "decomposed_phase_size_load" else feature_set
    features = _features(record, run, timestamp, effective_feature_set)[None]
    return float(model.predict_ns(features)[0])


def _predict_events(
    run: dict[str, np.ndarray],
    records: list[PhaseRecord],
    kernel: PhaseKernel,
    *,
    oracle_kv: bool = False,
    oracle_waiting_delay: bool = False,
) -> dict[str, np.ndarray]:
    windows = len(run["starts"]); boundaries = np.concatenate((run["starts"], [run["starts"][-1] + WINDOW_NS]))
    flows = {name: np.zeros((windows, 2, 2), dtype=np.float64) for name in ("kv_ready", "admission")}
    state = {stage: np.zeros((windows + 1, 2, 2), dtype=np.float64) for stage in ("handoff", "waiting")}
    for record in records:
        handoff_delay = _predict_delay_ns(kernel.handoff, record, run, record.route_ns, kernel.feature_set)
        predicted_kv = record.kv_ready_ns if oracle_kv else max(record.route_ns, int(record.route_ns + handoff_delay))
        waiting_delay = _predict_delay_ns(kernel.waiting, record, run, predicted_kv, kernel.feature_set)
        if oracle_waiting_delay:
            waiting_delay = float(record.admission_ns - record.kv_ready_ns)
        predicted_admission = max(predicted_kv, int(predicted_kv + waiting_delay))
        mass = np.asarray([1.0, record.expected_tokens])
        for name, timestamp in (("kv_ready", predicted_kv), ("admission", predicted_admission)):
            index = _window_index(timestamp, run["starts"])
            if index is not None:
                flows[name][index, record.side] += mass
        for stage, begin, end in (("handoff", record.route_ns, predicted_kv), ("waiting", predicted_kv, predicted_admission)):
            first = int(np.searchsorted(boundaries, begin, side="left")); last = int(np.searchsorted(boundaries, end, side="left"))
            if first < last:
                state[stage][first:min(last, len(boundaries)), record.side] += mass
    return {**flows, **state}


def _attribution_metrics(runs: list[dict[str, np.ndarray]], records: dict[str, list[PhaseRecord]], kernel: PhaseKernel) -> dict[str, Any]:
    output = {}
    for name, options in {
        "oracle_kv_predicted_waiting": {"oracle_kv": True},
        "predicted_kv_oracle_waiting_delay": {"oracle_waiting_delay": True},
        "oracle_kv_oracle_waiting_delay": {"oracle_kv": True, "oracle_waiting_delay": True},
    }.items():
        errors = {stage: np.zeros(2) for stage in ("handoff", "waiting")}; energies = {stage: np.zeros(2) for stage in errors}
        for run in runs:
            predicted = _predict_events(run, records[run["run_id"]], kernel, **options)
            for stage_index, stage in enumerate(("handoff", "waiting")):
                actual = run["state"][:, :, stage_index]
                for q in range(2):
                    a = actual[:, 0, q] - actual[:, 1, q]; p = predicted[stage][:, 0, q] - predicted[stage][:, 1, q]
                    errors[stage][q] += np.square(p - a).sum(); energies[stage][q] += np.square(a).sum()
        output[name] = {stage: {QUANTITIES[q]: math.sqrt(errors[stage][q] / max(energies[stage][q], 1e-30)) for q in range(2)} for stage in errors}
    return output


def _metric_accumulator() -> dict[str, np.ndarray]:
    return {name: np.zeros(2) for name in ("handoff_error", "handoff_energy", "waiting_error", "waiting_energy", "kv_ready_error", "kv_ready_energy", "admission_error", "admission_energy", "running_error", "running_energy")}


def _evaluate(runs: list[dict[str, np.ndarray]], records: dict[str, list[PhaseRecord]], kernel: PhaseKernel, running_model: Any) -> dict[str, Any]:
    totals = _metric_accumulator(); per_run = []; conservation_max = 0.0; finite = True
    for run in runs:
        predicted = _predict_events(run, records[run["run_id"]], kernel)
        row = {"run_id": run["run_id"], "workload": _run_factors(run["run_id"])[0], "load": _run_factors(run["run_id"])[1], "stages": {}, "flows": {}, "running": {}}
        for stage_index, stage in enumerate(("handoff", "waiting")):
            actual = run["state"][:, :, stage_index]; estimate = predicted[stage]
            row["stages"][stage] = _differential_metrics(actual, estimate)
            for q in range(2):
                a = actual[:, 0, q] - actual[:, 1, q]; p = estimate[:, 0, q] - estimate[:, 1, q]
                totals[f"{stage}_error"][q] += np.square(p - a).sum(); totals[f"{stage}_energy"][q] += np.square(a).sum()
        for name in ("kv_ready", "admission"):
            row["flows"][name] = _differential_metrics(run[name], predicted[name])
            for q in range(2):
                a = run[name][:, 0, q] - run[name][:, 1, q]; p = predicted[name][:, 0, q] - predicted[name][:, 1, q]
                totals[f"{name}_error"][q] += np.square(p - a).sum(); totals[f"{name}_energy"][q] += np.square(a).sum()
        running = _running_rollout(run, predicted["admission"], running_model); actual_running = run["state"][::5, :, 2]
        row["running"] = _differential_metrics(actual_running, running)
        for q in range(2):
            a = actual_running[:, 0, q] - actual_running[:, 1, q]; p = running[:, 0, q] - running[:, 1, q]
            totals["running_error"][q] += np.square(p - a).sum(); totals["running_energy"][q] += np.square(a).sum()
        for index in range(len(run["route"])):
            h = predicted["handoff"][index] + run["route"][index] - predicted["kv_ready"][index]
            w = predicted["waiting"][index] + predicted["kv_ready"][index] - predicted["admission"][index]
            conservation_max = max(conservation_max, float(np.abs(h - predicted["handoff"][index + 1]).max()), float(np.abs(w - predicted["waiting"][index + 1]).max()))
        finite = finite and all(np.isfinite(value).all() for value in predicted.values())
        per_run.append(row)
    def values(name: str) -> dict[str, float]:
        return {QUANTITIES[q]: math.sqrt(totals[f"{name}_error"][q] / max(totals[f"{name}_energy"][q], 1e-30)) for q in range(2)}
    grouped = {}
    for workload in ("balanced", "mixed-bimodal"):
        for load in (55, 75, 92):
            rows = [row for row in per_run if row["workload"] == workload and row["load"] == load]
            grouped[f"{workload}.l{load}"] = {"run_count": len(rows), "stage_mean": {stage: {q: float(np.mean([row["stages"][stage][q] for row in rows])) for q in QUANTITIES} for stage in ("handoff", "waiting")}}
    return {
        "finite": finite,
        "differential_nrmse": {stage: values(stage) for stage in ("handoff", "waiting")},
        "transition_flow_nrmse": {name: values(name) for name in ("kv_ready", "admission")},
        "running": values("running"),
        "conservation_residual_max": conservation_max,
        "workload_load_groups": grouped,
        "per_run": per_run,
    }


def _ready(metrics: dict[str, Any], thresholds: dict[str, float]) -> tuple[bool, dict[str, bool]]:
    stage = metrics["differential_nrmse"]; running = metrics["running"]
    checks = {
        "handoff.requests": stage["handoff"]["requests"] < thresholds["handoff_request_nrmse"],
        "handoff.tokens": stage["handoff"]["tokens"] < thresholds["handoff_token_nrmse"],
        "waiting.requests": stage["waiting"]["requests"] < thresholds["waiting_request_nrmse"],
        "waiting.tokens": stage["waiting"]["tokens"] < thresholds["waiting_token_nrmse"],
        "running.requests": running["requests"] < thresholds["running_request_nrmse"],
        "running.tokens": running["tokens"] < thresholds["running_token_nrmse"],
        "finite": bool(metrics["finite"]), "conservation": metrics["conservation_residual_max"] < 1e-9,
    }
    return all(checks.values()), checks


def _model_json(kernel: PhaseKernel) -> dict[str, Any]:
    def encode(model: Any) -> dict[str, Any]:
        if isinstance(model, StratifiedDelayModel):
            return {"kind": "stratified_median", "stage": model.stage, "medians_ns": model.medians_ns, "fallback_ns": model.fallback_ns}
        if isinstance(model, CompositeDelayModel):
            return {"kind": "sum_of_log_delay_components", "components": [encode(component) for component in model.components]}
        return {"feature_names": list(model.feature_names), "mean": model.mean.tolist(), "scale": model.scale.tolist(), "theta": model.theta.tolist(), "ridge": model.ridge, "target": "log1p(delay_seconds)"}
    return {"schema_version": "servingrom.phase-kernel.v1", "feature_set": kernel.feature_set, "ridge": kernel.ridge, "shared_across_decoders": True, "handoff": encode(kernel.handoff), "waiting": encode(kernel.waiting)}


def _delay_audit(records: list[PhaseRecord]) -> dict[str, Any]:
    output = {"schema_version": "servingrom.subwindow-delay-audit.v1", "split": "train", "count": len(records)}
    for name, values in {
        "route_to_enqueue_ms": [(r.enqueue_ns - r.route_ns) / 1e6 for r in records],
        "transfer_ms": [(r.kv_ready_ns - r.enqueue_ns) / 1e6 for r in records],
        "handoff_ms": [(r.kv_ready_ns - r.route_ns) / 1e6 for r in records],
        "waiting_ms": [(r.admission_ns - r.kv_ready_ns) / 1e6 for r in records],
    }.items():
        output[name] = dict(zip(("p50", "p75", "p90", "p95", "p99", "max"), map(float, np.percentile(values, [50, 75, 90, 95, 99, 100]))))
    output["available_features"] = ["route_phase", "kv_ready_phase", *FEATURE_NAMES[3:]]
    return output


def _write_report(root: Path, result: dict[str, Any]) -> None:
    m = result["selected"]["validation"]
    lines = ["# ServingROM Step 15C-2B.3 Phase-Conditioned Transition Kernel", "", "## 状态", "", f"- `phase_transition_ready={str(result['phase_transition_ready']).lower()}`", f"- `transition_pipeline_ready={str(result['transition_pipeline_ready']).lower()}`", "- `control_rom_ready=false`", "", "## Validation", "", "| Stage | Request NRMSE | Token NRMSE |", "|---|---:|---:|"]
    for stage in ("handoff", "waiting"):
        lines.append(f"| {stage} | {m['differential_nrmse'][stage]['requests']:.6f} | {m['differential_nrmse'][stage]['tokens']:.6f} |")
    lines += ["", f"- running request/token: `{m['running']['requests']:.6f}` / `{m['running']['tokens']:.6f}`", "", "## 结论", "", result["conclusion"]]
    (root / "STEP15C2B3_PHASE_CONDITIONED_TRANSITION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_transition_pipeline(dataset_root: Path, forcing_root: Path, outflow_root: Path, transition_root: Path, age_root: Path, output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    dataset_root, forcing_root, outflow_root, transition_root, age_root, output_root = (path.resolve() for path in (dataset_root, forcing_root, outflow_root, transition_root, age_root, output_root))
    output_root.mkdir(parents=True, exist_ok=True); dataset = _verify_dataset(dataset_root, config["dataset_id"]); ranges = _load_runs(dataset_root)
    runs: dict[str, list[dict[str, np.ndarray]]] = {"train": []}; records: dict[str, list[PhaseRecord]] = {}
    for rr in ranges["train"]:
        run, _ = _reconstruct_run(rr.run_id, forcing_root, outflow_root); run["run_id"] = rr.run_id; runs["train"].append(run); records[rr.run_id] = _load_records(rr.run_id, outflow_root)
    run_map = {run["run_id"]: run for run in runs["train"]}; train_records = [record for run in runs["train"] for record in records[run["run_id"]]]
    audit = _delay_audit(train_records); save_json(output_root / "SUBWINDOW_DELAY_AUDIT.json", audit)
    save_json(output_root / "PHASE_FEATURE_SCHEMA.json", {"schema_version": "servingrom.phase-feature-schema.v1", "selected_from": "train_only", "features": list(FEATURE_NAMES), "a_b_signed_features": [], "time_alignment": "[start_wall_ns,end_wall_ns)", "oracle_input": "observed route timestamp and routed request/token forcing"})
    runs["validation"] = []
    for rr in ranges["validation"]:
        run, _ = _reconstruct_run(rr.run_id, forcing_root, outflow_root); run["run_id"] = rr.run_id; runs["validation"].append(run); records[rr.run_id] = _load_records(rr.run_id, outflow_root)
    run_map.update({run["run_id"]: run for run in runs["validation"]}); running_model = _load_running_model(transition_root / "models/transition_service_flow_model.npz")
    scan = []
    for feature_set in ("constant", "phase_size_load"):
        for ridge in map(float, config["candidate_ridges"]):
            kernel = _fit_kernel(train_records, run_map, feature_set, ridge); metrics = _evaluate(runs["validation"], records, kernel, running_model)
            score = float(np.mean([metrics["differential_nrmse"][stage][q] for stage in ("handoff", "waiting") for q in QUANTITIES])); ready, checks = _ready(metrics, config["readiness"])
            scan.append({"feature_set": feature_set, "ridge": ridge, "score": score, "ready": ready, "checks": checks, "validation": metrics, "kernel": kernel})
    stratified = _fit_stratified_kernel(train_records)
    metrics = _evaluate(runs["validation"], records, stratified, running_model)
    score = float(np.mean([metrics["differential_nrmse"][stage][q] for stage in ("handoff", "waiting") for q in QUANTITIES])); ready, checks = _ready(metrics, config["readiness"])
    scan.append({"feature_set": "stratified_median", "ridge": 0.0, "score": score, "ready": ready, "checks": checks, "validation": metrics, "kernel": stratified})
    for ridge in map(float, config["candidate_ridges"]):
        decomposed = _fit_decomposed_kernel(train_records, run_map, ridge)
        metrics = _evaluate(runs["validation"], records, decomposed, running_model)
        score = float(np.mean([metrics["differential_nrmse"][stage][q] for stage in ("handoff", "waiting") for q in QUANTITIES])); ready, checks = _ready(metrics, config["readiness"])
        scan.append({"feature_set": "decomposed_phase_size_load", "ridge": ridge, "score": score, "ready": ready, "checks": checks, "validation": metrics, "kernel": decomposed})
    winner = min(scan, key=lambda row: row["score"]); kernel = winner["kernel"]
    frozen = {"schema_version": "servingrom.phase-selection.v1", "selection_split": "validation", "test_accessed": False, "feature_set": winner["feature_set"], "ridge": winner["ridge"], "validation_score": winner["score"], "validation_ready": winner["ready"]}
    save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen)
    runs["test"] = []
    for rr in ranges["test"]:
        run, _ = _reconstruct_run(rr.run_id, forcing_root, outflow_root); run["run_id"] = rr.run_id; runs["test"].append(run); records[rr.run_id] = _load_records(rr.run_id, outflow_root)
    test = _evaluate(runs["test"], records, kernel, running_model); test_ready, test_checks = _ready(test, config["readiness"])
    attribution = {
        "validation": _attribution_metrics(runs["validation"], records, kernel),
        "test": _attribution_metrics(runs["test"], records, kernel),
        "not_used_for_selection": True,
    }
    save_json(output_root / "TRANSITION_ERROR_ATTRIBUTION.json", attribution)
    selected = {"feature_set": winner["feature_set"], "ridge": winner["ridge"], "validation": winner["validation"], "test": test, "validation_ready": winner["ready"], "test_ready": test_ready}
    age_metrics = json.loads((age_root / "PIPELINE_ROLLOUT_METRICS.json").read_text(encoding="utf-8"))["selected"]
    ablation = {"H1_age_only": {"validation": age_metrics["validation"], "test": age_metrics["test"]}, "phase_scan": [{k: v for k, v in row.items() if k != "kernel"} for row in scan], "selected": selected}
    save_json(output_root / "PHASE_KERNEL_ABLATION.json", ablation); save_json(output_root / "models/phase_transition_kernel.json", _model_json(kernel))
    audit_result = {"shared_parameters_A_B": True, "signed_A_B_features": [], "validation_conservation_residual_max": winner["validation"]["conservation_residual_max"], "test_conservation_residual_max": test["conservation_residual_max"], "all_runs_finite": bool(winner["validation"]["finite"] and test["finite"]), "test_checks": test_checks}
    save_json(output_root / "SYMMETRY_CONSERVATION_AUDIT.json", audit_result)
    phase_ready = bool(winner["ready"])
    result = {"schema_version": "servingrom.step15c2b3.result.v1", "dataset": dataset, "phase_transition_ready": phase_ready, "transition_pipeline_ready": phase_ready, "control_rom_ready": False, "selected": selected, "error_attribution": attribution, "conclusion": "Phase-conditioned sub-window timing recovers the transit dynamics and preserves running propagation; the plant-side pipeline has headroom for actuator realization." if phase_ready else "Phase-conditioned timing remains below the preregistered gate. Stop before actuator realization; use the frozen oracle attribution to identify whether KV timing propagation or waiting-service timing dominates.", "data_isolation": {"one_p_two_d_started": False, "new_runs_collected": False, "heldout_read": False, "test_accessed_after_freeze": True}}
    save_json(output_root / "PIPELINE_ROLLOUT_METRICS.json", result); _write_report(output_root, result)
    manifest = {"schema_version": "servingrom.step15c2b3.sha256-manifest.v1", "model_id": config["model_id"], "artifacts": {}}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.json": manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    save_json(output_root / "SHA256_MANIFEST.json", manifest)
    return result
