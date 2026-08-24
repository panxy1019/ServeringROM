from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from servingrom_modeling.preprocessing import save_json

from .pipeline import _load_json, _load_runs, _nrmse, _sha256, _verify_dataset


STAGES = ("handoff", "waiting", "running")
QUANTITIES = ("requests", "tokens")
WINDOW_NS = 200_000_000
FLOW_GROUPS = ("kv_ready", "admission", "running_outflow")


@dataclass
class SharedFlowModel:
    ridge: float
    theta: dict[str, np.ndarray]
    feature_scale: dict[str, np.ndarray]
    target_scale: dict[str, np.ndarray]

    def predict(self, group: str, source: np.ndarray, inflow: np.ndarray) -> np.ndarray:
        total_source = source.sum(axis=0, keepdims=True).repeat(2, axis=0)
        total_inflow = inflow.sum(axis=0, keepdims=True).repeat(2, axis=0)
        features = np.concatenate((source, inflow, total_source, total_inflow), axis=1)
        prediction = (features / self.feature_scale[group]) @ self.theta[group]
        prediction *= self.target_scale[group]
        return np.clip(prediction, 0.0, np.maximum(source + inflow, 0.0))


def _decoder(value: Any) -> int | None:
    text = str(value or "")
    if text.endswith(":13701") or text == "decode-0":
        return 0
    if text.endswith(":13702") or text == "decode-1":
        return 1
    return None


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _boundary_index(timestamp: int, starts: np.ndarray) -> int:
    return int(np.searchsorted(starts, int(timestamp), side="left"))


def _post_event_boundary_index(timestamp: int, starts: np.ndarray) -> int:
    """Return the first boundary whose snapshot includes an event at timestamp."""
    return int(np.searchsorted(starts, int(timestamp), side="right"))


def _window_index(timestamp: Any, starts: np.ndarray) -> int | None:
    if timestamp is None:
        return None
    value = int(timestamp)
    index = int((value - int(starts[0])) // WINDOW_NS)
    if 0 <= index < len(starts) and starts[index] <= value < starts[index] + WINDOW_NS:
        return index
    return None


def _remaining_before(emissions: list[tuple[int, float]], timestamp: int, expected: float) -> float:
    return max(expected - sum(count for event_ts, count in emissions if event_ts < timestamp), 0.0)


def _stage(timestamp: int, route: int, kv_ready: int, admission: int, terminal: int) -> int | None:
    if route <= timestamp < kv_ready:
        return 0
    if kv_ready <= timestamp < admission:
        return 1
    if admission <= timestamp < terminal:
        return 2
    return None


def _add_interval(
    count_delta: np.ndarray,
    token_delta: np.ndarray,
    side: int,
    stage: int,
    start_ts: int,
    end_ts: int,
    expected: float,
    emissions: list[tuple[int, float]],
    starts: np.ndarray,
) -> None:
    begin = min(max(_boundary_index(start_ts, starts), 0), len(starts))
    end = min(max(_boundary_index(end_ts, starts), 0), len(starts))
    if begin >= end:
        return
    remaining_start = _remaining_before(emissions, int(starts[begin]), expected)
    remaining_end = _remaining_before(emissions, int(starts[end]) if end < len(starts) else end_ts, expected)
    count_delta[begin, side, stage] += 1.0
    count_delta[end, side, stage] -= 1.0
    token_delta[begin, side, stage] += remaining_start
    token_delta[end, side, stage] -= remaining_end
    for event_ts, count in emissions:
        if event_ts < int(starts[begin]):
            continue
        boundary = _post_event_boundary_index(event_ts, starts)
        if begin <= boundary <= end:
            token_delta[boundary, side, stage] -= count


def _reconstruct_run(
    run_id: str,
    forcing_root: Path,
    outflow_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    controls = _read(forcing_root / run_id / "derived/control/control_windows.parquet")
    starts = np.asarray([int(row["start_wall_ns"]) for row in controls], dtype=np.int64)
    if len(starts) != 3000 or np.any(np.diff(starts) != WINDOW_NS):
        raise ValueError(f"invalid 200ms windows: {run_id}")
    boundaries = np.concatenate((starts, [starts[-1] + WINDOW_NS]))
    root = outflow_root / run_id / "derived"
    attempts = _read(root / "attempt_lifecycle.parquet")
    traces = _read(root / "trace_lifecycle.parquet")
    transfers = _read(root / "kv_transfers.parquet")
    memberships = _read(root / "scheduler_membership.parquet")
    emissions_rows = _read(root / "token_emissions.parquet")

    trace_by_id = {str(row["trace_id"]): row for row in traces if row.get("trace_id")}
    transfer_by_request = {str(row["request_id"]): row for row in transfers if row.get("request_id")}
    first_membership: dict[str, int] = {}
    for row in memberships:
        if _decoder(row.get("component")) is None or not row.get("request_id"):
            continue
        request_id = str(row["request_id"]); timestamp = int(row["ts_wall_ns"])
        first_membership[request_id] = min(timestamp, first_membership.get(request_id, timestamp))
    emissions: dict[str, list[tuple[int, float]]] = {}
    for row in emissions_rows:
        # Prefill emits one internal handoff token that is emitted again by the
        # selected Decode engine. Only Decode emissions represent client-visible
        # service and therefore decrement the three-stage output-token inventory.
        if _decoder(row.get("component")) is None or not row.get("request_id"):
            continue
        emissions.setdefault(str(row["request_id"]), []).append((int(row["ts_wall_ns"]), float(row.get("new_token_count") or 0)))
    for rows in emissions.values():
        rows.sort()

    count_delta = np.zeros((len(boundaries) + 1, 2, 3), dtype=np.float64)
    token_delta = np.zeros_like(count_delta)
    flows = {
        name: np.zeros((len(starts), 2, 2), dtype=np.float64)
        for name in ("route", "kv_ready", "admission", "service_handoff", "service_waiting", "service_running", "terminal")
    }
    coverage = {
        "attempts_total": len(attempts),
        "pd_routed_attempts": 0,
        "ignored_without_pd_route": 0,
        "complete_lifecycle": 0,
        "missing_trace": 0,
        "missing_kv_ready": 0,
        "missing_admission": 0,
        "invalid_stage_order": 0,
        "excluded_prefill_internal_token_events": sum(
            _decoder(row.get("component")) is None for row in emissions_rows
        ),
    }
    for attempt in attempts:
        request_id = str(attempt.get("request_id") or "")
        side = _decoder(attempt.get("decoder_backend"))
        trace = trace_by_id.get(str(attempt.get("trace_id") or ""))
        transfer = transfer_by_request.get(request_id)
        route = attempt.get("route_wall_ns")
        kv_ready = transfer.get("kv_ready_wall_ns") if transfer else None
        admission = first_membership.get(request_id)
        terminal = trace.get("terminal_wall_ns") if trace else None
        if side is None or route is None:
            coverage["ignored_without_pd_route"] += 1
            continue
        coverage["pd_routed_attempts"] += 1
        if trace is None or terminal is None:
            coverage["missing_trace"] += 1; continue
        if kv_ready is None:
            coverage["missing_kv_ready"] += 1; continue
        if admission is None:
            coverage["missing_admission"] += 1; continue
        route, kv_ready, admission, terminal = map(int, (route, kv_ready, admission, terminal))
        if not (route <= kv_ready <= admission <= terminal):
            coverage["invalid_stage_order"] += 1
            continue
        expected = float(trace.get("expected_output_tokens") or 0)
        request_emissions = emissions.get(request_id, [])
        coverage["complete_lifecycle"] += 1
        intervals = ((0, route, kv_ready), (1, kv_ready, admission), (2, admission, terminal))
        for stage_index, begin, end in intervals:
            _add_interval(count_delta, token_delta, side, stage_index, begin, end, expected, request_emissions, boundaries)
        transitions = (("route", route), ("kv_ready", kv_ready), ("admission", admission))
        for name, timestamp in transitions:
            index = _window_index(timestamp, starts)
            if index is not None:
                flows[name][index, side, 0] += 1.0
                flows[name][index, side, 1] += _remaining_before(request_emissions, timestamp, expected)
        for timestamp, count in request_emissions:
            stage_index = _stage(timestamp, route, kv_ready, admission, terminal)
            if stage_index is None:
                continue
            index = _window_index(timestamp, starts)
            if index is not None:
                flows[("service_handoff", "service_waiting", "service_running")[stage_index]][index, side, 1] += count
        index = _window_index(terminal, starts)
        if index is not None:
            flows["terminal"][index, side, 0] += 1.0
            flows["terminal"][index, side, 1] += _remaining_before(request_emissions, terminal, expected)

    requests = np.cumsum(count_delta, axis=0)[:len(boundaries)]
    tokens = np.cumsum(token_delta, axis=0)[:len(boundaries)]
    state = np.stack((requests, tokens), axis=-1)
    output = {"state": state, **flows, "starts": starts}
    coverage["complete_fraction"] = coverage["complete_lifecycle"] / max(coverage["pd_routed_attempts"], 1)
    return output, coverage


def _replay(run: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    state = run["state"]
    prediction = np.empty_like(state)
    prediction[0] = state[0]
    residuals = {stage: np.zeros((len(state) - 1, 2, 2), dtype=np.float64) for stage in STAGES}
    for index in range(len(state) - 1):
        h = prediction[index, :, 0] + run["route"][index] - run["kv_ready"][index] - run["service_handoff"][index]
        w = prediction[index, :, 1] + run["kv_ready"][index] - run["admission"][index] - run["service_waiting"][index]
        r = prediction[index, :, 2] + run["admission"][index] - run["service_running"][index] - run["terminal"][index]
        prediction[index + 1] = np.stack((h, w, r), axis=1)
        for stage_index, stage in enumerate(STAGES):
            residuals[stage][index] = state[index + 1, :, stage_index] - prediction[index + 1, :, stage_index]
    return prediction, residuals


def _replay_metrics(runs: dict[str, list[dict[str, np.ndarray]]]) -> dict[str, Any]:
    output = {}
    for split, split_runs in runs.items():
        error = {(stage, quantity): 0.0 for stage in STAGES for quantity in range(2)}
        energy = dict(error); exact = dict(error); count = dict(error)
        for run in split_runs:
            prediction, _ = _replay(run)
            actual = run["state"]
            for stage_index, stage in enumerate(STAGES):
                for quantity in range(2):
                    diff = prediction[1:, :, stage_index, quantity] - actual[1:, :, stage_index, quantity]
                    error[(stage, quantity)] += float(np.square(diff).sum())
                    energy[(stage, quantity)] += float(np.square(actual[1:, :, stage_index, quantity]).sum())
                    exact[(stage, quantity)] += float(np.sum(np.abs(diff) < 1e-9))
                    count[(stage, quantity)] += diff.size
        output[split] = {
            stage: {
                QUANTITIES[q]: {
                    "nrmse": math.sqrt(error[(stage, q)] / max(energy[(stage, q)], 1e-30)),
                    "exact_fraction": exact[(stage, q)] / max(count[(stage, q)], 1.0),
                }
                for q in range(2)
            }
            for stage in STAGES
        }
    return output


def _observed_gate(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {}
    for split in ("train", "validation"):
        for stage in STAGES:
            for quantity in QUANTITIES:
                row = metrics[split][stage][quantity]
                threshold = config["maximum_request_replay_nrmse"] if quantity == "requests" else config["maximum_token_replay_nrmse"]
                exact = config["minimum_request_exact_fraction"] if quantity == "requests" else config["minimum_token_exact_fraction"]
                checks[f"{split}.{stage}.{quantity}.nrmse"] = row["nrmse"] < float(threshold)
                checks[f"{split}.{stage}.{quantity}.exact"] = row["exact_fraction"] >= float(exact)
    return all(checks.values()), checks


def _aggregate(run: dict[str, np.ndarray], factor: int = 5) -> dict[str, np.ndarray]:
    windows = (len(run["state"]) - 1) // factor
    result = {"state": run["state"][np.arange(windows + 1) * factor]}
    for name in ("route", "kv_ready", "admission", "service_handoff", "service_waiting", "service_running", "terminal"):
        result[name] = run[name][:windows * factor].reshape(windows, factor, 2, 2).sum(axis=1)
    return result


def _flow_samples(runs: list[dict[str, np.ndarray]], group: str) -> tuple[np.ndarray, np.ndarray]:
    features = []; targets = []
    mapping = {
        "kv_ready": (0, "route", "kv_ready"),
        "admission": (1, "kv_ready", "admission"),
        "running_outflow": (2, "admission", None),
    }
    stage, inflow_name, target_name = mapping[group]
    for run in runs:
        for index in range(len(run["state"]) - 1):
            source = run["state"][index, :, stage]
            inflow = run[inflow_name][index]
            total_source = source.sum(axis=0, keepdims=True).repeat(2, axis=0)
            total_inflow = inflow.sum(axis=0, keepdims=True).repeat(2, axis=0)
            features.append(np.concatenate((source, inflow, total_source, total_inflow), axis=1))
            if target_name:
                target = run[target_name][index]
            else:
                target = run["service_running"][index] + run["terminal"][index]
            targets.append(target)
    return np.concatenate(features), np.concatenate(targets)


def _fit_flow_models(runs: list[dict[str, np.ndarray]], ridge: float) -> SharedFlowModel:
    theta = {}; feature_scale = {}; target_scale = {}
    for group in FLOW_GROUPS:
        x, y = _flow_samples(runs, group)
        xs = x.std(axis=0); xs[xs < 1e-12] = 1.0
        ys = y.std(axis=0); ys[ys < 1e-12] = 1.0
        xn, yn = x / xs, y / ys
        theta[group] = np.linalg.solve(xn.T @ xn + ridge * np.eye(xn.shape[1]), xn.T @ yn)
        feature_scale[group] = xs; target_scale[group] = ys
    return SharedFlowModel(ridge, theta, feature_scale, target_scale)


def _flow_rollout(model: SharedFlowModel, runs: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    error = np.zeros((3, 2)); energy = np.zeros((3, 2)); per_run = []; finite = True
    for run_number, run in enumerate(runs):
        actual = run["state"]; predicted = np.empty_like(actual); predicted[0] = actual[0]
        for index in range(len(actual) - 1):
            route = run["route"][index]
            kv = model.predict("kv_ready", predicted[index, :, 0], route)
            handoff = np.maximum(predicted[index, :, 0] + route - kv, 0.0)
            admission = model.predict("admission", predicted[index, :, 1], kv)
            waiting = np.maximum(predicted[index, :, 1] + kv - admission, 0.0)
            outflow = model.predict("running_outflow", predicted[index, :, 2], admission)
            running = np.maximum(predicted[index, :, 2] + admission - outflow, 0.0)
            predicted[index + 1] = np.stack((handoff, waiting, running), axis=1)
            if not np.isfinite(predicted[index + 1]).all():
                finite = False; break
        diff_actual = actual[:, 0] - actual[:, 1]
        diff_predicted = predicted[:, 0] - predicted[:, 1]
        err = diff_predicted - diff_actual
        error += np.square(err).sum(axis=0); energy += np.square(diff_actual).sum(axis=0)
        per_run.append({
            "run_number": run_number,
            "nrmse": {stage: {QUANTITIES[q]: _nrmse(diff_actual[:, s:s + 1, q], diff_predicted[:, s:s + 1, q], np.zeros(1)) for q in range(2)} for s, stage in enumerate(STAGES)},
        })
    return {
        "finite": finite,
        "differential_nrmse": {stage: {QUANTITIES[q]: math.sqrt(error[s, q] / max(energy[s, q], 1e-30)) for q in range(2)} for s, stage in enumerate(STAGES)},
        "runs": per_run,
    }


def _save_model(path: Path, model: SharedFlowModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ridge": np.asarray(model.ridge)}
    for group in FLOW_GROUPS:
        payload[f"theta_{group}"] = model.theta[group]
        payload[f"feature_scale_{group}"] = model.feature_scale[group]
        payload[f"target_scale_{group}"] = model.target_scale[group]
    np.savez_compressed(path, **payload)


def _write_report(output_root: Path, result: dict[str, Any]) -> None:
    replay = result["observed_flow_replay"]["validation"]
    lines = [
        "# ServingROM Step 15C-2B.1 三阶段库存与转移流重构",
        "", "## 状态", "",
        f"- `observed_flow_conservation_pass={str(result['observed_flow_conservation_pass']).lower()}`",
        f"- `transition_flow_model_trained={str(result['transition_flow_model_trained']).lower()}`",
        "- `control_rom_ready=false`",
        "- 未启动 1P2D、未重新采集、未读取 held-out、未实现 MPC。",
        "", "## Validation Observed-Flow Replay", "",
        "| 阶段 | request NRMSE | request exact | token NRMSE | token exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        lines.append(f"| {stage} | {replay[stage]['requests']['nrmse']:.6f} | {replay[stage]['requests']['exact_fraction']:.2%} | {replay[stage]['tokens']['nrmse']:.6f} | {replay[stage]['tokens']['exact_fraction']:.2%} |")
    if result["transition_flow_model_trained"]:
        lines += ["", "## Transition/Service Flow Model", ""]
        for split in ("validation", "test"):
            row = result["flow_model"][split]["differential_nrmse"]
            lines.append(f"### {split}")
            for stage in STAGES:
                lines.append(f"- `{stage}` requests/tokens：`{row[stage]['requests']:.6f}` / `{row[stage]['tokens']:.6f}`")
    lines += [
        "", "## 缺失字段与下一步", "",
        f"- `{result['missing_field_audit']['conclusion']}`",
        f"- `{result['next_step']}`",
    ]
    (output_root / "STEP15C2B1_TRANSITION_INVENTORY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_transition_inventory_pipeline(
    dataset_root: Path,
    forcing_root: Path,
    outflow_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    dataset_root, forcing_root, outflow_root, output_root = (p.resolve() for p in (dataset_root, forcing_root, outflow_root, output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_audit = _verify_dataset(dataset_root, config["dataset_id"])
    run_ranges = _load_runs(dataset_root)
    runtime: dict[str, list[dict[str, np.ndarray]]] = {split: [] for split in ("train", "validation")}
    coverage = {}; sidecar_rows = []

    def append_sidecar(split: str, run_id: str, reconstructed: dict[str, np.ndarray]) -> None:
        state = reconstructed["state"]
        for index in range(len(state) - 1):
            row = {"split": split, "run_id": run_id, "window_id": index, "start_wall_ns": int(reconstructed["starts"][index])}
            for side, side_name in enumerate(("A", "B")):
                for stage_index, stage in enumerate(STAGES):
                    row[f"{stage}_{side_name}_requests"] = float(state[index, side, stage_index, 0])
                    row[f"{stage}_{side_name}_tokens"] = float(state[index, side, stage_index, 1])
            sidecar_rows.append(row)

    for split in ("train", "validation"):
        for run in run_ranges[split]:
            reconstructed, audit = _reconstruct_run(run.run_id, forcing_root, outflow_root)
            runtime[split].append(reconstructed); coverage[run.run_id] = audit
            append_sidecar(split, run.run_id, reconstructed)
    replay = _replay_metrics(runtime)
    replay_pass, replay_checks = _observed_gate(replay, config)
    missing = {
        "schema_version": "servingrom.transition-inventory-missing-fields.v1",
        "coverage": coverage,
        "missing_fields": [],
        "conclusion": "existing sealed telemetry is sufficient for three-stage conservation replay" if replay_pass else "one or more stage boundaries do not close; inspect failed replay checks before modeling",
        "failed_checks": [name for name, passed in replay_checks.items() if not passed],
    }
    save_json(output_root / "OBSERVED_FLOW_REPLAY.json", {"metrics": replay, "checks": replay_checks, "pass": replay_pass})
    save_json(output_root / "MISSING_FIELD_AUDIT.json", missing)
    save_json(output_root / "STAGE_INVENTORY_MANIFEST.json", {
        "schema_version": "servingrom.stage-inventory.v1", "fast_window_ms": 200,
        "stages": list(STAGES), "quantities": list(QUANTITIES),
        "boundaries": {"handoff": ["p_to_d_route", "kv_ready"], "waiting": ["kv_ready", "first_decode_scheduler_membership"], "running": ["first_decode_scheduler_membership", "terminal"]},
        "token_semantics": "expected_output_tokens minus cumulative emissions strictly before each boundary",
        "a_b_symmetry": "same construction for both decoders; differential=A-B",
    })

    flow_result = None; selected_model = None; frozen_selection = {"test_accessed": False, "selection_split": "validation", "observed_flow_conservation_pass": replay_pass}
    if replay_pass:
        aggregate = {split: [_aggregate(run) for run in runtime[split]] for split in runtime}
        scan = []
        for ridge in [float(value) for value in config["candidate_ridges"]]:
            model = _fit_flow_models(aggregate["train"], ridge)
            metrics = _flow_rollout(model, aggregate["validation"])
            score = float(np.mean([metrics["differential_nrmse"][stage][quantity] for stage in STAGES for quantity in QUANTITIES]))
            scan.append({"ridge": ridge, "score": score, "validation": metrics, "model": model})
        winner = min(scan, key=lambda row: row["score"])
        selected_model = winner["model"]
        frozen_selection.update({"ridge": winner["ridge"], "validation_score": winner["score"], "test_accessed": False})
        save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen_selection)
        test_runtime = []
        for run in run_ranges["test"]:
            reconstructed, audit = _reconstruct_run(run.run_id, forcing_root, outflow_root)
            test_runtime.append(reconstructed); coverage[run.run_id] = audit
            append_sidecar("test", run.run_id, reconstructed)
        test_replay = _replay_metrics({"test": test_runtime})["test"]
        replay["test"] = test_replay
        test_metrics = _flow_rollout(selected_model, [_aggregate(run) for run in test_runtime])
        flow_result = {
            "ridge_scan": [{key: value for key, value in row.items() if key != "model"} for row in scan],
            "selected_ridge": winner["ridge"], "validation": winner["validation"], "test": test_metrics,
        }
        _save_model(output_root / "models/transition_service_flow_model.npz", selected_model)
        missing["coverage"] = coverage
        save_json(output_root / "MISSING_FIELD_AUDIT.json", missing)
    else:
        save_json(output_root / "FROZEN_SELECTION_BEFORE_TEST.json", frozen_selection)

    sidecar = output_root / "sidecar/stage_inventory_200ms.parquet"; sidecar.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sidecar_rows), sidecar, compression="zstd")
    result = {
        "schema_version": "servingrom.step15c2b1.result.v1", "dataset": dataset_audit,
        "observed_flow_replay": replay, "observed_flow_conservation_pass": replay_pass,
        "transition_flow_model_trained": replay_pass, "flow_model": flow_result,
        "selection": {**frozen_selection, "test_accessed": bool(replay_pass)},
        "missing_field_audit": missing, "control_rom_ready": False,
        "next_step": "evaluate stage-flow model and redesign only failed transition heads" if replay_pass else "stop and instrument missing stage boundary fields",
        "data_isolation": {"one_p_two_d_started": False, "new_runs_collected": False, "heldout_read": False, "test_accessed_after_validation_freeze": bool(replay_pass)},
    }
    save_json(output_root / "TRANSITION_FLOW_ABLATION.json", flow_result or {"skipped": True, "reason": "observed replay failed"})
    save_json(output_root / "evaluation/final_metrics.json", result)
    _write_report(output_root, result)
    manifest = {"schema_version": "servingrom.step15c2b1.sha256-manifest.v1", "model_id": config["model_id"], "dataset_sha256_manifest": dataset_audit["sha256_manifest"], "artifacts": {}}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256_MANIFEST.json":
            manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
    save_json(output_root / "SHA256_MANIFEST.json", manifest)
    return result
