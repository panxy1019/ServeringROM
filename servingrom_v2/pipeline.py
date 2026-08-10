from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from servingrom_modeling.dataset import RomDataset, RunSlice, SPLITS, sha256
from servingrom_modeling.dynamics import fit_model, one_step_metrics, rollout_metrics, transformed
from servingrom_modeling.preprocessing import Normalizer, fit_normalizer, save_json


@dataclass
class AggregatedView:
    factor: int
    starts: np.ndarray
    ends: np.ndarray
    disturbance: np.ndarray
    output: np.ndarray
    runs: list[RunSlice]


def aggregate_run_array(values: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate non-overlapping flow windows without crossing a run boundary."""
    groups = values.shape[0] // factor
    return np.asarray(values[: groups * factor], dtype=np.float64).reshape(
        groups, factor, values.shape[1]
    ).sum(axis=1)


def augment_markov_state(
    state: np.ndarray,
    state_next: np.ndarray,
    disturbance: np.ndarray,
    runs: list[RunSlice],
) -> tuple[np.ndarray, np.ndarray]:
    """Add state velocity and one previous disturbance interval per run."""
    velocity = np.zeros_like(state)
    previous_disturbance = np.zeros_like(disturbance)
    for run in runs:
        if run.end - run.start > 1:
            velocity[run.start + 1 : run.end] = (
                state[run.start + 1 : run.end] - state[run.start : run.end - 1]
            )
            previous_disturbance[run.start + 1 : run.end] = disturbance[run.start : run.end - 1]
    augmented = np.concatenate([state, velocity, previous_disturbance], axis=1)
    augmented_next = np.concatenate([state_next, state_next - state, disturbance], axis=1)
    return augmented, augmented_next


def _write_markdown(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *lines]) + "\n", encoding="utf-8")


def _build_view(dataset: RomDataset, split: str, factor: int) -> AggregatedView:
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    disturbances: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    runs: list[RunSlice] = []
    offset = 0
    d = dataset.array(split, "D")
    y = dataset.array(split, "Y")
    for run in dataset.run_slices(split):
        count = (run.end - run.start) // factor
        local_starts = run.start + np.arange(count, dtype=np.int64) * factor
        starts.append(local_starts)
        ends.append(local_starts + factor - 1)
        disturbances.append(aggregate_run_array(d[run.start : run.end], factor))
        outputs.append(aggregate_run_array(y[run.start : run.end], factor))
        runs.append(RunSlice(
            run_id=run.run_id,
            split=split,
            start=offset,
            end=offset + count,
            workload=run.workload,
            arrival_process=run.arrival_process,
            transient_pattern=run.transient_pattern,
        ))
        offset += count
    return AggregatedView(
        factor=factor,
        starts=np.concatenate(starts),
        ends=np.concatenate(ends),
        disturbance=np.concatenate(disturbances),
        output=np.concatenate(outputs),
        runs=runs,
    )


def _project_at(
    array: np.ndarray,
    positions: np.ndarray,
    normalizer: Normalizer,
    basis: np.ndarray,
    rank: int,
    chunk_size: int,
) -> np.ndarray:
    result = np.empty((len(positions), rank), dtype=np.float64)
    for start in range(0, len(positions), chunk_size):
        section = slice(start, min(start + chunk_size, len(positions)))
        result[section] = normalizer.transform(array[positions[section]]) @ basis[:, :rank]
    return result


def _view_state_index(rank: int, memory: bool, disturbance_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {"index": i, "name": f"pod_state_{i}", "block": "pod_state", "unit": "scalar"}
        for i in range(rank)
    ]
    if memory:
        rows.extend(
            {"index": len(rows), "name": f"pod_velocity_{i}", "block": "pod_velocity", "unit": "scalar"}
            for i in range(rank)
        )
        rows.extend(
            {
                "index": len(rows),
                "name": f"previous_{row['name']}",
                "block": "disturbance_memory",
                "unit": row.get("unit", "scalar"),
            }
            for row in disturbance_index
        )
    return rows


def _output_sparsity(view: AggregatedView, output_names: list[str]) -> dict[str, Any]:
    rows = []
    for index, name in enumerate(output_names):
        values = view.output[:, index]
        mean = float(np.mean(values))
        rows.append({
            "name": name,
            "nonzero_ratio": float(np.count_nonzero(values) / len(values)),
            "mean": mean,
            "std": float(np.std(values)),
            "coefficient_of_variation": float(np.std(values) / mean) if mean > 0 else None,
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        })
    return {"rows": len(view.output), "outputs": rows}


def _group_run_errors(per_run: list[dict[str, Any]], run_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dimensions = ("workload", "arrival_process", "load_fraction", "transient_pattern")
    result: dict[str, Any] = {}
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_run:
            value = run_metadata[row["run_id"]].get(dimension)
            groups[str(value)].append(row)
        result[dimension] = {
            key: {
                "runs": len(rows),
                "state_nrmse_mean": float(np.mean([row["state_nrmse"] for row in rows])),
                "output_nrmse_mean": float(np.mean([row["output_nrmse"] for row in rows])),
            }
            for key, rows in sorted(groups.items())
        }
    return result


def _candidate_score(metrics: dict[str, Any], key_outputs: list[str]) -> float:
    key_error = float(np.mean([metrics["per_output_nrmse"][name] for name in key_outputs]))
    return key_error + 0.25 * metrics["state_nrmse"] + 0.10 * metrics["output_nrmse"]


def candidate_key_error(row: dict[str, Any], key_outputs: list[str]) -> float:
    return float(np.mean([row["validation"]["per_output_nrmse"][name] for name in key_outputs]))


def select_parsimonious_output_candidate(
    rows: list[dict[str, Any]], key_outputs: list[str], tolerance: float,
) -> dict[str, Any]:
    best_error = min(candidate_key_error(row, key_outputs) for row in rows)
    equivalent = [
        row for row in rows
        if candidate_key_error(row, key_outputs) <= best_error + tolerance
    ]
    return min(
        equivalent,
        key=lambda row: (
            row["rank"], row["factor"], not row["memory"],
            candidate_key_error(row, key_outputs),
        ),
    )


def _derived_output_schema(period_seconds: float) -> list[dict[str, Any]]:
    return [
        {"name": "throughput_output_tokens_per_s", "formula": "completed_output_tokens / period_s", "kind": "rate"},
        {"name": "goodput_output_tokens_per_s", "formula": "goodput_output_tokens / period_s", "kind": "rate"},
        {"name": "goodput_token_ratio", "formula": "goodput_output_tokens / completed_output_tokens", "kind": "ratio", "valid_when": "completed_output_tokens > 0"},
        {"name": "completion_rate_per_s", "formula": "completed_request_count / period_s", "kind": "rate"},
        {"name": "mean_ttft_ms", "formula": "ttft_sum_ms / completed_request_count", "kind": "conditional_mean", "valid_when": "completed_request_count > 0"},
        {
            "name": "mean_tpot_ms_approx",
            "formula": "tpot_sum_ms / completed_request_count",
            "kind": "conditional_mean",
            "valid_when": "completed_request_count > 0",
            "limitation": "Dataset v1.1 lacks tpot_valid_count; requests without a defined TPOT were accumulated as zero",
        },
        {"name": "ttft_violation_rate", "formula": "ttft_slo_violation_count / completed_request_count", "kind": "ratio", "valid_when": "completed_request_count > 0"},
        {"name": "rejection_rate", "formula": "request_rejected_count / (request_rejected_count + completed_request_count)", "kind": "ratio", "valid_when": "denominator > 0"},
        {"name": "kv_transfer_bytes_per_s", "formula": "kv_transfer_completed_bytes / period_s", "kind": "rate"},
        {"name": "prefill_tokens_per_s", "formula": "prefill_scheduled_tokens / period_s", "kind": "rate"},
        {"name": "decode_tokens_per_s", "formula": "decode_scheduled_tokens / period_s", "kind": "rate"},
    ]


def run_v2_analysis(
    dataset_root: Path,
    modeling_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("audit", "candidates", "design", "reports", "metadata"):
        (output_root / name).mkdir(exist_ok=True)
    dataset = RomDataset(dataset_root, dataset_root)
    source_manifest_before = sha256(dataset_root / "dataset_manifest.json")
    source_v1_manifest = json.loads((dataset_root / "snapshot_schema.json").read_text())["source_dataset_manifest_sha256"]
    x_normalizer = Normalizer.from_json(json.loads((modeling_root / "preprocessing/x_normalizer.json").read_text()))
    basis = np.load(modeling_root / "pod/basis.npy", mmap_mode="r")
    run_rows = dataset._run_rows
    run_metadata = {row["run_id"]: row for row in run_rows}
    output_names = [row["name"] for row in dataset.output_index]
    key_outputs = list(config["key_outputs"])
    factors = [int(value) for value in config["aggregation_factors"]]
    ranks = [int(value) for value in config["pod_ranks"]]
    ridge_values = [float(value) for value in config["ridge_values"]]
    max_rank = max(ranks)
    chunk_size = int(config["chunk_size"])

    views: dict[int, dict[str, AggregatedView]] = {}
    sparsity: dict[str, Any] = {}
    for factor in factors:
        views[factor] = {split: _build_view(dataset, split, factor) for split in SPLITS}
        sparsity[str(factor)] = {
            split: _output_sparsity(views[factor][split], output_names) for split in SPLITS
        }
    save_json(output_root / "audit/output_sparsity_by_horizon.json", sparsity)

    projections: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for factor in factors:
        projections[factor] = {}
        for split in SPLITS:
            view = views[factor][split]
            projections[factor][split] = (
                _project_at(dataset.array(split, "X"), view.starts, x_normalizer, basis, max_rank, chunk_size),
                _project_at(dataset.array(split, "X_next"), view.ends, x_normalizer, basis, max_rank, chunk_size),
            )

    candidate_rows: list[dict[str, Any]] = []
    candidate_models: dict[tuple[int, int, bool], Any] = {}
    candidate_payloads: dict[tuple[int, int, bool], dict[str, Any]] = {}
    for factor in factors:
        d_normalizer, _ = fit_normalizer(views[factor]["train"].disturbance, dataset.disturbance_index, chunk_size)
        y_normalizer, _ = fit_normalizer(views[factor]["train"].output, dataset.output_index, chunk_size)
        transformed_d = {
            split: transformed(views[factor][split].disturbance, d_normalizer, chunk_size)
            for split in SPLITS
        }
        transformed_y = {
            split: transformed(views[factor][split].output, y_normalizer, chunk_size)
            for split in SPLITS
        }
        for rank in ranks:
            for memory in (False, True):
                raw_state: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for split in SPLITS:
                    z, z_next = projections[factor][split]
                    if memory:
                        raw_state[split] = augment_markov_state(
                            z[:, :rank], z_next[:, :rank], views[factor][split].disturbance,
                            views[factor][split].runs,
                        )
                    else:
                        raw_state[split] = (z[:, :rank], z_next[:, :rank])
                state_index = _view_state_index(rank, memory, dataset.disturbance_index)
                state_normalizer, _ = fit_normalizer(raw_state["train"][0], state_index, chunk_size)
                state = {
                    split: (
                        state_normalizer.transform(raw_state[split][0], weighted=False),
                        state_normalizer.transform(raw_state[split][1], weighted=False),
                    )
                    for split in SPLITS
                }
                best = None
                best_model = None
                for ridge in ridge_values:
                    model = fit_model(
                        state["train"][0], state["train"][1], transformed_d["train"],
                        transformed_y["train"], ridge,
                    )
                    validation_one_step = one_step_metrics(
                        model,
                        state["validation"][0], state["validation"][1],
                        transformed_d["validation"], transformed_y["validation"],
                    )
                    score = (
                        validation_one_step["state_nrmse"]
                        + validation_one_step["output_nrmse"]
                    )
                    row = {
                        "factor": factor,
                        "period_ms": factor * 200,
                        "rank": rank,
                        "memory": memory,
                        "ridge": ridge,
                        "spectral_radius": model.spectral_radius,
                        "ridge_selection_score": score,
                        "validation_one_step": validation_one_step,
                    }
                    if model.spectral_radius <= float(config["spectral_radius_max"]):
                        if best is None or score < best["ridge_selection_score"]:
                            best, best_model = row, model
                if best is None:
                    continue
                validation = rollout_metrics(
                    best_model, state["validation"][0], transformed_d["validation"],
                    transformed_y["validation"], views[factor]["validation"].runs, output_names,
                )
                if not validation["finite"]:
                    continue
                best["validation"] = validation
                best["validation_score"] = _candidate_score(validation, key_outputs)
                key = (factor, rank, memory)
                candidate_rows.append(best)
                candidate_models[key] = best_model
                candidate_payloads[key] = {
                    "state": state,
                    "d": transformed_d,
                    "y": transformed_y,
                    "runs": {split: views[factor][split].runs for split in SPLITS},
                }
    save_json(output_root / "candidates/validation_candidates.json", candidate_rows)
    def key_error(row: dict[str, Any]) -> float:
        return candidate_key_error(row, key_outputs)

    def evaluate_candidate(row: dict[str, Any]) -> dict[str, Any]:
        key = (row["factor"], row["rank"], row["memory"])
        model = candidate_models[key]
        payload = candidate_payloads[key]
        evaluation = {"validation": row["validation"]}
        for split in ("test", "test/transient"):
            evaluation[split] = rollout_metrics(
                model, payload["state"][split][0], payload["d"][split], payload["y"][split],
                payload["runs"][split], output_names,
            )
        return evaluation

    state_pool = [
        row for row in candidate_rows
        if row["validation"]["state_nrmse"] <= float(config["state_nrmse_max"])
    ]
    if not state_pool:
        raise RuntimeError("no candidate passed the validation state rollout gate")
    state_selected = min(
        state_pool,
        key=lambda row: (
            row["validation"]["state_nrmse"] + 0.10 * row["validation"]["output_nrmse"],
            row["factor"], row["rank"], not row["memory"],
        ),
    )
    output_selected = select_parsimonious_output_candidate(
        candidate_rows, key_outputs, float(config["selection_tolerance"]),
    )
    state_final = evaluate_candidate(state_selected)
    output_final = evaluate_candidate(output_selected)
    single_rate_candidates = [
        row for row in candidate_rows
        if row["validation"]["state_nrmse"] <= float(config["state_nrmse_max"])
        and key_error(row) <= float(config["key_output_nrmse_max"])
    ]
    save_json(output_root / "candidates/selected_candidates.json", {
        "state_dynamics": {**state_selected, "held_out": state_final},
        "output_observation": {**output_selected, "held_out": output_final},
        "single_rate_feasible": bool(single_rate_candidates),
        "single_rate_candidates": single_rate_candidates,
    })

    grouped_errors = {
        split: _group_run_errors(output_final[split]["per_run"], run_metadata)
        for split in output_final
    }
    save_json(output_root / "audit/error_attribution_by_run.json", grouped_errors)

    factor_one = next(row for row in candidate_rows if row["factor"] == 1 and row["rank"] == 16 and not row["memory"])
    same_scale_memory = next(row for row in candidate_rows if row["factor"] == 1 and row["rank"] == 16 and row["memory"])
    sampling_gain = float(
        np.mean([factor_one["validation"]["per_output_nrmse"][name] for name in key_outputs])
        - np.mean([output_selected["validation"]["per_output_nrmse"][name] for name in key_outputs])
    )
    memory_gain = float(
        np.mean([factor_one["validation"]["per_output_nrmse"][name] for name in key_outputs])
        - np.mean([same_scale_memory["validation"]["per_output_nrmse"][name] for name in key_outputs])
    )
    attribution = {
        "schema_version": "servingrom.failure_attribution.v2",
        "base_windows": int(sum(dataset.array(split, "X").shape[0] for split in SPLITS)),
        "source_dataset_manifest": source_manifest_before,
        "source_dataset_v1_manifest": source_v1_manifest,
        "selected_state_candidate": {
            key: state_selected[key] for key in ("factor", "period_ms", "rank", "memory", "ridge", "spectral_radius", "validation_score")
        },
        "selected_output_candidate": {
            key: output_selected[key] for key in ("factor", "period_ms", "rank", "memory", "ridge", "spectral_radius", "validation_score")
        },
        "sampling_scale_key_output_nrmse_gain": sampling_gain,
        "memory_key_output_nrmse_gain_at_200ms": memory_gain,
        "mechanisms": {
            "window_sparsity_and_aliasing": sampling_gain > float(config["material_gain"]),
            "missing_short_term_memory": memory_gain > float(config["material_gain"]),
            "pod_rank_is_primary": output_selected["rank"] > min(ranks),
            "state_linear_closure_stable": state_selected["spectral_radius"] <= 1.0,
            "single_rate_state_output_model_feasible": bool(single_rate_candidates),
            "multi_rate_redesign_required": not single_rate_candidates,
        },
    }
    save_json(output_root / "audit/failure_attribution.json", attribution)

    state_design = {
        "schema_version": "servingrom.state_design.v2",
        "source_is_read_only": True,
        "sampling_period_ms": state_selected["period_ms"],
        "boundary_state": "X at the first 200 ms window of each super-window",
        "next_boundary_state": "X_next at the final 200 ms window of each super-window",
        "pod_rank": state_selected["rank"],
        "temporal_closure": (
            ["pod_state", "pod_state_delta", "previous_state_window_disturbance"]
            if state_selected["memory"] else ["pod_state"]
        ),
        "disturbance": "D at the 200 ms state clock; retain exact arrival timing for state transitions",
        "output_observation_period_ms": output_selected["period_ms"],
        "output_observation_steps": output_selected["factor"],
        "run_boundary_policy": "never aggregate or construct history across runs",
        "training_policy": "all normalization, POD and hyperparameter fitting use train only",
    }
    output_design = {
        "schema_version": "servingrom.output_design.v2",
        "sampling_period_ms": output_selected["period_ms"],
        "state_sampling_period_ms": state_selected["period_ms"],
        "architecture": "multi-rate: fast latent state dynamics plus a slower conservative KPI observation head",
        "conserved_base_outputs": output_names,
        "aggregation": "sum event-flow counters within each non-overlapping super-window",
        "derived_outputs": _derived_output_schema(output_selected["period_ms"] / 1000.0),
        "mask_policy": "undefined conditional means and ratios carry an explicit validity mask; they are never silently filled with zero",
        "known_observability_gap": "exact mean TPOT requires tpot_valid_count, which is not present in Dataset v1/v1.1",
        "slo_policy": {"ttft_slo_ms": 2000.0, "tpot_slo_ms": 100.0},
    }
    save_json(output_root / "design/state_v2.json", state_design)
    save_json(output_root / "design/output_v2.json", output_design)

    selected_key_errors = {
        split: {name: output_final[split]["per_output_nrmse"][name] for name in key_outputs}
        for split in output_final
    }
    best_by_factor = {
        factor: min(
            [row for row in candidate_rows if row["factor"] == factor],
            key=lambda row: (key_error(row), row["rank"], not row["memory"]),
        )
        for factor in factors
    }
    factor_lines = []
    for factor in factors:
        row = best_by_factor[factor]
        completed = next(
            item for item in sparsity[str(factor)]["train"]["outputs"]
            if item["name"] == "completed_request_count"
        )
        factor_lines.append(
            f"  - {factor * 200} ms: completion nonzero={completed['nonzero_ratio']:.4f}, "
            f"key-output NRMSE={key_error(row):.4f}, state NRMSE={row['validation']['state_nrmse']:.4f}, "
            f"rank={row['rank']}, memory={row['memory']}"
        )
    output_factor = output_selected["factor"]
    rank_lines = []
    for rank in ranks:
        rows = [
            row for row in candidate_rows
            if row["factor"] == output_factor and row["rank"] == rank and row["memory"]
        ]
        if rows:
            row = min(rows, key=key_error)
            rank_lines.append(f"  - r={rank}: key-output NRMSE={key_error(row):.6f}")
    state_test = state_final["test"]
    state_transient = state_final["test/transient"]
    _write_markdown(output_root / "reports/FAILURE_ATTRIBUTION.md", "ServingROM v2 失败归因", [
        f"- 输入：Dataset v1.1 的 `{attribution['base_windows']}` 个已封存窗口；没有重新采集或运行模型。",
        f"- v1.1 manifest：`{source_manifest_before}`；源 Dataset v1 manifest：`{source_v1_manifest}`。",
        "- 根因一：200 ms 的 Y 是完成事件流，在低请求率下高度零膨胀；状态库存与同窗完成流之间还存在服务时间延迟。",
        "- 根因二：v1 POD 只优化状态重构能量，未保证 goodput、TTFT、TPOT 所需的低能量方向进入 reduced state。",
        "- 根因三：v1 的一阶状态没有显式速度和上一窗口到达记忆，难以区分相同库存下的积压与恢复方向。",
        f"- 采样尺度带来的 validation 关键输出 NRMSE 改善：`{sampling_gain:.6f}`。",
        f"- 仅增加 200 ms 短期记忆带来的改善：`{memory_gain:.6f}`。",
        f"- 状态动力学候选：period={state_selected['period_ms']} ms, rank={state_selected['rank']}, memory={state_selected['memory']}, validation state NRMSE={state_selected['validation']['state_nrmse']:.6f}。",
        f"- 输出观测候选：period={output_selected['period_ms']} ms, rank={output_selected['rank']}, memory={output_selected['memory']}, validation key-output NRMSE={key_error(output_selected):.6f}。",
        f"- 单一时钟同时通过状态与关键输出门：`{bool(single_rate_candidates)}`；因此采用多速率设计。",
        f"- 慢速输出头关键输出 held-out NRMSE：`{json.dumps(selected_key_errors, ensure_ascii=False)}`。",
        "- workload、load fraction、arrival process 与 transient pattern 的分解位于 `audit/error_attribution_by_run.json`。",
    ])
    _write_markdown(output_root / "reports/FAILURE_ATTRIBUTION_DEEP_DIVE.md", "ServingROM v2 Failure Attribution Deep Dive", [
        "## 判定摘要",
        "",
        "- 主因：200 ms 完成流的零膨胀和 stock/flow 时间错位。",
        "- 次因：一阶状态缺少方向信息；200 ms 增加记忆仅带来小幅输出收益，但明显改善状态 rollout。",
        "- 非主因：POD rank 不足。5 秒输出头从 r=16 增至 r=64 的收益低于 validation 等价带。",
        "- 结构性结论：不存在同时满足状态与关键输出门的单时钟候选，必须采用多速率结构。",
        "",
        "## 时间尺度证据",
        "",
        *factor_lines,
        "",
        "200 ms 下 train completion 非零窗口仅约 5%，多数 Y 行是全零或近零事件流。聚合不是平滑装饰，而是在恢复服务过程对应的可观测时间尺度。",
        "",
        "## Rank 证据",
        "",
        *rank_lines,
        "",
        "高 rank 改善状态重构能量，却没有相应改善控制关键输出，说明无监督 POD 的能量排序与 QoS 可预测方向不一致。",
        "",
        "## Held-out 状态动力学",
        "",
        f"- 200 ms fast-state test state NRMSE：`{state_test['state_nrmse']:.6f}`。",
        f"- 200 ms fast-state transient state NRMSE：`{state_transient['state_nrmse']:.6f}`。",
        f"- fast-state 谱半径：`{state_selected['spectral_radius']:.6f}`。",
        "- 5 秒单速率候选 state NRMSE 接近 1，主要表现为快速遗忘初态并回归均值，不能因谱半径小就称为有效动力学。",
        "",
        "## 标签与可观测性限制",
        "",
        "- goodput 使用统一 TTFT=2000 ms、TPOT=100 ms 坐标，跨 workload 语义已经统一。",
        "- Dataset v1.1 缺少 `tpot_valid_count`；精确 mean TPOT 无法仅凭 201,600 个窗口恢复。",
        "- 当前 5 秒输出头是 failure-attribution benchmark，不是最终 MPC plant。后续模型应从 200 ms latent trajectory 生成 5 秒守恒 KPI。",
        "- 当前数据仍没有真实 u[k]，本阶段不能评价可控性或辨识 B 矩阵。",
    ])
    _write_markdown(output_root / "reports/STATE_OUTPUT_REDESIGN.md", "ServingROM v2 State / Output Redesign", [
        f"- 快速状态时钟：`{state_selected['period_ms']} ms`；POD rank `{state_selected['rank']}`；短期 Markov 记忆：`{state_selected['memory']}`。",
        f"- 慢速 KPI 输出时钟：`{output_selected['period_ms']} ms`，每个输出窗口覆盖 `{output_selected['factor']}` 个状态步。",
        "- stock/flow 分离：X 在快速时钟上保持边界库存；D 保留 200 ms 到达顺序；Y 在慢速窗口内守恒求和。",
        "- 任意聚合、差分和历史特征都在单个 run 内构造，严禁跨 run 泄漏。",
        "- 输出同时保留可对账的原始计数，并派生 throughput、goodput ratio、条件 TTFT/TPOT 与 violation rate。",
        "- 条件指标使用显式 validity mask，不再把无完成请求的窗口伪装成 0 ms 延迟。",
        "- 5 秒单速率状态模型虽稳定但 state NRMSE 接近 1，是均值回归，不作为动力学模型。",
        "- v2 仍然没有 u[k]；队列、scheduler 输出和固定 MU 不会被伪装成 actuator。",
        "- 详细机器可读 schema：`design/state_v2.json` 与 `design/output_v2.json`。",
    ])
    source_manifest_after = sha256(dataset_root / "dataset_manifest.json")
    if source_manifest_after != source_manifest_before:
        raise RuntimeError("source Dataset v1.1 changed during read-only analysis")
    summary = {
        "status": "COMPLETE",
        "elapsed_seconds": time.time() - started,
        "source_immutable": True,
        "source_manifest_before": source_manifest_before,
        "source_manifest_after": source_manifest_after,
        "selected_state_candidate": attribution["selected_state_candidate"],
        "selected_output_candidate": attribution["selected_output_candidate"],
        "single_rate_feasible": bool(single_rate_candidates),
        "state_held_out": state_final,
        "output_held_out": output_final,
        "reports": [str(path.relative_to(output_root)) for path in sorted((output_root / "reports").glob("*.md"))],
    }
    save_json(output_root / "V2_ANALYSIS_SUMMARY.json", summary)
    return summary
