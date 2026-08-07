from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import RomDataset, SPLITS
from .dynamics import fit_model, one_step_metrics, project, rollout_metrics, transformed
from .pod import fit_pod, mode_block_contributions, reconstruction_scan
from .preprocessing import fit_normalizer, save_json, split_statistics


def _write_markdown(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", *lines]) + "\n", encoding="utf-8")


def _constant_summary(array: np.ndarray, names: list[str]) -> dict[str, Any]:
    minimum = np.min(array, axis=0)
    maximum = np.max(array, axis=0)
    return {
        "constant": bool(np.array_equal(minimum, maximum)),
        "dimensions": [
            {"name": name, "min": float(minimum[i]), "max": float(maximum[i]), "constant": bool(minimum[i] == maximum[i])}
            for i, name in enumerate(names)
        ],
    }


def _partial_summary(output_root: Path, stage: str, started: float) -> dict[str, Any]:
    value = {
        "status": f"{stage.upper()}_COMPLETE",
        "elapsed_seconds": time.time() - started,
        "reports": [str(path.relative_to(output_root)) for path in sorted((output_root / "reports").glob("*.md"))],
    }
    save_json(output_root / "ROM_MODELING_SUMMARY.json", value)
    return value


def run_pipeline(
    dataset_root: Path, index_dir: Path, output_root: Path,
    config: dict[str, Any], stop_after: str = "step13",
) -> dict[str, Any]:
    started = time.time()
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("reports", "audit", "preprocessing", "pod", "models", "evaluation", "metadata"):
        (output_root / name).mkdir(exist_ok=True)
    dataset = RomDataset(dataset_root, index_dir)
    save_json(output_root / "metadata/dataset_provenance.json", dataset.provenance())
    shutil.copy2(Path(config["config_path"]), output_root / "metadata/modeling_config.json")
    for name in ("disturbance_index.json", "output_index.json", "static_config.json"):
        shutil.copy2(index_dir / name, output_root / "metadata" / name)

    chunk_size = int(config["chunk_size"])
    indexes = {
        "X": dataset.state_index, "D": dataset.disturbance_index,
        "Y": dataset.output_index, "MU": dataset.static_index,
    }
    normalizers = {}
    audit = {"schema_version": "servingrom.rom_audit.v1", "fit_split": "train", "arrays": {}}
    for name, index in indexes.items():
        normalizer, train_audit = fit_normalizer(dataset.array("train", name), index, chunk_size)
        normalizers[name] = normalizer
        save_json(output_root / f"preprocessing/{name.lower()}_normalizer.json", normalizer.to_json())
        split_stats = {
            split: split_statistics(dataset.array(split, name), normalizer, chunk_size)
            for split in SPLITS
        }
        audit["arrays"][name] = {"train": train_audit, "splits": split_stats}
    mu_all = np.concatenate([np.asarray(dataset.array(split, "MU")) for split in SPLITS], axis=0)
    audit["mu_dataset_constant"] = _constant_summary(mu_all, [row["name"] for row in dataset.static_index])
    varying_mu = [row["name"] for row in audit["mu_dataset_constant"]["dimensions"] if not row["constant"]]
    slo_state_dimensions = [row["name"] for row in dataset.state_index if "ttft_slack" in row["name"]]
    slo_output_dimensions = [
        row["name"] for row in dataset.output_index
        if row["name"] in {"goodput_request_count", "goodput_output_tokens", "ttft_slo_violation_count"}
    ]
    structural_issues = []
    if varying_mu:
        structural_issues.append({
            "code": "mu_not_constant",
            "varying_dimensions": varying_mu,
            "affected_state_dimensions": len(slo_state_dimensions),
            "affected_outputs": slo_output_dimensions,
            "reason": "ttft_slo_ms changes the state-coordinate and output-label definitions across workloads",
        })
    audit["structural_gate"] = {"passed": not structural_issues, "issues": structural_issues}
    save_json(output_root / "audit/data_audit.json", audit)
    x_features = audit["arrays"]["X"]["train"]["features"]
    constant_x = sum(not row["active"] for row in x_features)
    _write_markdown(output_root / "reports/STEP9_DATA_AUDIT.md", "Step 9 ROM 数据审计与状态预处理", [
        f"- 数据集 manifest：`{dataset.provenance()['dataset_manifest_sha256']}`",
        f"- Run 隔离：`{json.dumps(dataset.provenance()['runs_by_split'], ensure_ascii=False)}`",
        f"- X 恒零/近常量维度：`{constant_x}/1804`",
        f"- X 有效维度：`{1804 - constant_x}`",
        f"- MU 全 Dataset 恒定：`{audit['mu_dataset_constant']['constant']}`",
        "- 拟合策略：只在 train 上执行 log1p、均值/方差估计和常量维检测。",
        "- 尺度策略：各维 z-score 后按物理 block 的有效维数平方根进行平衡，防止 bytes/token mass 和高维 histogram block 支配 POD。",
        "- validation、test、test/transient 仅复用冻结 normalizer，不参与任何拟合。",
        "- Dataset v1 未修改；缺失的 D/Y/MU 索引已作为建模 provenance 单独保存。",
        f"- Step 9 结构门：`{'PASS' if not structural_issues else 'FAIL'}`",
        f"- 变化的 MU 维度：`{varying_mu}`",
        f"- 受 TTFT SLO 定义影响的 X 维度：`{len(slo_state_dimensions)}`",
        f"- 受 TTFT SLO 定义影响的 Y 维度：`{slo_output_dimensions}`",
    ])
    if structural_issues:
        _write_markdown(output_root / "reports/STEP9_STRUCTURAL_BLOCKER.md", "Step 9 结构性阻断分析", [
            "- 结论：当前合并 Dataset v1 不能直接按单一固定 `mu0` 训练统一 POD/DMDc。",
            "- `ttft_slo_ms` 按 workload 分别为 balanced=2000、mixed-bimodal=3000、long-prefill=5000 ms。",
            f"- 该参数改变了 X 中 `{len(slo_state_dimensions)}` 个 `prefill.ttft_slack_*` 分箱坐标。",
            f"- 该参数同时改变 Y 中 `{slo_output_dimensions}` 的标签语义。",
            "- 若直接继续，POD 会把坐标定义差异吸收到模态，DMDc 会把 workload 与 SLO 的共线关系误判为动力学。",
            "- 路线 A：建立不含 SLO-dependent X/Y 的 mu-independent 建模视图，可研究 queue/KV/throughput/TTFT sum，但不能统一预测 goodput。",
            "- 路线 B：从已封存 raw telemetry 用统一 TTFT SLO 重建派生 Snapshot Dataset v1.1；无需重新推理或重新采集，且能保留统一 goodput 语义。",
            "- 路线 C：按 2000/3000/5000 ms 分别训练三个固定-mu ROM；每个模型只有对应 workload 的 8 个 train runs，统计辨识能力较弱。",
            "- 不建议把 `ttft_slo_ms` 当作 actuator；它是静态评价/坐标参数，不是运行时控制输入。",
            "- 本轮按 fail-closed 规则停止，不执行 Step 10–13。",
        ])
        value = {
            "status": "STEP9_BLOCKED",
            "elapsed_seconds": time.time() - started,
            "structural_issues": structural_issues,
            "reports": [str(path.relative_to(output_root)) for path in sorted((output_root / "reports").glob("*.md"))],
        }
        save_json(output_root / "ROM_MODELING_SUMMARY.json", value)
        return value
    if stop_after == "step9":
        return _partial_summary(output_root, "step9", started)

    ranks = [int(value) for value in config["pod_ranks"]]
    pod_result = fit_pod(dataset.array("train", "X"), normalizers["X"], max(ranks), chunk_size)
    basis = pod_result["basis"]
    eigenvalues = pod_result["eigenvalues"]
    np.save(output_root / "pod/basis.npy", basis)
    np.save(output_root / "pod/eigenvalues.npy", eigenvalues)
    np.save(output_root / "pod/singular_values.npy", pod_result["singular_values"])
    total_energy = float(eigenvalues.sum())
    cumulative = np.cumsum(eigenvalues) / max(total_energy, 1e-30)
    spectrum = [
        {"mode": i + 1, "eigenvalue": float(value), "singular_value": float(pod_result["singular_values"][i]), "cumulative_energy": float(cumulative[i])}
        for i, value in enumerate(eigenvalues)
    ]
    save_json(output_root / "pod/spectrum.json", spectrum)
    save_json(output_root / "pod/mode_block_contributions.json", mode_block_contributions(basis, normalizers["X"], modes=32))
    reconstruction = {
        split: reconstruction_scan(dataset.array(split, "X"), normalizers["X"], basis, ranks, chunk_size)
        for split in SPLITS
    }
    save_json(output_root / "pod/reconstruction_scan.json", reconstruction)
    with (output_root / "pod/spectrum.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("mode", "eigenvalue", "singular_value", "cumulative_energy"))
        writer.writeheader()
        writer.writerows(spectrum)
    _write_markdown(output_root / "reports/STEP10_POD_REPORT.md", "Step 10 POD 状态空间降阶", [
        f"- Train 有效协方差维度：`{int(normalizers['X'].active.sum())}`",
        f"- 候选 ranks：`{ranks}`",
        "- Rank 扫描重构误差：",
        *[f"  - r={rank}: " + ", ".join(f"{split}={reconstruction[split][rank]:.6f}" for split in SPLITS) for rank in ranks],
        "- Rank 不按单一累计能量冻结；全部候选进入线性动力学识别。",
        "- 完整谱位于 `pod/spectrum.csv`，物理块模态贡献位于 `pod/mode_block_contributions.json`。",
    ])
    if stop_after == "step10":
        return _partial_summary(output_root, "step10", started)

    projected: dict[str, dict[str, np.ndarray]] = {}
    max_rank = max(ranks)
    for split in SPLITS:
        projected[split] = {
            "z": project(dataset.array(split, "X"), normalizers["X"], basis, max_rank, chunk_size),
            "z_next": project(dataset.array(split, "X_next"), normalizers["X"], basis, max_rank, chunk_size),
            "d": transformed(dataset.array(split, "D"), normalizers["D"], chunk_size),
            "y": transformed(dataset.array(split, "Y"), normalizers["Y"], chunk_size),
        }
    ridge_values = [float(value) for value in config["ridge_values"]]
    candidates = []
    best_by_rank = {}
    for rank in ranks:
        rank_rows = []
        for ridge in ridge_values:
            model = fit_model(
                projected["train"]["z"][:, :rank], projected["train"]["z_next"][:, :rank],
                projected["train"]["d"], projected["train"]["y"], ridge,
            )
            train_one = one_step_metrics(model, projected["train"]["z"][:, :rank], projected["train"]["z_next"][:, :rank], projected["train"]["d"], projected["train"]["y"])
            validation_one = one_step_metrics(model, projected["validation"]["z"][:, :rank], projected["validation"]["z_next"][:, :rank], projected["validation"]["d"], projected["validation"]["y"])
            row = {"rank": rank, "ridge": ridge, "spectral_radius": model.spectral_radius, "train_one_step": train_one, "validation_one_step": validation_one}
            rank_rows.append((row, model))
            candidates.append(row)
        stable = [item for item in rank_rows if item[1].spectral_radius <= float(config["candidate_spectral_radius_max"])]
        pool = stable or rank_rows
        best_by_rank[rank] = min(pool, key=lambda item: item[0]["validation_one_step"]["state_nrmse"])
    save_json(output_root / "models/one_step_candidates.json", candidates)

    output_names = [row["name"] for row in dataset.output_index]
    rollout_candidates = []
    for rank in ranks:
        row, model = best_by_rank[rank]
        metrics = rollout_metrics(
            model, projected["validation"]["z"][:, :rank], projected["validation"]["d"],
            projected["validation"]["y"], dataset.run_slices("validation"), output_names,
        )
        value = {**row, "validation_rollout": metrics}
        rollout_candidates.append(value)
        model.save(output_root / f"models/candidate_r{rank}_ridge{model.ridge:g}.npz")
    save_json(output_root / "models/validation_rollout_candidates.json", rollout_candidates)
    finite = [row for row in rollout_candidates if row["validation_rollout"]["finite"]]
    if not finite:
        raise RuntimeError("all validation rollouts diverged; stop before test evaluation")
    selected = min(finite, key=lambda row: (
        row["validation_rollout"]["state_nrmse"] + row["validation_rollout"]["output_nrmse"],
        row["rank"],
    ))
    selected_model = best_by_rank[selected["rank"]][1]
    selected_model.save(output_root / "models/selected_model.npz")
    selection = {"selected_rank": selected["rank"], "selected_ridge": selected["ridge"], "spectral_radius": selected["spectral_radius"], "selection_split": "validation", "validation": selected}
    save_json(output_root / "models/model_selection.json", selection)
    _write_markdown(output_root / "reports/STEP11_DMDC_REPORT.md", "Step 11 DMDc / Reduced Dynamics Identification", [
        "- 模型：`z[k+1] = A z[k] + E d[k] + c`，`y[k] = C z[k] + F d[k] + b`。",
        "- MU 已确认固定，不作为控制输入；Dataset v1 不包含 u[k]。",
        f"- Ridge 扫描：`{ridge_values}`；超参数选择只使用 validation。",
        f"- 选定 rank：`{selection['selected_rank']}`",
        f"- 选定 ridge：`{selection['selected_ridge']}`",
        f"- A 谱半径：`{selection['spectral_radius']:.8f}`",
        f"- Validation one-step state NRMSE：`{selected['validation_one_step']['state_nrmse']:.6f}`",
        f"- Validation rollout state NRMSE：`{selected['validation_rollout']['state_nrmse']:.6f}`",
    ])
    if stop_after == "step11":
        return _partial_summary(output_root, "step11", started)

    final_evaluation = {"validation": selected["validation_rollout"]}
    for split in ("test", "test/transient"):
        final_evaluation[split] = rollout_metrics(
            selected_model, projected[split]["z"][:, :selected_model.rank], projected[split]["d"],
            projected[split]["y"], dataset.run_slices(split), output_names,
        )
    save_json(output_root / "evaluation/final_rollout_metrics.json", final_evaluation)
    transient_patterns: dict[str, list[dict[str, Any]]] = {}
    for row in final_evaluation["test/transient"]["per_run"]:
        transient_patterns.setdefault(str(row["transient_pattern"]), []).append(row)
    transient_summary = {
        pattern: {
            "runs": len(rows),
            "state_nrmse_mean": float(np.mean([row["state_nrmse"] for row in rows])),
            "output_nrmse_mean": float(np.mean([row["output_nrmse"] for row in rows])),
            "all_finite": all(row["finite"] for row in rows),
        }
        for pattern, rows in transient_patterns.items()
    }
    save_json(output_root / "evaluation/transient_pattern_summary.json", transient_summary)
    key_outputs = config["key_outputs"]
    unavailable_key_outputs = [
        name for name in key_outputs
        if not bool(normalizers["Y"].active[output_names.index(name)])
    ]
    gate_checks = {
        "spectral_radius": selected_model.spectral_radius <= float(config["gate"]["spectral_radius_max"]),
        "validation_rollout_finite": final_evaluation["validation"]["finite"],
        "test_rollout_finite": final_evaluation["test"]["finite"],
        "transient_rollout_finite": final_evaluation["test/transient"]["finite"],
        "validation_state_skill_positive": final_evaluation["validation"]["state_skill_vs_initial_persistence"] > 0,
        "test_state_skill_positive": final_evaluation["test"]["state_skill_vs_initial_persistence"] > 0,
        "transient_state_skill_positive": final_evaluation["test/transient"]["state_skill_vs_initial_persistence"] > 0,
        "validation_output_skill_positive": final_evaluation["validation"]["output_skill_vs_train_mean"] > 0,
        "transient_output_skill_positive": final_evaluation["test/transient"]["output_skill_vs_train_mean"] > 0,
        "key_outputs_observable": not unavailable_key_outputs,
    }
    actuator_ready = all(gate_checks.values())
    gate = {"actuator_mpc_ready": actuator_ready, "checks": gate_checks, "unavailable_key_outputs": unavailable_key_outputs}
    save_json(output_root / "evaluation/actuator_mpc_gate.json", gate)
    _write_markdown(output_root / "reports/STEP12_ROLLOUT_REPORT.md", "Step 12 多步 Rollout 与 Held-out Transient 验证", [
        f"- Validation state/output NRMSE：`{final_evaluation['validation']['state_nrmse']:.6f}` / `{final_evaluation['validation']['output_nrmse']:.6f}`",
        f"- Test state/output NRMSE：`{final_evaluation['test']['state_nrmse']:.6f}` / `{final_evaluation['test']['output_nrmse']:.6f}`",
        f"- Transient state/output NRMSE：`{final_evaluation['test/transient']['state_nrmse']:.6f}` / `{final_evaluation['test/transient']['output_nrmse']:.6f}`",
        f"- Transient pattern：`{json.dumps(transient_summary, ensure_ascii=False)}`",
        f"- 不可观测关键输出：`{unavailable_key_outputs}`",
        "- 所有指标按完整 held-out run 自由 rollout 计算，未随机打散窗口。",
    ])
    if stop_after == "step12":
        return _partial_summary(output_root, "step12", started)
    _write_markdown(output_root / "reports/STEP13_ACTUATOR_MPC_GATE.md", "Step 13 Actuator / MPC 准入结论", [
        f"- 是否允许进入 actuator excitation 与 MPC：`{actuator_ready}`",
        f"- 自动门：`{json.dumps(gate_checks, ensure_ascii=False)}`",
        f"- 缺失关键输出：`{unavailable_key_outputs}`",
        "- 当前 Dataset v1 中 MU 为固定配置，且不存在正式运行时 actuator；不会把 scheduler 输出、queue 或 MU 伪装成 u[k]。",
        "- 只有全部门通过，后续才设计可热更新 token budget、max-num-seqs 或路由比例的独立 excitation 数据集。",
    ])
    summary = {
        "status": "COMPLETE", "elapsed_seconds": time.time() - started,
        "selected_model": selection, "gate": gate,
        "reports": [str(path.relative_to(output_root)) for path in sorted((output_root / "reports").glob("*.md"))],
    }
    save_json(output_root / "ROM_MODELING_SUMMARY.json", summary)
    return summary
