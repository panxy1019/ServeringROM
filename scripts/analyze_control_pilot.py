#!/usr/bin/env python3
"""Analyze Round 14.1 actuator response without fitting a control model."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


PERIOD_SECONDS = 0.2
SLOW_FACTOR = 25
MAX_LAG_SECONDS = 60
DIRECT_SIGNALS = (
    "decode_running_imbalance",
    "decode_waiting_imbalance",
    "decode_expected_remaining_imbalance",
)


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if len(left) < 5 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def effect_size(low: np.ndarray, high: np.ndarray) -> float | None:
    low, high = finite(low), finite(high)
    if len(low) < 5 or len(high) < 5:
        return None
    pooled = math.sqrt((float(np.var(low, ddof=1)) + float(np.var(high, ddof=1))) / 2)
    if pooled < 1e-12:
        return None if abs(float(np.mean(high) - np.mean(low))) < 1e-12 else math.copysign(float("inf"), float(np.mean(high) - np.mean(low)))
    return float((np.mean(high) - np.mean(low)) / pooled)


def state_indices(path: Path) -> dict[str, int]:
    return {row["name"]: int(row["index"]) for row in json.loads(path.read_text())}


def slow_mean(values: np.ndarray) -> np.ndarray:
    if len(values) % SLOW_FACTOR:
        raise ValueError(f"fast window count {len(values)} is not divisible by {SLOW_FACTOR}")
    reshaped = values.reshape(-1, SLOW_FACTOR)
    counts = np.isfinite(reshaped).sum(axis=1)
    totals = np.nansum(reshaped, axis=1)
    return np.divide(
        totals, counts, out=np.full(len(counts), np.nan, dtype=float), where=counts > 0
    )


def load_run(root: Path) -> dict[str, Any]:
    metadata = json.loads((root / "metadata" / "run.json").read_text())
    workload_result = json.loads((root / "metadata" / "workload_result.json").read_text())
    quality = json.loads((root / "reports" / "control_pilot_quality.json").read_text())
    snapshot_quality = json.loads((root / "reports" / "snapshot_data_quality.json").read_text())
    if not quality.get("valid"):
        raise ValueError(f"unsealed or invalid run: {root.name}")
    control = pq.read_table(root / "derived" / "control" / "control_windows.parquet").to_pylist()
    slow_kpi = pq.read_table(root / "derived" / "control" / "slow_control_kpi_windows.parquet").to_pylist()
    x = np.load(root / "derived" / "snapshots" / "full_state.npy")
    index = state_indices(root / "derived" / "snapshots" / "state_index.json")
    u = np.asarray([float(row["u_rho_A"]) for row in control])
    signals = {
        "decode_running_imbalance": x[:, index["decode_d1_running_count"]] - x[:, index["decode_d2_running_count"]],
        "decode_waiting_imbalance": x[:, index["decode_d1_waiting_count"]] - x[:, index["decode_d2_waiting_count"]],
        "decode_expected_remaining_imbalance": x[:, index["decode_d1_expected_remaining_tokens"]] - x[:, index["decode_d2_expected_remaining_tokens"]],
        "route_request_imbalance": np.asarray([
            np.nan if row["actual_request_ratio"] is None else 2 * float(row["actual_request_ratio"]) - 1
            for row in slow_kpi
        ]),
        "route_token_imbalance": np.asarray([
            np.nan if row["actual_token_ratio"] is None else 2 * float(row["actual_token_ratio"]) - 1
            for row in slow_kpi
        ]),
    }
    state_signals = {name: slow_mean(value) for name, value in signals.items() if name in DIRECT_SIGNALS}
    route_signals = {name: value for name, value in signals.items() if name not in DIRECT_SIGNALS}
    return {
        "root": root,
        "metadata": metadata,
        "u_fast": u,
        "u": slow_mean(u),
        "signals": {**state_signals, **route_signals},
        "slow_kpi": slow_kpi,
        "workload_result": workload_result,
        "snapshot_quality": snapshot_quality,
    }


def lag_analysis(u: np.ndarray, signal: np.ndarray) -> dict[str, Any]:
    rows = []
    max_lag = int(MAX_LAG_SECONDS / (PERIOD_SECONDS * SLOW_FACTOR))
    for lag in range(max_lag + 1):
        value = correlation(u[: len(u) - lag or None], signal[lag:])
        rows.append({"lag_seconds": lag * 5, "correlation": value})
    available = [row for row in rows if row["correlation"] is not None]
    strongest = max(available, key=lambda row: abs(row["correlation"])) if available else None
    positive = [row for row in available if row["correlation"] > 0]
    best_positive = max(positive, key=lambda row: row["correlation"]) if positive else None
    return {"best_positive": best_positive, "strongest_absolute": strongest, "curve": rows}


def level_analysis(u: np.ndarray, signal: np.ndarray) -> dict[str, Any]:
    values = {}
    for level in (0.3, 0.5, 0.7):
        selected = finite(signal[np.isclose(u, level, atol=1e-6)])
        values[str(level)] = {
            "count": int(len(selected)),
            "mean": float(np.mean(selected)) if len(selected) else None,
            "std": float(np.std(selected)) if len(selected) else None,
        }
    low = signal[np.isclose(u, 0.3, atol=1e-6)]
    high = signal[np.isclose(u, 0.7, atol=1e-6)]
    delta = None
    if len(finite(low)) and len(finite(high)):
        delta = float(np.mean(finite(high)) - np.mean(finite(low)))
    return {"levels": values, "high_minus_low": delta, "cohens_d": effect_size(low, high)}


def step_analysis(u: np.ndarray, signal: np.ndarray) -> dict[str, Any]:
    steps = []
    changed = np.flatnonzero(np.abs(np.diff(u, prepend=u[0])) > 1e-12)
    for index in changed:
        if index < 2:
            continue
        next_change = next((value for value in changed if value > index), len(u))
        stop = min(next_change, index + 13)
        before = finite(signal[max(0, index - 2):index])
        after = finite(signal[index:stop])
        if not len(before) or len(after) < 2:
            continue
        baseline = float(np.mean(before))
        plateau = float(np.mean(after[-min(3, len(after)):]))
        amplitude = plateau - baseline
        direction = math.copysign(1.0, float(u[index] - u[index - 1]))
        response_delay = None
        settle = None
        if abs(amplitude) > 1e-12:
            threshold = baseline + 0.2 * amplitude
            for offset, value in enumerate(signal[index:stop]):
                if np.isfinite(value) and (value - threshold) * amplitude >= 0:
                    response_delay = offset * 5.0
                    break
            tolerance = max(abs(amplitude) * 0.2, 1e-9)
            for offset in range(len(after)):
                tail = after[offset:]
                if len(tail) >= 2 and np.all(np.abs(tail - plateau) <= tolerance):
                    settle = offset * 5.0
                    break
        steps.append({
            "slow_window": int(index), "old_u": float(u[index - 1]), "new_u": float(u[index]),
            "command_direction": direction, "baseline": baseline, "plateau": plateau,
            "amplitude": amplitude, "direction_correct": bool(amplitude * direction > 0),
            "response_delay_seconds": response_delay, "settling_time_seconds": settle,
        })
    valid = [row for row in steps if row["response_delay_seconds"] is not None]
    return {
        "steps": steps,
        "direction_correct_fraction": sum(row["direction_correct"] for row in steps) / len(steps) if steps else None,
        "median_response_delay_seconds": float(np.median([row["response_delay_seconds"] for row in valid])) if valid else None,
        "median_settling_time_seconds": float(np.median([row["settling_time_seconds"] for row in steps if row["settling_time_seconds"] is not None])) if any(row["settling_time_seconds"] is not None for row in steps) else None,
    }


def analyze(runs_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    sealed = [row for row in manifest["runs"] if row["status"] == "SEALED"]
    if len(sealed) != 12:
        raise ValueError(f"expected 12 sealed runs, got {len(sealed)}")
    runs = [load_run(runs_root / row["run_id"]) for row in sealed]
    run_reports = []
    for run in runs:
        signals = {}
        for name, values in run["signals"].items():
            signals[name] = {
                "lag": lag_analysis(run["u"], values),
                "level": level_analysis(run["u"], values),
                "step": step_analysis(run["u"], values),
            }
        workload_summary = run["workload_result"]["summary"]
        quality_metrics = run["snapshot_quality"]["metrics"]
        run_reports.append({
            "run_id": run["root"].name,
            **run["metadata"],
            "signals": signals,
            "workload_summary": workload_summary,
            "minimum_dwell_retry_count": sum(
                int(row.get("minimum_dwell_retry_count", 0))
                for row in run["workload_result"].get("control_schedule", [])
            ),
            "quality": {
                "valid": bool(run["snapshot_quality"].get("valid")),
                "event_seq_gap_processes": quality_metrics["event_seq_gap_processes"],
                "jsonl_damaged_lines": quality_metrics["jsonl_damaged_lines"],
                "writer_failure_count": quality_metrics["writer_failure_count"],
                "request_inventory_conservation_ratio": quality_metrics["request_inventory_conservation_ratio"],
                "stage_inventory_conservation_ratio": quality_metrics["stage_inventory_conservation_ratio"],
                "kv_lifecycle_violation_count": quality_metrics["kv_lifecycle_violation_count"],
            },
        })

    grouped: dict[str, list[float]] = defaultdict(list)
    authority_runs = 0
    for report in run_reports:
        direct_hits = 0
        for name in DIRECT_SIGNALS:
            level = report["signals"][name]["level"]
            effect = level["cohens_d"]
            if level["high_minus_low"] is not None and level["high_minus_low"] > 0 and effect is not None and effect >= 0.25:
                direct_hits += 1
            if effect is not None and math.isfinite(effect):
                grouped[f"{report['workload']}:{int(report['load_fraction'] * 100)}:{name}"].append(effect)
        report["direct_authority_signal_count"] = direct_hits
        report["direct_authority_pass"] = direct_hits >= 1
        authority_runs += report["direct_authority_pass"]

    every_working_point = True
    working_points = []
    for workload in ("balanced", "mixed-bimodal"):
        for fraction in (0.55, 0.85):
            selected = [row for row in run_reports if row["workload"] == workload and math.isclose(row["load_fraction"], fraction)]
            passed = sum(row["direct_authority_pass"] for row in selected)
            point_pass = passed >= 2
            every_working_point &= point_pass
            working_points.append({"workload": workload, "load_fraction": fraction, "runs_passed": passed, "runs": len(selected), "pass": point_pass})

    persistent = all(
        np.var(run["u"]) >= 0.005 and set(np.round(run["u"], 1)) >= {0.3, 0.5, 0.7}
        for run in runs
    )
    authority = authority_runs >= 8 and every_working_point
    quality_pass = all(
        row["quality"]["valid"]
        and row["quality"]["event_seq_gap_processes"] == 0
        and row["quality"]["jsonl_damaged_lines"] == 0
        and row["quality"]["writer_failure_count"] == 0
        and row["quality"]["request_inventory_conservation_ratio"] == 1.0
        and row["quality"]["stage_inventory_conservation_ratio"] == 1.0
        and row["quality"]["kv_lifecycle_violation_count"] == 0
        and row["workload_summary"]["error_count"] == 0
        for row in run_reports
    )
    return {
        "schema_version": "servingrom.control_pilot_analysis.v1",
        "run_count": len(runs),
        "persistent_excitation_pass": bool(persistent),
        "control_authority_pass": bool(authority),
        "data_quality_pass": bool(quality_pass),
        "authority_gate": {
            "definition": "positive high-minus-low and Cohen's d >= 0.25 for at least one direct Decode state signal in >=8/12 runs and >=2/3 runs at every working point",
            "passing_runs": authority_runs,
            "working_points": working_points,
        },
        "run_reports": run_reports,
        "grouped_effect_sizes": dict(grouped),
        "aggregate": {
            "successful_requests": sum(row["workload_summary"]["success_count"] for row in run_reports),
            "completion_tokens": sum(row["workload_summary"]["completion_tokens"] for row in run_reports),
            "request_errors": sum(row["workload_summary"]["error_count"] for row in run_reports),
            "minimum_dwell_retries": sum(row["minimum_dwell_retry_count"] for row in run_reports),
            "event_seq_gap_processes": sum(row["quality"]["event_seq_gap_processes"] for row in run_reports),
            "jsonl_damaged_lines": sum(row["quality"]["jsonl_damaged_lines"] for row in run_reports),
            "writer_failures": sum(row["quality"]["writer_failure_count"] for row in run_reports),
            "kv_lifecycle_violations": sum(row["quality"]["kv_lifecycle_violation_count"] for row in run_reports),
        },
    }


def write_report(output: Path, result: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# ServingROM Round 14.1 控制激励 Pilot 报告", "",
        "## 结论", "",
        f"- sealed runs: `{result['run_count']}/12`",
        f"- persistent_excitation_pass: `{str(result['persistent_excitation_pass']).lower()}`",
        f"- control_authority_pass: `{str(result['control_authority_pass']).lower()}`",
        f"- data_quality_pass: `{str(result['data_quality_pass']).lower()}`",
        f"- direct authority passing runs: `{result['authority_gate']['passing_runs']}/12`", "",
        "门控定义：" + result["authority_gate"]["definition"], "",
        "## 工作点", "",
        "| Workload | Load | Passed runs | Gate |", "|---|---:|---:|---|",
    ]
    for row in result["authority_gate"]["working_points"]:
        lines.append(f"| {row['workload']} | {row['load_fraction']:.0%} | {row['runs_passed']}/{row['runs']} | {row['pass']} |")
    lines += [
        "", "## 汇总质量", "",
        f"- successful requests: `{result['aggregate']['successful_requests']}`",
        f"- completion tokens: `{result['aggregate']['completion_tokens']}`",
        f"- request errors: `{result['aggregate']['request_errors']}`",
        f"- dwell-boundary retries: `{result['aggregate']['minimum_dwell_retries']}`",
        f"- event-seq gaps / damaged JSONL / writer failures: `{result['aggregate']['event_seq_gap_processes']} / {result['aggregate']['jsonl_damaged_lines']} / {result['aggregate']['writer_failures']}`",
        f"- KV lifecycle violations: `{result['aggregate']['kv_lifecycle_violations']}`",
        "", "## Run 级响应", "",
        "| Run | Excitation | Requests | TTFT P95 | Direct signals | Positive lag | Remaining effect |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in result["run_reports"]:
        key = run["signals"]["decode_expected_remaining_imbalance"]
        best = key["lag"]["best_positive"] or {}
        effect = key["level"]["cohens_d"]
        lines.append(
            f"| {run['run_id']} | {run['excitation']} | {run['workload_summary']['success_count']} | "
            f"{run['workload_summary']['ttft_p95_seconds']:.3f} s | {run['direct_authority_signal_count']} | "
            f"{best.get('lag_seconds', 'n/a')} s | {effect if effect is not None else 'n/a'} |"
        )
    lines += [
        "", "## 解释边界", "",
        "本报告只识别 actuator 到状态/输出的经验响应，不拟合 Control-ROM，不推断因果闭环稳定性，也不实现 MPC。",
        "`U` 只来自 `actuator_applied.effective_value`；actual request/token ratio 仅是诊断输出。",
    ]
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.runs_root, args.manifest)
    write_report(args.output, result)
    print(json.dumps({
        key: result[key]
        for key in ("run_count", "persistent_excitation_pass", "control_authority_pass", "data_quality_pass")
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
