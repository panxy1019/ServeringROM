#!/usr/bin/env python3
"""Build and analyze the isolated Round 14.3 held-out control benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def effect(low: np.ndarray, high: np.ndarray) -> float | None:
    low, high = low[np.isfinite(low)], high[np.isfinite(high)]
    if len(low) < 2 or len(high) < 2: return None
    pooled = math.sqrt(((len(low)-1)*low.var(ddof=1)+(len(high)-1)*high.var(ddof=1))/(len(low)+len(high)-2))
    return float((high.mean()-low.mean())/pooled) if pooled > 0 else None


def response_lags(u: np.ndarray, state: np.ndarray) -> dict[str, Any]:
    changes = np.flatnonzero(np.abs(np.diff(u)) > 1e-12) + 1; lags = []; settling = []
    threshold = max(float(np.nanstd(state)) * 0.10, 0.05)
    for index in changes:
        direction = np.sign(u[index] - u[index-1]); baseline = float(np.nanmean(state[max(0,index-25):index]))
        segment = (state[index:min(len(state),index+100)] - baseline) * direction
        hits = np.flatnonzero(segment > threshold)
        if len(hits): lags.append(float(hits[0]) * 0.2)
        for cursor in range(max(0, len(segment)-4)):
            if np.all(segment[cursor:cursor+5] > 0): settling.append(float(cursor) * 0.2); break
    return {"transition_count": int(len(changes)),
            "response_lag_median_seconds": float(np.median(lags)) if lags else None,
            "settling_median_seconds": float(np.median(settling)) if settling else None,
            "observable_transition_ratio": len(lags)/len(changes) if len(changes) else None}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runs-root", type=Path, required=True); p.add_argument("--training-quality-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    manifest = json.loads(args.manifest.read_text()); plans = manifest["runs"]
    if len(plans) != 10 or any(row["status"] != "SEALED" for row in plans):
        raise RuntimeError("held-out benchmark requires 10/10 SEALED runs")
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True); arrays = {name: [] for name in ("X","D","U","X_next","U_aux")}
    run_index = []; response_rows = []; slow_rows = []; offset = 0; reference = None
    for plan in plans:
        root = args.runs_root / plan["run_id"]; snap = root / "derived" / "snapshots"; ctl = root / "derived" / "control"
        quality = json.loads((root / "reports" / "control_heldout_run_quality.json").read_text())
        status = json.loads((root / "metadata" / "run_status.json").read_text())
        if not quality["valid"] or status["status"] != "SEALED": raise RuntimeError(f"invalid run {plan['run_id']}")
        current = {"X":np.load(snap/"full_state.npy"), "D":np.load(snap/"disturbance.npy"),
                   "U":np.load(ctl/"control_input.npy"), "X_next":np.load(snap/"next_state.npy"),
                   "U_aux":np.load(ctl/"control_auxiliary.npy")}
        if any(len(value)!=3000 for value in current.values()): raise RuntimeError(f"row count drift {plan['run_id']}")
        for name,value in current.items(): arrays[name].append(value)
        run_index.append({**{key:plan[key] for key in ("plan_id","run_id","workload","load_fraction","trajectory_family","benchmark_class","split","arrival_seed")},
                          "trajectory_seed":str(plan["trajectory_seed"]), "row_start":offset, "row_stop":offset+3000})
        offset += 3000
        controls = pq.read_table(ctl/"control_windows.parquet").to_pylist(); u = current["U"][:,0]
        names = {row["name"]:int(row["index"]) for row in json.loads((snap/"state_index.json").read_text())}
        state_series = {
            "running_imbalance": current["X"][:,names["decode_d1_running_count"]]-current["X"][:,names["decode_d2_running_count"]],
            "waiting_imbalance": current["X"][:,names["decode_d1_waiting_count"]]-current["X"][:,names["decode_d2_waiting_count"]],
            "remaining_imbalance": current["X"][:,names["decode_d1_expected_remaining_tokens"]]-current["X"][:,names["decode_d2_expected_remaining_tokens"]],
        }
        levels = {}
        for level in sorted(set(float(v) for v in u)):
            mask = np.isclose(u,level); selected = [controls[i] for i in np.flatnonzero(mask)]
            routed=sum(row["routed_request_count"] for row in selected); routed_a=sum(row["routed_A_request_count"] for row in selected)
            tokens=sum(row["routed_expected_token_mass"] for row in selected); tokens_a=sum(row["routed_A_expected_token_mass"] for row in selected)
            levels[str(level)]={"windows":int(mask.sum()), "realized_request_ratio":routed_a/routed if routed else None,
                                "realized_token_ratio":tokens_a/tokens if tokens else None,
                                **{f"{name}_mean":float(np.mean(values[mask])) for name,values in state_series.items()}}
        low_level,high_level=(0.4,0.6) if plan["trajectory_family"] in ("interpolation","unseen-composite") else ((0.3,0.7) if plan["trajectory_family"]=="slow-ramp" else (0.2,0.8))
        effects={name:effect(values[np.isclose(u,low_level)],values[np.isclose(u,high_level)]) for name,values in state_series.items()}
        response_rows.append({**{key:plan[key] for key in ("plan_id","run_id","workload","load_fraction","trajectory_family","benchmark_class")},
                              "low_level":low_level,"high_level":high_level,"levels":levels,"effects":effects,
                              "running_response":response_lags(u,state_series["running_imbalance"]),
                              "remaining_response":response_lags(u,state_series["remaining_imbalance"])})
        for row in pq.read_table(ctl/"slow_control_kpi_windows.parquet").to_pylist():
            slow_rows.append({"plan_id":plan["plan_id"],"run_id":plan["run_id"],"split":"test/control-heldout",**row})
        refs={name:path for name,path in {"state_index.json":snap/"state_index.json","disturbance_index.json":snap/"disturbance_index.json",
              "output_index.json":snap/"output_index.json","control_index.json":ctl/"control_index.json",
              "control_auxiliary_index.json":ctl/"control_auxiliary_index.json"}.items()}
        hashes={name:sha256(path) for name,path in refs.items()}
        if reference is None: reference=(refs,hashes)
        elif hashes!=reference[1]: raise RuntimeError(f"schema drift {plan['run_id']}")
    heldout=args.output/"test-control-heldout"; heldout.mkdir()
    for name,parts in arrays.items(): np.save(heldout/f"{name}.npy",np.concatenate(parts))
    pq.write_table(pa.Table.from_pylist(run_index),args.output/"run_index.parquet",compression="zstd")
    pq.write_table(pa.Table.from_pylist(slow_rows),args.output/"slow_kpi_windows.parquet",compression="zstd")
    dump(args.output/"response_statistics.json",response_rows)
    for name,path in reference[0].items(): shutil.copy2(path,args.output/name)

    training=json.loads(args.training_quality_summary.read_text()); training_groups={
        (row["workload"],round(float(row["load_fraction"]),2),row["arrival_process"]):row
        for row in training["control_authority_groups"]}
    interpolation=[]; robustness=[]
    for row in response_rows:
        levels=row["levels"]; low=levels.get("0.4"); high=levels.get("0.6")
        if row["trajectory_family"] in ("interpolation","unseen-composite"):
            known=training_groups[(row["workload"],round(float(row["load_fraction"]),2),"poisson")]
            route_direction=bool(low and high and high["realized_request_ratio"]>low["realized_request_ratio"])
            state_candidates=[]
            for name in ("running_imbalance","remaining_imbalance"):
                d=row["effects"][name]; lo=low[f"{name}_mean"]; hi=high[f"{name}_mean"]
                known_lo=known["effects"][name]["low_mean"]; known_hi=known["effects"][name]["high_mean"]
                span=abs(known_hi-known_lo); between=(known_lo-0.2*span)<=lo<=hi<=(known_hi+0.2*span)
                state_candidates.append({"name":name,"direction":hi>lo,"cohen_d":d,"between_known_extremes":between})
            state_pass=any(item["direction"] and item["cohen_d"] is not None and item["cohen_d"]>=0.25 and item["between_known_extremes"] for item in state_candidates)
            interpolation.append({"plan_id":row["plan_id"],"family":row["trajectory_family"],"route_direction_pass":route_direction,
                                  "state_response_pass":state_pass,"state_candidates":state_candidates,
                                  "running_response":row["running_response"],"remaining_response":row["remaining_response"]})
        if row["trajectory_family"] in ("slow-ramp","boundary-near"):
            ordered=sorted((float(level),value) for level,value in levels.items())
            route=np.asarray([value["realized_request_ratio"] for _,value in ordered],dtype=float)
            running=np.asarray([value["running_imbalance_mean"] for _,value in ordered],dtype=float)
            remaining=np.asarray([value["remaining_imbalance_mean"] for _,value in ordered],dtype=float)
            axis=np.asarray([level for level,_ in ordered]); route_corr=float(np.corrcoef(axis,route)[0,1])
            state_corr=max(float(np.corrcoef(axis,running)[0,1]),float(np.corrcoef(axis,remaining)[0,1]))
            robustness.append({"plan_id":row["plan_id"],"family":row["trajectory_family"],
                               "route_correlation":route_corr,"best_state_correlation":state_corr,
                               "direction_pass":route_corr>=0.7 and state_corr>=0.5,
                               "running_response":row["running_response"],"remaining_response":row["remaining_response"]})
    quality_pass=all(row["quality"]["heldout_control_quality_pass"] for row in plans)
    interpolation_pass=len(interpolation)==8 and all(row["route_direction_pass"] and row["state_response_pass"] for row in interpolation)
    robustness_pass=len(robustness)==2 and all(row["direction_pass"] for row in robustness)
    response_pass=interpolation_pass and robustness_pass
    quality={"schema_version":"servingrom.control_heldout_quality.v1","runs":10,"fast_windows":30000,"slow_windows":1200,
             "split":"test/control-heldout","excluded_from_training":True,
             "heldout_control_quality_pass":quality_pass,"heldout_control_response_pass":response_pass,
             "control_heldout_ready":quality_pass and response_pass,"control_interpolation_ready":interpolation_pass,
             "step15_control_rom_ready":quality_pass and response_pass,"robustness_pass":robustness_pass,
             "interpolation_results":interpolation,"robustness_results":robustness}
    dump(args.output/"quality_summary.json",quality)
    dump(args.output/"CONTROL_HELDOUT_MANIFEST.json",{"benchmark_id":"servingrom-control-heldout-v1","runs":run_index,
         "split":"test/control-heldout","excluded_from_training":True,"training_dataset_reference_sha256":sha256(args.training_quality_summary)})
    report=["# ServingROM Control Held-out v1", "", f"- `heldout_control_quality_pass={str(quality_pass).lower()}`",
            f"- `heldout_control_response_pass={str(response_pass).lower()}`",f"- `control_heldout_ready={str(quality['control_heldout_ready']).lower()}`",
            f"- `control_interpolation_ready={str(interpolation_pass).lower()}`",f"- `step15_control_rom_ready={str(quality['step15_control_rom_ready']).lower()}`",
            "- 该 benchmark 完全属于 `test/control-heldout`，未并入 Control Dataset v1。"]
    (args.output/"CONTROL_HELDOUT_REPORT.md").write_text("\n".join(report)+"\n")
    lines=["# Control Interpolation Analysis","", "| Plan | Family | Route | State |", "|---|---|---:|---:|"]
    lines += [f"| {row['plan_id']} | {row['family']} | {row['route_direction_pass']} | {row['state_response_pass']} |" for row in interpolation]
    (args.output/"CONTROL_INTERPOLATION_ANALYSIS.md").write_text("\n".join(lines)+"\n")
    lines=["# Control Robustness Analysis","", "| Plan | Family | Route corr | State corr | Pass |", "|---|---|---:|---:|---:|"]
    lines += [f"| {row['plan_id']} | {row['family']} | {row['route_correlation']:.3f} | {row['best_state_correlation']:.3f} | {row['direction_pass']} |" for row in robustness]
    (args.output/"CONTROL_ROBUSTNESS_ANALYSIS.md").write_text("\n".join(lines)+"\n")
    files={str(path.relative_to(args.output)):sha256(path) for path in sorted(args.output.rglob("*")) if path.is_file()}
    dump(args.output/"SHA256SUMS.json",files); print(json.dumps(quality,indent=2)); return 0 if quality["control_heldout_ready"] else 2


if __name__ == "__main__": raise SystemExit(main())
