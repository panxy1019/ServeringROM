#!/usr/bin/env python3
"""Fail-closed 10-run Round 14.3 campaign on one warm Control-v1 Pod."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any

from run_control_dataset_campaign import FormalCampaign
from run_control_pilot_campaign import atomic_json, sha256, stamp


def trajectory_seed(benchmark_id: str, plan_id: str, arrival_seed: int) -> int:
    import hashlib
    raw = f"{benchmark_id}\0{plan_id}\0trajectory\0{arrival_seed}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


class HeldoutCampaign(FormalCampaign):
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.project_root.resolve(); self.config = json.loads((self.root / args.config).read_text())
        self.namespace = self.config["namespace"]; self.deployment = self.config["deployment"]
        self.ray_namespace = self.config["ray_namespace"]; self.ray_pod = self.config["ray_head_pod"]
        self.experiment_id = self.config["benchmark_id"]
        self.result_root = Path(self.config["results_root"]); self.pod_result_root = Path(self.config["pod_results_root"])
        self.benchmark_root = Path(self.config["benchmark_root"]); self.training_dataset = Path(self.config["frozen_training_dataset"])
        self.state_dir = self.root / ".campaign" / self.experiment_id
        self.manifest_path = self.state_dir / "CONTROL_HELDOUT_MANIFEST.json"
        self.ray_source = f"/tmp/{self.experiment_id}-source"; self.ray_stage = f"/tmp/{self.experiment_id}-stage"
        self.kubectl_command = shlex.split(os.getenv("KUBECTL", "kubectl")); self.pod = ""; self.pod_uid = ""

    def plan(self) -> list[dict[str, Any]]:
        rows, seed = [], int(self.config["arrival_seed_base"])
        for workload in ("balanced", "mixed-bimodal"):
            capacity = float(self.config["capacity"][workload])
            for fraction in self.config["load_fractions"]:
                for family in self.config["core_trajectories"]:
                    plan_id = f"{workload}-l{int(fraction*100)}-{family}"
                    rows.append({"plan_id": plan_id, "workload": workload, "load_fraction": fraction,
                                 "capacity": capacity, "arrival_rate": capacity * fraction,
                                 "arrival_process": "poisson", "trajectory_family": family,
                                 "benchmark_class": "core", "split": "test/control-heldout",
                                 "arrival_seed": seed, "trajectory_seed": trajectory_seed(self.experiment_id, plan_id, seed),
                                 "status": "PENDING", "run_id": None, "error": None})
                    seed += 1
        for family in self.config["robustness_trajectories"]:
            workload, fraction = "mixed-bimodal", 0.92
            capacity = float(self.config["capacity"][workload]); plan_id = f"{workload}-l92-{family}"
            rows.append({"plan_id": plan_id, "workload": workload, "load_fraction": fraction,
                         "capacity": capacity, "arrival_rate": capacity * fraction,
                         "arrival_process": "poisson", "trajectory_family": family,
                         "benchmark_class": "robustness", "split": "test/control-heldout",
                         "arrival_seed": seed, "trajectory_seed": trajectory_seed(self.experiment_id, plan_id, seed),
                         "status": "PENDING", "run_id": None, "error": None})
            seed += 1
        if len(rows) != 10: raise AssertionError("held-out benchmark must contain 10 runs")
        return rows

    def initialize(self) -> dict[str, Any]:
        paths = (
            "scripts/control_heldout_workload.py", "scripts/run_control_heldout_campaign.py",
            "scripts/run_control_heldout_pipeline.sh", "scripts/build_control_heldout_v1.py",
            "scripts/validate_control_heldout_run.py", "scripts/build_control_snapshots.py",
            "servingrom_pipeline/control_snapshot.py", "configs/servingrom_control_heldout_v1.json",
            "configs/workloads/balanced.yaml", "configs/workloads/mixed-bimodal.yaml",
        )
        hashes = {path: sha256(self.root / path) for path in paths}
        frozen = self.training_dataset / "SHA256SUMS.json"
        if not frozen.exists(): raise RuntimeError(f"frozen Control Dataset v1 is missing: {frozen}")
        if self.manifest_path.exists():
            value = json.loads(self.manifest_path.read_text())
            if value["source_hashes"] != hashes: raise RuntimeError("held-out sources changed after manifest creation")
            if any(row["status"] == "RUNNING" for row in value["runs"]):
                raise RuntimeError("interrupted held-out run requires fail-closed inspection")
            return value
        value = {"schema_version": "servingrom.control_heldout_manifest.v1", "benchmark_id": self.experiment_id,
                 "status": "INITIALIZING", "created_at": stamp(), "updated_at": stamp(), "planned_runs": 10,
                 "split": "test/control-heldout", "excluded_from_training": True,
                 "frozen_training_dataset": str(self.training_dataset),
                 "frozen_training_sha256_manifest": sha256(frozen), "source_hashes": hashes,
                 "topology": self.config["topology"], "runs": self.plan()}
        atomic_json(self.manifest_path, value); return value

    def save(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = stamp()
        manifest["counts"] = {name: sum(row["status"] == name for row in manifest["runs"])
                              for name in ("PENDING", "RUNNING", "SEALED", "FAILED")}
        atomic_json(self.manifest_path, manifest)
        if self.pod:
            self.kubectl("exec", self.pod, "--", "mkdir", "-p", str(self.pod_result_root))
            self.kubectl("cp", str(self.manifest_path), f"{self.pod}:{self.pod_result_root}/CONTROL_HELDOUT_MANIFEST.json")

    def metadata(self, row: dict[str, Any], run_id: str, workload: dict[str, Any]) -> None:
        deployment = self.kubectl("get", "deployment", self.deployment, "-o", "yaml").stdout
        pod_yaml = self.kubectl("get", "pod", self.pod, "-o", "yaml").stdout
        run_value = {**{key: row[key] for key in (
            "plan_id", "workload", "load_fraction", "arrival_rate", "capacity", "arrival_process",
            "trajectory_family", "benchmark_class", "split", "arrival_seed", "trajectory_seed",
            "pod_uid_before", "pod_uid_after", "restart_before", "restart_after")},
            "run_id": run_id, "benchmark_id": self.experiment_id, "config_id": self.config["config_id"],
            "trajectory_program_version": self.config["trajectory_program_version"],
            "graph_mode": "FULL_DECODE_ONLY", "async_scheduling": True,
            "topology": {"prefill_tp": 2, "decode_a_tp": 2, "decode_b_tp": 2}}
        base_workload = json.loads((self.root / "configs" / "workloads" / f"{row['workload']}.yaml").read_text())
        values = {"run.json": run_value,
                  "workload.json": {**base_workload, "target_arrival_rate": row["arrival_rate"],
                                    "load_fraction": row["load_fraction"], "arrival_process": "poisson",
                                    "arrival_seed": row["arrival_seed"], "trajectory_seed": row["trajectory_seed"]},
                  "workload_result.json": workload,
                  "measurement.json": {"measurement_start_wall_ns": workload["measurement_start_wall_ns"],
                                       "measurement_end_wall_ns": workload["measurement_end_wall_ns"], "snapshot_period_ms": 200}}
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for name, value in values.items(): atomic_json(temp / name, value)
            (temp / "deployment.yaml").write_text(deployment); (temp / "pod.yaml").write_text(pod_yaml)
            target = f"{self.pod_result_root}/{run_id}/metadata"; self.kubectl("exec", self.pod, "--", "mkdir", "-p", target)
            for path in temp.iterdir(): self.kubectl("cp", str(path), f"{self.pod}:{target}/{path.name}")

    def process(self, run_id: str) -> dict[str, Any]:
        source, target = f"{self.pod_result_root}/{run_id}", f"{self.ray_stage}/runs/{run_id}"
        kubectl = shlex.join(self.kubectl_command)
        self.shell(f"{kubectl} -n {self.ray_namespace} exec {self.ray_pod} -- mkdir -p {target} && "
                   f"{kubectl} -n {self.namespace} exec {self.pod} -- tar -C {source} -cf - . | "
                   f"{kubectl} -n {self.ray_namespace} exec -i {self.ray_pod} -- tar -C {target} -xf -", timeout=3600)
        self.ray("bash", f"{self.ray_source}/scripts/run_control_heldout_pipeline.sh", target, timeout=7200)
        quality = json.loads(self.ray("cat", f"{target}/reports/control_heldout_run_quality.json").stdout)
        if not quality["valid"]: raise RuntimeError(f"held-out quality failed: {quality}")
        self.shell(f"{kubectl} -n {self.ray_namespace} exec {self.ray_pod} -- tar -C {target} -cf - derived reports metadata | "
                   f"{kubectl} -n {self.namespace} exec -i {self.pod} -- tar -C {source} -xf -", timeout=3600)
        return quality

    def execute_one(self, row: dict[str, Any], manifest: dict[str, Any]) -> None:
        run_id = f"sr-control-heldout-{row['plan_id']}-{stamp()}"
        row.update({"run_id": run_id, "status": "RUNNING", "started_at": stamp(), "finished_at": None,
                    "error": None, "pod_uid_before": self.pod_uid, "restart_before": 0})
        self.save(manifest); self.run_control("activate", run_id); output = f"/tmp/{run_id}-workload.json"
        try:
            self.ray("python3", f"{self.ray_source}/scripts/control_heldout_workload.py",
                     "--workload-config", f"{self.ray_source}/configs/workloads/{row['workload']}.yaml",
                     "--endpoint", f"http://{self.config['service']}:8080", "--tokenize-endpoint", f"http://{self.config['service']}:13700",
                     "--benchmark-id", self.experiment_id, "--run-id", run_id, "--plan-id", row["plan_id"], "--output", output,
                     "--trajectory-family", row["trajectory_family"], "--arrival-seed", str(row["arrival_seed"]),
                     "--trajectory-seed", str(row["trajectory_seed"]), "--arrival-rate", str(row["arrival_rate"]),
                     "--load-fraction", str(row["load_fraction"]), "--trajectory-program-version", self.config["trajectory_program_version"],
                     "--warmup-seconds", str(self.config["warmup_seconds"]), "--measurement-seconds", str(self.config["measurement_seconds"]),
                     "--drain-timeout-seconds", str(self.config["drain_timeout_seconds"]), timeout=2400)
            workload = json.loads(self.ray("cat", output).stdout)
            if not workload["drain"]["drained"] or workload["summary"]["error_count"]:
                raise RuntimeError(f"held-out workload failed: {workload['summary']} {workload['drain']}")
            row["pod_uid_after"] = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.metadata.uid}").stdout.strip()
            row["restart_after"] = int(self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].restartCount}").stdout)
            self.metadata(row, run_id, workload)
        finally:
            rollback_error = None
            try: self.ensure_baseline(run_id)
            except Exception as exc: rollback_error = exc
            self.run_control("deactivate", run_id)
            if rollback_error is not None: raise rollback_error
        quality = self.process(run_id)
        current_uid = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.metadata.uid}").stdout.strip()
        current_restart = int(self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].restartCount}").stdout)
        if row["pod_uid_after"] != self.pod_uid or current_uid != self.pod_uid or row["restart_after"] != 0 or current_restart != 0:
            raise RuntimeError(f"frozen Pod changed: uid={current_uid} restart={current_restart}")
        row.update({"status": "SEALED", "finished_at": stamp(), "quality": quality})
        print(f"SEALED {row['plan_id']} {run_id}", flush=True)

    def build_benchmark(self, manifest: dict[str, Any]) -> None:
        self.kubectl("cp", str(self.manifest_path), f"{self.ray_pod}:{self.ray_stage}/CONTROL_HELDOUT_MANIFEST.json", namespace=self.ray_namespace)
        training_quality = self.training_dataset / "quality_summary.json"
        self.kubectl("cp", str(training_quality), f"{self.ray_pod}:{self.ray_stage}/training_quality_summary.json", namespace=self.ray_namespace)
        target = f"{self.ray_stage}/benchmark"; self.ray("rm", "-rf", target, timeout=600)
        self.ray("python3", f"{self.ray_source}/scripts/build_control_heldout_v1.py",
                 "--manifest", f"{self.ray_stage}/CONTROL_HELDOUT_MANIFEST.json", "--runs-root", f"{self.ray_stage}/runs",
                 "--training-quality-summary", f"{self.ray_stage}/training_quality_summary.json",
                 "--output", target, timeout=7200)
        if self.benchmark_root.exists(): raise FileExistsError(f"immutable benchmark exists: {self.benchmark_root}")
        self.benchmark_root.mkdir(parents=True)
        files = self.ray("find", target, "-type", "f", "-printf", "%P\\n").stdout.splitlines()
        for relative in files:
            destination = self.benchmark_root / relative; destination.parent.mkdir(parents=True, exist_ok=True)
            self.kubectl("cp", "-c", "ray-head", f"{self.ray_pod}:{target}/{relative}", str(destination), namespace=self.ray_namespace, timeout=1800)
        hashes = json.loads((self.benchmark_root / "SHA256SUMS.json").read_text())
        bad = [path for path, expected in hashes.items() if sha256(self.benchmark_root / path) != expected]
        if bad: raise RuntimeError(f"benchmark copy SHA256 mismatch: {bad}")
        quality = json.loads((self.benchmark_root / "quality_summary.json").read_text())
        manifest["benchmark_quality"] = quality
        manifest["status"] = "SEALED" if quality["control_heldout_ready"] and quality["control_interpolation_ready"] else "STOPPED_QUALITY_GATE"
        self.save(manifest)
        if manifest["status"] != "SEALED": raise RuntimeError(f"held-out benchmark gate failed: {quality}")

    def execute(self) -> int:
        manifest = self.initialize(); self.verify_pod(); self.sync_ray()
        manifest.update({"status": "RUNNING", "pod": self.pod, "pod_uid": self.pod_uid}); self.save(manifest)
        for row in manifest["runs"]:
            if row["status"] == "SEALED": continue
            if row["status"] != "PENDING": raise RuntimeError(f"held-out run is not resumable: {row}")
            try: self.execute_one(row, manifest); self.save(manifest)
            except Exception as exc:
                row.update({"status": "FAILED", "finished_at": stamp(), "error": repr(exc)})
                manifest["status"] = "STOPPED_FAIL_CLOSED"; self.save(manifest); raise
        manifest["status"] = "COLLECTION_COMPLETE"; self.save(manifest); self.build_benchmark(manifest); return 0


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--config", default="configs/servingrom_control_heldout_v1.json")
    return HeldoutCampaign(p.parse_args()).execute()


if __name__ == "__main__": raise SystemExit(main())
