#!/usr/bin/env python3
"""Fail-closed 36-run Round 14.2 campaign on one frozen warm Pod."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

from run_control_pilot_campaign import Pilot, atomic_json, sha256, stamp


def derive_control_seed(dataset_id: str, plan_id: str, split_seed: int) -> int:
    material = f"{dataset_id}\0{plan_id}\0control\0{split_seed}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


class FormalCampaign(Pilot):
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.project_root.resolve()
        self.config = json.loads((self.root / args.config).read_text())
        self.namespace = self.config["namespace"]
        self.deployment = self.config["deployment"]
        self.ray_namespace = self.config["ray_namespace"]
        self.ray_pod = self.config["ray_head_pod"]
        self.experiment_id = self.config["dataset_id"]
        self.result_root = Path(self.config["results_root"])
        self.pod_result_root = Path(self.config["pod_results_root"])
        self.pod_dataset_root = Path(self.config["dataset_root"])
        self.state_dir = self.root / ".campaign" / self.experiment_id
        self.manifest_path = self.state_dir / "dataset_run_manifest.json"
        self.ray_source = f"/tmp/{self.experiment_id}-source"
        self.ray_stage = f"/tmp/{self.experiment_id}-stage"
        self.kubectl_command = shlex.split(os.getenv("KUBECTL", "kubectl"))
        self.pod = ""; self.pod_uid = ""

    def plan(self) -> list[dict[str, Any]]:
        rows = []
        for workload in ("balanced", "mixed-bimodal"):
            capacity = float(self.config["capacity"][workload])
            for fraction in self.config["load_fractions"]:
                for arrival in self.config["arrival_processes"]:
                    for split, arrival_seed in self.config["split_seeds"].items():
                        plan_id = f"{workload}-l{int(fraction*100)}-{arrival}-{split}"
                        rows.append({
                            "plan_id": plan_id, "workload": workload, "load_fraction": fraction,
                            "capacity": capacity, "arrival_rate": capacity * fraction,
                            "arrival_process": arrival, "split": split, "arrival_seed": int(arrival_seed),
                            "control_seed": derive_control_seed(self.experiment_id, plan_id, int(arrival_seed)),
                            "status": "PENDING", "run_id": None, "error": None,
                        })
        if len(rows) != 36:
            raise AssertionError("formal matrix must contain exactly 36 runs")
        return rows

    def initialize(self) -> dict[str, Any]:
        paths = (
            "scripts/control_dataset_workload.py", "scripts/run_control_dataset_campaign.py",
            "scripts/run_control_dataset_pipeline.sh", "scripts/build_control_dataset_v1.py",
            "scripts/build_control_snapshots.py", "scripts/validate_control_dataset_run.py",
            "servingrom_pipeline/control_snapshot.py", "configs/servingrom_control_dataset_v1.json",
            "configs/workloads/balanced.yaml", "configs/workloads/mixed-bimodal.yaml",
        )
        hashes = {path: sha256(self.root / path) for path in paths}
        if self.manifest_path.exists():
            value = json.loads(self.manifest_path.read_text())
            if value["source_hashes"] != hashes:
                raise RuntimeError("formal sources changed after campaign manifest creation")
            if any(row["status"] == "RUNNING" for row in value["runs"]):
                raise RuntimeError("manifest contains an interrupted RUNNING run; fail-closed inspection required")
            return value
        value = {
            "schema_version": "servingrom.control_dataset_run_manifest.v1",
            "dataset_id": self.experiment_id, "status": "INITIALIZING", "created_at": stamp(),
            "updated_at": stamp(), "planned_runs": 36, "source_hashes": hashes,
            "topology": self.config["topology"], "runs": self.plan(),
        }
        atomic_json(self.manifest_path, value)
        return value

    def save(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = stamp()
        manifest["counts"] = {name: sum(row["status"] == name for row in manifest["runs"])
                              for name in ("PENDING", "RUNNING", "SEALED", "FAILED")}
        atomic_json(self.manifest_path, manifest)
        if self.pod:
            self.kubectl("exec", self.pod, "--", "mkdir", "-p", str(self.pod_result_root))
            self.kubectl("cp", str(self.manifest_path), f"{self.pod}:{self.pod_result_root}/dataset_run_manifest.json")

    def metadata(self, row: dict[str, Any], run_id: str, workload: dict[str, Any]) -> None:
        deployment = self.kubectl("get", "deployment", self.deployment, "-o", "yaml").stdout
        pod_yaml = self.kubectl("get", "pod", self.pod, "-o", "yaml").stdout
        run_value = {
            **{key: row[key] for key in ("plan_id", "workload", "load_fraction", "arrival_rate", "capacity", "arrival_process", "split", "arrival_seed", "control_seed")},
            "run_id": run_id, "dataset_id": self.experiment_id, "config_id": self.config["config_id"],
            "control_program_version": self.config["control_program_version"],
            "pod": self.pod, "pod_uid": self.pod_uid, "graph_mode": "FULL_DECODE_ONLY",
            "async_scheduling": True, "topology": {"prefill_tp": 2, "decode_a_tp": 2, "decode_b_tp": 2},
        }
        workload_config = json.loads((self.root / "configs" / "workloads" / f"{row['workload']}.yaml").read_text())
        values = {
            "run.json": run_value,
            "workload.json": {**workload_config, "target_arrival_rate": row["arrival_rate"],
                              "load_fraction": row["load_fraction"], "arrival_process": row["arrival_process"],
                              "arrival_seed": row["arrival_seed"], "control_seed": row["control_seed"]},
            "workload_result.json": workload,
            "measurement.json": {"measurement_start_wall_ns": workload["measurement_start_wall_ns"],
                                 "measurement_end_wall_ns": workload["measurement_end_wall_ns"], "snapshot_period_ms": 200},
        }
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for name, value in values.items(): atomic_json(temp / name, value)
            (temp / "deployment.yaml").write_text(deployment); (temp / "pod.yaml").write_text(pod_yaml)
            target = f"{self.pod_result_root}/{run_id}/metadata"
            self.kubectl("exec", self.pod, "--", "mkdir", "-p", target)
            for path in temp.iterdir(): self.kubectl("cp", str(path), f"{self.pod}:{target}/{path.name}")

    def process(self, run_id: str) -> dict[str, Any]:
        source, target = f"{self.pod_result_root}/{run_id}", f"{self.ray_stage}/runs/{run_id}"
        kubectl = shlex.join(self.kubectl_command)
        self.shell(
            f"{kubectl} -n {self.ray_namespace} exec {self.ray_pod} -- mkdir -p {target} && "
            f"{kubectl} -n {self.namespace} exec {self.pod} -- tar -C {source} -cf - . | "
            f"{kubectl} -n {self.ray_namespace} exec -i {self.ray_pod} -- tar -C {target} -xf -", timeout=3600)
        self.ray("bash", f"{self.ray_source}/scripts/run_control_dataset_pipeline.sh", target, timeout=7200)
        quality = json.loads(self.ray("cat", f"{target}/reports/control_dataset_run_quality.json").stdout)
        if not quality["valid"]:
            raise RuntimeError(f"formal control quality failed: {quality}")
        self.shell(
            f"{kubectl} -n {self.ray_namespace} exec {self.ray_pod} -- tar -C {target} -cf - derived reports metadata | "
            f"{kubectl} -n {self.namespace} exec -i {self.pod} -- tar -C {source} -xf -", timeout=3600)
        return quality

    def execute_one(self, row: dict[str, Any], manifest: dict[str, Any]) -> None:
        run_id = f"sr-control-v1-{row['plan_id']}-{stamp()}"
        row.update({"run_id": run_id, "status": "RUNNING", "started_at": stamp(), "finished_at": None, "error": None})
        self.save(manifest); self.run_control("activate", run_id)
        output = f"/tmp/{run_id}-workload.json"
        burst = self.config["on_off_burst"]
        try:
            self.ray(
                "python3", f"{self.ray_source}/scripts/control_dataset_workload.py",
                "--workload-config", f"{self.ray_source}/configs/workloads/{row['workload']}.yaml",
                "--endpoint", f"http://{self.config['service']}:8080", "--tokenize-endpoint", f"http://{self.config['service']}:13700",
                "--dataset-id", self.experiment_id, "--run-id", run_id, "--plan-id", row["plan_id"], "--split", row["split"],
                "--output", output, "--arrival-seed", str(row["arrival_seed"]), "--control-seed", str(row["control_seed"]),
                "--arrival-rate", str(row["arrival_rate"]), "--load-fraction", str(row["load_fraction"]),
                "--arrival-process", row["arrival_process"], "--control-program-version", self.config["control_program_version"],
                "--on-seconds", str(burst["on_seconds"]), "--off-seconds", str(burst["off_seconds"]),
                "--on-multiplier", str(burst["on_multiplier"]), "--warmup-seconds", str(self.config["warmup_seconds"]),
                "--measurement-seconds", str(self.config["measurement_seconds"]),
                "--drain-timeout-seconds", str(self.config["drain_timeout_seconds"]), timeout=2400)
            workload = json.loads(self.ray("cat", output).stdout)
            if not workload["drain"]["drained"] or workload["summary"]["error_count"]:
                raise RuntimeError(f"workload failed or did not drain: {workload['summary']} {workload['drain']}")
            self.metadata(row, run_id, workload)
        finally:
            rollback_error = None
            try: self.ensure_baseline(run_id)
            except Exception as exc: rollback_error = exc
            self.run_control("deactivate", run_id)
            if rollback_error is not None: raise rollback_error
        quality = self.process(run_id)
        current_uid = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.metadata.uid}").stdout.strip()
        restart = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].restartCount}").stdout.strip()
        if current_uid != self.pod_uid or restart != "0":
            raise RuntimeError(f"frozen Pod changed: uid={current_uid} restart={restart}")
        row.update({"status": "SEALED", "finished_at": stamp(), "quality": quality})
        print(f"SEALED {row['plan_id']} {run_id}", flush=True)

    def build_dataset(self, manifest: dict[str, Any]) -> None:
        self.kubectl("cp", str(self.manifest_path), f"{self.ray_pod}:{self.ray_stage}/dataset_run_manifest.json", namespace=self.ray_namespace)
        target = f"{self.ray_stage}/dataset"
        self.ray("python3", f"{self.ray_source}/scripts/build_control_dataset_v1.py",
                 "--manifest", f"{self.ray_stage}/dataset_run_manifest.json", "--runs-root", f"{self.ray_stage}/runs",
                 "--output", target, timeout=7200)
        kubectl = shlex.join(self.kubectl_command)
        self.kubectl("exec", self.pod, "--", "mkdir", "-p", str(self.pod_dataset_root.parent))
        self.shell(
            f"{kubectl} -n {self.ray_namespace} exec {self.ray_pod} -- tar -C {target} -cf - . | "
            f"{kubectl} -n {self.namespace} exec -i {self.pod} -- sh -c 'mkdir -p {self.pod_dataset_root} && tar -C {self.pod_dataset_root} -xf -'", timeout=7200)
        quality = json.loads(self.ray("cat", f"{target}/quality_summary.json").stdout)
        manifest["dataset_quality"] = quality
        manifest["status"] = "SEALED" if quality["control_dataset_ready"] and quality["control_identifiability_ready"] else "STOPPED_QUALITY_GATE"
        self.save(manifest)
        if manifest["status"] != "SEALED":
            raise RuntimeError(f"dataset identifiability gate failed: {quality}")

    def execute(self) -> int:
        manifest = self.initialize(); self.verify_pod(); self.sync_ray()
        manifest.update({"status": "RUNNING", "pod": self.pod, "pod_uid": self.pod_uid}); self.save(manifest)
        for row in manifest["runs"]:
            if row["status"] == "SEALED": continue
            if row["status"] != "PENDING":
                raise RuntimeError(f"formal run is not safely resumable: {row}")
            try:
                self.execute_one(row, manifest); self.save(manifest)
            except Exception as exc:
                row.update({"status": "FAILED", "finished_at": stamp(), "error": repr(exc)})
                manifest["status"] = "STOPPED_FAIL_CLOSED"; self.save(manifest); raise
        manifest["status"] = "COLLECTION_COMPLETE"; self.save(manifest)
        self.build_dataset(manifest)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/servingrom_control_dataset_v1.json")
    return FormalCampaign(parser.parse_args()).execute()


if __name__ == "__main__":
    raise SystemExit(main())
