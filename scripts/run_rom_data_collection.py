#!/usr/bin/env python3
"""Fail-closed controller for the warm-Pod ServingROM ROM Dataset v1 campaign."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


TRANSIENT_PATTERNS = ("step-up", "step-down", "ramp-up", "held-out-composite")
FATAL_PATTERNS = (
    "out of memory", "oom", "engine dead", "engine death", "enginecore died",
    "mooncake.*fatal", "segmentation fault", "hccl.*error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug(value: str) -> str:
    return value.replace("_", "-").replace(".", "p")


def build_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for workload in ("balanced", "long-prefill", "mixed-bimodal"):
        for fraction in config["load_fractions"]:
            for arrival in config["arrival_processes"]:
                for seed_text, split in config["splits"].items():
                    seed = int(seed_text)
                    rows.append({
                        "plan_id": f"core-{workload}-{arrival}-l{fraction:.2f}-seed{seed}",
                        "workload": workload, "arrival_process": arrival,
                        "load_fraction": float(fraction), "seed": seed, "split": split,
                        "transient_pattern": None, "status": "PENDING", "attempts": [],
                    })
    transient_seed = 401
    for workload in ("balanced", "long-prefill", "mixed-bimodal"):
        for pattern in TRANSIENT_PATTERNS:
            rows.append({
                "plan_id": f"transient-{workload}-{pattern}", "workload": workload,
                "arrival_process": pattern, "load_fraction": None,
                "seed": transient_seed, "split": "test", "transient_pattern": pattern,
                "status": "PENDING", "attempts": [],
            })
            transient_seed += 1
    if len(rows) != 84 or len({row["plan_id"] for row in rows}) != 84:
        raise AssertionError("formal plan must contain exactly 72 core and 12 transient runs")
    return rows


class Campaign:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.project_root.resolve()
        self.config_path = (self.root / args.config).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.namespace = self.config["namespace"]
        self.experiment = self.config["experiment_deployment"]
        self.production = self.config["production_deployment"]
        self.ray_namespace = self.config["ray_namespace"]
        self.ray_pod = self.config["ray_head_pod"]
        self.experiment_id = self.config["experiment_id"]
        self.dataset_root = Path(self.config["dataset_root"])
        self.pod_dataset_root = Path("/servingrom-results/datasets") / self.config["dataset_id"]
        self.local_state = self.root / ".campaign" / self.config["dataset_id"]
        self.progress_path = self.local_state / "collection_progress.json"
        self.capacity_path = self.local_state / "capacity_summary.json"
        self.ray_source = f"/tmp/{self.config['dataset_id']}-source"
        self.ray_stage = f"/tmp/{self.config['dataset_id']}-stage"
        self.pod: str | None = None

    def run(self, command: list[str], *, check: bool = True, capture: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        if self.args.dry_run:
            print("DRY-RUN", shlex.join(command), flush=True)
            return subprocess.CompletedProcess(command, 0, "", "")
        try:
            return subprocess.run(command, check=check, text=True, capture_output=capture, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            print(f"COMMAND FAILED: {shlex.join(command)}", flush=True)
            if exc.stdout:
                print(exc.stdout, flush=True)
            if exc.stderr:
                print(exc.stderr, flush=True)
            raise

    def kubectl(self, *parts: str, namespace: str | None = None, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        command = ["kubectl"]
        selected = self.namespace if namespace is None else namespace
        if selected:
            command += ["-n", selected]
        command += list(parts)
        return self.run(command, **kwargs)

    def shell(self, command: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return self.run(["/bin/bash", "-lc", command], **kwargs)

    def verify_frozen_inputs(self) -> dict[str, str]:
        frozen = self.config["frozen"]
        files = {
            "state_index_sha256": self.root / "configs/frozen" / self.config["dataset_id"] / "state_index.json",
            "bin_schema_sha256": self.root / "configs/frozen" / self.config["dataset_id"] / "bin_schema.yaml",
            "snapshot_builder_sha256": self.root / "servingrom_pipeline/snapshot_builder.py",
            "deployment_config_sha256": self.root / "k8s/servingrom/qwen36-servingrom-d2.yaml",
        }
        actual = {name: sha256(path) for name, path in files.items()}
        for name, value in actual.items():
            if not frozen.get(name) or frozen[name] != value:
                raise RuntimeError(f"frozen hash mismatch for {name}: expected={frozen.get(name)} actual={value}")
        actual["dataset_config_sha256"] = sha256(self.config_path)
        for name, relative in (
            ("workload_generator_sha256", "scripts/rom_workload.py"),
            ("campaign_controller_sha256", "scripts/run_rom_data_collection.py"),
            ("dataset_builder_sha256", "scripts/build_rom_dataset.py"),
            ("snapshot_validation_sha256", "servingrom_pipeline/snapshot_validation.py"),
        ):
            actual[name] = sha256(self.root / relative)
        for workload in ("balanced", "long-prefill", "mixed-bimodal"):
            actual[f"workload_{workload}_sha256"] = sha256(self.root / "configs/workloads" / f"{workload}.yaml")
        return actual

    def initialize_progress(self, hashes: dict[str, str]) -> dict[str, Any]:
        if self.progress_path.exists():
            progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if progress["frozen_hashes"] != hashes:
                raise RuntimeError("campaign source hashes changed after collection began")
            return progress
        progress = {
            "schema_version": "servingrom.collection_progress.v1",
            "dataset_id": self.config["dataset_id"], "planned_runs": 84,
            "core_runs": 72, "transient_runs": 12, "runs": build_plan(self.config),
            "frozen_hashes": hashes, "started_at": utc_stamp(), "updated_at": utc_stamp(),
            "status": "INITIALIZING",
        }
        atomic_json(self.progress_path, progress)
        return progress

    def update_progress(self, progress: dict[str, Any]) -> None:
        progress["updated_at"] = utc_stamp()
        progress["counts"] = {
            status: sum(row["status"] == status for row in progress["runs"])
            for status in ("PENDING", "RUNNING", "SEALED", "INVALID")
        }
        atomic_json(self.progress_path, progress)
        if self.pod and not self.args.dry_run:
            self.kubectl("exec", self.pod, "--", "mkdir", "-p", str(self.pod_dataset_root))
            self.kubectl("cp", str(self.progress_path), f"{self.pod}:{self.pod_dataset_root}/collection_progress.json")

    def create_configmaps(self) -> None:
        telemetry_files = " ".join(
            f"--from-file={shlex.quote(str(path))}" for path in sorted((self.root / "servingrom_telemetry").glob("*.py"))
        )
        command = (
            f"kubectl -n {shlex.quote(self.namespace)} create configmap servingrom-telemetry-hot-v1 "
            f"{telemetry_files} --dry-run=client -o yaml | kubectl apply -f -"
        )
        self.shell(command)
        entrypoint = self.root / "scripts/servingrom/pd-worker-entrypoint-instrumented.sh"
        self.shell(
            f"kubectl -n {shlex.quote(self.namespace)} create configmap servingrom-entrypoint-hot-v1 "
            f"--from-file=pd-worker-entrypoint-instrumented.sh={shlex.quote(str(entrypoint))} "
            "--dry-run=client -o yaml | kubectl apply -f -"
        )
        files = {
            "pd_proxy.py": self.root / "scripts/pd_proxy.py",
            "discover_npu_mapping.py": self.root / "scripts/discover_npu_mapping.py",
            "servingrom_run_control.py": self.root / "scripts/servingrom_run_control.py",
        }
        options = " ".join(f"--from-file={name}={shlex.quote(str(path))}" for name, path in files.items())
        self.shell(
            f"kubectl -n {shlex.quote(self.namespace)} create configmap qwen36-pd-servingrom-d2-scripts {options} "
            "--dry-run=client -o yaml | kubectl apply -f -"
        )

    def sync_source_to_ray(self) -> None:
        selected = "scripts servingrom_pipeline servingrom_telemetry configs"
        command = (
            f"kubectl -n {shlex.quote(self.ray_namespace)} exec {shlex.quote(self.ray_pod)} -- "
            f"mkdir -p {shlex.quote(self.ray_source)} {shlex.quote(self.ray_stage)}/runs && "
            f"tar -C {shlex.quote(str(self.root))} -cf - {selected} | "
            f"kubectl -n {shlex.quote(self.ray_namespace)} exec -i {shlex.quote(self.ray_pod)} -- "
            f"tar -C {shlex.quote(self.ray_source)} -xf -"
        )
        self.shell(command)

    def prepare_warm_pod(self) -> None:
        if not self.args.dry_run:
            ready = self.kubectl(
                "get", "deployment", self.experiment,
                "-o", "jsonpath={.status.readyReplicas}", check=False,
            ).stdout.strip()
            if ready == "1":
                self.pod = self.kubectl(
                    "get", "pod", "-l", f"app={self.experiment}",
                    "-o", "jsonpath={.items[0].metadata.name}",
                ).stdout.strip()
                self.verify_warm_pod()
                self.sync_source_to_ray()
                return
        self.create_configmaps()
        self.kubectl("apply", "-f", str(self.root / "k8s/servingrom/qwen36-servingrom-d2.yaml"))
        self.kubectl("apply", "-f", str(self.root / "k8s/servingrom/rom-results-helper.yaml"))
        self.kubectl("scale", f"deployment/{self.production}", "--replicas=0")
        self.kubectl("rollout", "status", f"deployment/{self.production}", "--timeout=15m")
        self.kubectl("scale", f"deployment/{self.experiment}", "--replicas=1")
        self.kubectl("rollout", "status", f"deployment/{self.experiment}", "--timeout=90m", timeout=5500)
        if self.args.dry_run:
            self.pod = "DRY-RUN-POD"
            return
        self.pod = self.kubectl(
            "get", "pod", "-l", f"app={self.experiment}",
            "-o", "jsonpath={.items[0].metadata.name}",
        ).stdout.strip()
        if not self.pod:
            raise RuntimeError("experiment Pod was not found after rollout")
        self.verify_warm_pod()
        self.sync_source_to_ray()

    def verify_warm_pod(self) -> None:
        assert self.pod
        restart = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].restartCount}").stdout.strip()
        image_id = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].imageID}").stdout.strip()
        if restart != "0":
            raise RuntimeError(f"experiment Pod restartCount is {restart}")
        if self.config["image_digest"].split(":", 1)[1] not in image_id:
            raise RuntimeError(f"runtime image digest mismatch: {image_id}")
        effective = self.kubectl("exec", self.pod, "--", "cat", "/var/run/qwen36-pd/effective-config.txt").stdout
        if "FULL_DECODE_ONLY" not in effective or "--async-scheduling" not in effective:
            raise RuntimeError(f"frozen D2 configuration is not active:\n{effective}")
        mapping = self.kubectl("exec", self.pod, "--", "cat", "/var/run/qwen36-pd/service-device-map.txt").stdout
        if not all(name in mapping for name in ("prefill=", "decode_a=", "decode_b=")):
            raise RuntimeError(f"incomplete NPU mapping:\n{mapping}")

    def run_control(self, action: str, run_id: str) -> dict[str, Any]:
        assert self.pod
        result = self.kubectl(
            "exec", self.pod, "--", "python3", "/opt/qwen36-pd/servingrom_run_control.py",
            action, "--run-id", run_id,
            "--experiment-id", self.experiment_id,
            "--config-id", self.config["config_id"],
            "--timeout", "120",
            timeout=180,
        )
        value = json.loads(result.stdout) if result.stdout else {}
        print(f"RUN CONTROL {action}: run_id={run_id} ack={value.get('acknowledged', len(value.get('acks', [])))}", flush=True)
        return value

    def ray_exec(self, *command: str, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        return self.kubectl("exec", self.ray_pod, "--", *command, namespace=self.ray_namespace, timeout=timeout)

    def run_calibration(self) -> dict[str, Any]:
        if self.capacity_path.exists():
            return json.loads(self.capacity_path.read_text(encoding="utf-8"))
        reports = {}
        calibration_runs = []
        for workload in ("balanced", "long-prefill", "mixed-bimodal"):
            run_id = f"sr-v1-capacity-{slug(workload)}-{utc_stamp()}"
            calibration_runs.append(run_id)
            self.run_control("activate", run_id)
            print(f"CALIBRATION START: workload={workload} run_id={run_id}", flush=True)
            try:
                candidates = self.config["calibration"][f"{workload}_candidates"]
                output = f"/tmp/{run_id}-{workload}.json"
                command = [
                    "python3", f"{self.ray_source}/scripts/rom_workload.py", "--mode", "calibration",
                    "--workload-config", f"{self.ray_source}/configs/workloads/{workload}.yaml",
                    "--endpoint", f"http://{self.config['experiment_service']}:8080",
                    "--tokenize-endpoint", f"http://{self.config['experiment_service']}:13700",
                    "--run-id", run_id, "--output", output, "--seed", "17",
                    "--candidate-rates", ",".join(map(str, candidates)),
                    "--measurement-seconds", str(self.config["calibration"]["duration_seconds"]),
                    "--drain-timeout-seconds", str(self.config["drain_timeout_seconds"]),
                    "--reject-rate-max", str(self.config["calibration"]["reject_rate_max"]),
                    "--error-rate-max", str(self.config["calibration"]["error_rate_max"]),
                    "--backlog-slope-max", str(self.config["calibration"]["backlog_slope_max_per_second"]),
                ]
                self.ray_exec(*command, timeout=7200)
                reports[workload] = json.loads(self.ray_exec("cat", output).stdout)
                print(f"CALIBRATION SEALED: workload={workload} lambda_stable={reports[workload]['lambda_stable']}", flush=True)
            finally:
                self.run_control("deactivate", run_id)
        result = {
            "schema_version": "servingrom.capacity_summary.v1", "run_ids": calibration_runs,
            "created_at": utc_stamp(), "workloads": reports,
            "lambda_stable": {name: value["lambda_stable"] for name, value in reports.items()},
        }
        atomic_json(self.capacity_path, result)
        assert self.pod
        self.kubectl("cp", str(self.capacity_path), f"{self.pod}:{self.pod_dataset_root}/capacity_summary.json")
        return result

    def workload_command(self, row: dict[str, Any], run_id: str, rate: float, output: str) -> list[str]:
        return [
            "python3", f"{self.ray_source}/scripts/rom_workload.py", "--mode", "formal",
            "--workload-config", f"{self.ray_source}/configs/workloads/{row['workload']}.yaml",
            "--endpoint", f"http://{self.config['experiment_service']}:8080",
            "--tokenize-endpoint", f"http://{self.config['experiment_service']}:13700",
            "--run-id", run_id, "--output", output, "--seed", str(row["seed"]),
            "--arrival-rate", str(rate), "--arrival-process", row["arrival_process"],
            "--warmup-seconds", str(self.config["warmup_seconds"]),
            "--measurement-seconds", str(self.config["measurement_seconds"]),
            "--drain-timeout-seconds", str(self.config["drain_timeout_seconds"]),
            "--snapshot-period-ms", str(self.config["snapshot_period_ms"]),
            "--on-seconds", str(self.config["on_off_burst"]["on_seconds"]),
            "--off-seconds", str(self.config["on_off_burst"]["off_seconds"]),
            "--on-multiplier", str(self.config["on_off_burst"]["on_rate_multiplier"]),
        ]

    def write_run_metadata(self, run_id: str, row: dict[str, Any], result: dict[str, Any], rate: float, hashes: dict[str, str]) -> None:
        assert self.pod
        pod_uid = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.metadata.uid}").stdout.strip()
        image_id = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].imageID}").stdout.strip()
        workload_config = json.loads((self.root / "configs/workloads" / f"{row['workload']}.yaml").read_text(encoding="utf-8"))
        metadata = {
            "run.yaml": {
                "dataset_id": self.config["dataset_id"], "experiment_id": self.experiment_id,
                "run_id": run_id, "config_id": self.config["config_id"], "plan_id": row["plan_id"],
                "workload": row["workload"], "arrival_process": row["arrival_process"],
                "target_arrival_rate": rate, "lambda_stable": row["lambda_stable"],
                "load_fraction": row.get("load_fraction"), "seed": row["seed"], "split": row["split"],
                "transient_pattern": row.get("transient_pattern"), "image": self.config["image"],
                "image_digest": self.config["image_digest"], "runtime_image_id": image_id,
                "pod": self.pod, "pod_uid": pod_uid, "graph_mode": "FULL_DECODE_ONLY",
                "async_scheduling": True, "prefill_tp": 2, "decode_tp": 2,
                "snapshot_period_ms": self.config["snapshot_period_ms"], "frozen_hashes": hashes,
            },
            "workload.json": {**workload_config, "target_arrival_rate": rate, "lambda_stable": row["lambda_stable"], "load_fraction": row.get("load_fraction"), "arrival_process": row["arrival_process"], "seed": row["seed"]},
            "workload_result.json": result,
            "measurement.json": {
                "measurement_start_wall_ns": result["measurement_start_wall_ns"],
                "measurement_end_wall_ns": result["measurement_end_wall_ns"],
                "snapshot_period_ms": self.config["snapshot_period_ms"],
            },
            "frozen_schema.json": {
                "state_index_sha256": self.config["frozen"]["state_index_sha256"],
                "bin_schema_sha256": self.config["frozen"]["bin_schema_sha256"],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in metadata.items():
                atomic_json(root / name, value)
            self.kubectl("exec", self.pod, "--", "mkdir", "-p", f"/servingrom-results/{self.experiment_id}/{run_id}/metadata")
            for path in root.iterdir():
                self.kubectl("cp", str(path), f"{self.pod}:/servingrom-results/{self.experiment_id}/{run_id}/metadata/{path.name}")
        deployment = self.kubectl("get", "deployment", self.experiment, "-o", "yaml").stdout
        pod_yaml = self.kubectl("get", "pod", self.pod, "-o", "yaml").stdout
        for name, content in (("deployment.yaml", deployment), ("pod.yaml", pod_yaml)):
            encoded = content.encode().hex()
            self.kubectl("exec", self.pod, "--", "python3", "-c", f"open('/servingrom-results/{self.experiment_id}/{run_id}/metadata/{name}','wb').write(bytes.fromhex('{encoded}'))")

    def copy_run_to_ray(self, run_id: str) -> str:
        assert self.pod
        source = f"/servingrom-results/{self.experiment_id}/{run_id}"
        target = f"{self.ray_stage}/runs/{run_id}"
        command = (
            f"kubectl -n {shlex.quote(self.ray_namespace)} exec {shlex.quote(self.ray_pod)} -- mkdir -p {shlex.quote(target)} && "
            f"kubectl -n {shlex.quote(self.namespace)} exec {shlex.quote(self.pod)} -- tar -C {shlex.quote(source)} -cf - . | "
            f"kubectl -n {shlex.quote(self.ray_namespace)} exec -i {shlex.quote(self.ray_pod)} -- tar -C {shlex.quote(target)} -xf -"
        )
        self.shell(command, timeout=3600)
        return target

    def process_and_seal(self, run_id: str) -> dict[str, Any]:
        target = self.copy_run_to_ray(run_id)
        self.ray_exec("bash", f"{self.ray_source}/scripts/run_snapshot_phase_a.sh", target, timeout=7200)
        status = json.loads(self.ray_exec("cat", f"{target}/metadata/run_status.json").stdout)
        if status.get("status") != "SEALED" or not status.get("eligible_for_training"):
            raise RuntimeError(f"run failed fail-closed seal: {json.dumps(status.get('reasons'), ensure_ascii=False)}")
        assert self.pod
        command = (
            f"kubectl -n {shlex.quote(self.ray_namespace)} exec {shlex.quote(self.ray_pod)} -- "
            f"tar -C {shlex.quote(target)} -cf - derived reports metadata | "
            f"kubectl -n {shlex.quote(self.namespace)} exec -i {shlex.quote(self.pod)} -- "
            f"tar -C /servingrom-results/{shlex.quote(self.experiment_id)}/{shlex.quote(run_id)} -xf -"
        )
        self.shell(command, timeout=3600)
        return status

    def fatal_runtime_evidence(self) -> str:
        assert self.pod
        logs = self.kubectl("logs", self.pod, "--tail=5000", check=False).stdout.lower()
        return "\n".join(line for line in logs.splitlines() if any(pattern.replace(".*", "") in line for pattern in FATAL_PATTERNS))

    def execute_one(self, row: dict[str, Any], hashes: dict[str, str]) -> None:
        load = "transient" if row["transient_pattern"] else f"{row['load_fraction']:.2f}"
        run_id = f"sr-v1-{slug(row['workload'])}-{slug(row['arrival_process'])}-l{slug(load)}-seed{row['seed']}-{utc_stamp()}"
        row["status"] = "RUNNING"
        row["run_id"] = run_id
        row["attempts"].append({"run_id": run_id, "started_at": utc_stamp(), "status": "RUNNING"})
        print(f"FORMAL RUN START: plan_id={row['plan_id']} run_id={run_id}", flush=True)
        self.run_control("activate", run_id)
        output = f"/tmp/{run_id}-workload.json"
        try:
            self.ray_exec(*self.workload_command(row, run_id, row["target_arrival_rate"], output), timeout=3600)
            result = json.loads(self.ray_exec("cat", output).stdout)
            if not result.get("drain", {}).get("drained"):
                raise RuntimeError(f"run did not drain: {result.get('drain')}")
            self.write_run_metadata(run_id, row, result, row["target_arrival_rate"], hashes)
        finally:
            self.run_control("deactivate", run_id)
        fatal = self.fatal_runtime_evidence()
        if fatal:
            raise RuntimeError(f"systemic runtime failure after workload:\n{fatal}")
        status = self.process_and_seal(run_id)
        row.update({
            "status": "SEALED", "image_digest": self.config["image_digest"],
            "git_commit": self.git_commit(), "schema_hash": hashes["state_index_sha256"],
            "sealed_at": utc_stamp(),
        })
        row["attempts"][-1].update({"status": "SEALED", "finished_at": utc_stamp()})
        print(f"FORMAL RUN SEALED: plan_id={row['plan_id']} run_id={run_id}", flush=True)

    def git_commit(self) -> str:
        result = self.run(["git", "rev-parse", "HEAD"], check=False)
        return result.stdout.strip() or "uncommitted"

    def restore_production(self) -> None:
        self.kubectl("scale", f"deployment/{self.experiment}", "--replicas=0", check=False)
        self.kubectl("rollout", "status", f"deployment/{self.experiment}", "--timeout=15m", check=False)
        self.kubectl("scale", f"deployment/{self.production}", "--replicas=1", check=False)
        self.kubectl("rollout", "status", f"deployment/{self.production}", "--timeout=90m", check=False, timeout=5500)

    def execute(self) -> int:
        hashes = self.verify_frozen_inputs()
        progress = self.initialize_progress(hashes)
        if self.args.plan_only:
            print(json.dumps({
                "dataset_id": progress["dataset_id"], "planned_runs": progress["planned_runs"],
                "core_runs": progress["core_runs"], "transient_runs": progress["transient_runs"],
                "frozen_hashes": hashes,
            }, indent=2))
            return 0
        self.prepare_warm_pod()
        progress["status"] = "CALIBRATING"
        self.update_progress(progress)
        capacity = self.run_calibration()
        for row in progress["runs"]:
            row["lambda_stable"] = float(capacity["lambda_stable"][row["workload"]])
            row["target_arrival_rate"] = (
                row["lambda_stable"] if row["transient_pattern"]
                else row["lambda_stable"] * row["load_fraction"]
            )
        progress["status"] = "COLLECTING"
        self.update_progress(progress)
        completed_this_invocation = 0
        try:
            for row in progress["runs"]:
                if row["status"] == "SEALED":
                    continue
                if row["status"] == "RUNNING":
                    row["status"] = "PENDING"
                for attempt in range(len(row["attempts"]), 2):
                    try:
                        self.execute_one(row, hashes)
                        self.update_progress(progress)
                        completed_this_invocation += 1
                        break
                    except Exception as exc:
                        evidence = self.fatal_runtime_evidence() if self.pod else ""
                        row["attempts"][-1].update({"status": "INVALID", "finished_at": utc_stamp(), "error": repr(exc)})
                        row["status"] = "INVALID" if attempt == 1 or evidence or "fail-closed seal" in str(exc) else "PENDING"
                        self.update_progress(progress)
                        if row["status"] == "INVALID":
                            progress["status"] = "STOPPED_FAIL_CLOSED"
                            self.update_progress(progress)
                            raise
                    if self.args.max_runs and completed_this_invocation >= self.args.max_runs:
                        progress["status"] = "PAUSED"
                        self.update_progress(progress)
                        return 0
            progress["status"] = "COLLECTION_COMPLETE"
            self.update_progress(progress)
            self.kubectl("cp", str(self.progress_path), f"{self.ray_pod}:{self.ray_stage}/collection_progress.json", namespace=self.ray_namespace)
            self.kubectl("cp", str(self.capacity_path), f"{self.ray_pod}:{self.ray_stage}/capacity_summary.json", namespace=self.ray_namespace)
            # Dataset merge is deliberately invoked only after every planned run is SEALED.
            self.ray_exec(
                "python3", f"{self.ray_source}/scripts/build_rom_dataset.py",
                "--staging-root", self.ray_stage, "--output-root", f"{self.ray_stage}/dataset",
                "--progress", f"{self.ray_stage}/collection_progress.json",
                "--frozen-dir", f"{self.ray_source}/configs/frozen/{self.config['dataset_id']}",
                "--capacity-summary", f"{self.ray_stage}/capacity_summary.json", timeout=7200,
            )
            assert self.pod
            self.kubectl("exec", self.pod, "--", "mkdir", "-p", str(self.pod_dataset_root))
            self.shell(
                f"kubectl -n {shlex.quote(self.ray_namespace)} exec {shlex.quote(self.ray_pod)} -- "
                f"tar -C {shlex.quote(self.ray_stage)}/dataset -cf - . | "
                f"kubectl -n {shlex.quote(self.namespace)} exec -i {shlex.quote(self.pod)} -- "
                f"tar -C {shlex.quote(str(self.pod_dataset_root))} -xf -",
                timeout=7200,
            )
            progress["status"] = "DATASET_SEALED"
            self.update_progress(progress)
            return 0
        finally:
            if progress.get("status") in {"DATASET_SEALED", "STOPPED_FAIL_CLOSED"}:
                self.restore_production()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default="configs/servingrom_dataset_v1.yaml")
    parser.add_argument("--max-runs", type=int, default=0, help="development/resume guard; 0 runs the full campaign")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(Campaign(parse_args()).execute())
