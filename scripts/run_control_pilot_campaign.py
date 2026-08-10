#!/usr/bin/env python3
"""Fail-closed 12-run Round 14.1 campaign on one warm Control-v1 Pod."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Pilot:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.project_root.resolve()
        self.config = json.loads((self.root / args.config).read_text())
        self.namespace = self.config["namespace"]
        self.deployment = self.config["deployment"]
        self.ray_namespace = self.config["ray_namespace"]
        self.ray_pod = self.config["ray_head_pod"]
        self.experiment_id = self.config["pilot_id"]
        self.result_root = Path(self.config["results_root"])
        self.pod_result_root = Path("/servingrom-results") / self.experiment_id
        self.state_dir = self.root / ".campaign" / self.experiment_id
        self.manifest_path = self.state_dir / "pilot_manifest.json"
        self.ray_source = f"/tmp/{self.experiment_id}-source"
        self.ray_stage = f"/tmp/{self.experiment_id}-stage"
        self.kubectl_command = shlex.split(os.getenv("KUBECTL", "kubectl"))
        self.pod = ""
        self.pod_uid = ""

    def run(self, command: list[str], *, timeout=None, check=True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(command, text=True, capture_output=True, check=check, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            print(f"COMMAND FAILED: {shlex.join(command)}", flush=True)
            print(exc.stdout or "", flush=True)
            print(exc.stderr or "", flush=True)
            raise

    def kubectl(self, *parts: str, namespace: str | None = None, **kwargs):
        selected = self.namespace if namespace is None else namespace
        return self.run([*self.kubectl_command, "-n", selected, *parts], **kwargs)

    def ray(self, *parts: str, **kwargs):
        return self.kubectl("exec", self.ray_pod, "--", *parts, namespace=self.ray_namespace, **kwargs)

    def shell(self, command: str, **kwargs):
        return self.run(["/bin/bash", "-lc", command], **kwargs)

    def plan(self) -> list[dict[str, Any]]:
        rows = []
        seed = 140100
        for workload in ("balanced", "mixed-bimodal"):
            capacity = float(self.config["capacity"][workload])
            for fraction in self.config["load_fractions"]:
                for excitation in self.config["excitations"]:
                    rows.append({
                        "plan_id": f"{workload}-l{int(fraction*100)}-{excitation}",
                        "workload": workload, "load_fraction": fraction,
                        "arrival_rate": capacity * fraction, "capacity": capacity,
                        "excitation": excitation, "seed": seed,
                        "status": "PENDING", "run_id": None, "error": None,
                    })
                    seed += 1
        if len(rows) != 12:
            raise AssertionError("pilot must contain 12 runs")
        return rows

    def initialize(self) -> dict[str, Any]:
        hashes = {
            path: sha256(self.root / path) for path in (
                "scripts/control_pilot_workload.py", "scripts/run_control_pilot_campaign.py",
                "scripts/run_control_pilot_pipeline.sh", "scripts/build_control_snapshots.py",
                "scripts/validate_control_pilot_run.py", "servingrom_pipeline/control_snapshot.py",
                "configs/servingrom_control_pilot_v1.json",
                "configs/workloads/balanced.yaml", "configs/workloads/mixed-bimodal.yaml",
            )
        }
        if self.manifest_path.exists():
            value = json.loads(self.manifest_path.read_text())
            if value["source_hashes"] != hashes:
                raise RuntimeError("pilot sources changed after manifest creation")
            return value
        value = {
            "schema_version": "servingrom.control_pilot_manifest.v1",
            "pilot_id": self.experiment_id, "status": "INITIALIZING",
            "created_at": stamp(), "updated_at": stamp(), "planned_runs": 12,
            "source_hashes": hashes, "runs": self.plan(),
        }
        atomic_json(self.manifest_path, value)
        return value

    def save(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = stamp()
        manifest["counts"] = {
            name: sum(row["status"] == name for row in manifest["runs"])
            for name in ("PENDING", "RUNNING", "SEALED", "FAILED")
        }
        atomic_json(self.manifest_path, manifest)
        self.kubectl("exec", self.pod, "--", "mkdir", "-p", str(self.pod_result_root))
        self.kubectl("cp", str(self.manifest_path), f"{self.pod}:{self.pod_result_root}/pilot_manifest.json")

    def verify_pod(self) -> None:
        self.pod = self.kubectl(
            "get", "pod", "-l", f"app={self.deployment}",
            "-o", "jsonpath={.items[0].metadata.name}",
        ).stdout.strip()
        ready = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].ready}").stdout.strip()
        restart = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].restartCount}").stdout.strip()
        self.pod_uid = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.metadata.uid}").stdout.strip()
        if ready != "true" or restart != "0":
            raise RuntimeError(f"pilot Pod is not frozen healthy: ready={ready} restart={restart}")
        state = json.loads(self.kubectl("exec", self.pod, "--", "curl", "--noproxy", "*", "-fsS", "http://127.0.0.1:8080/servingrom/control/state").stdout)
        if not state.get("enabled") or not state.get("decoders_healthy"):
            raise RuntimeError(f"control plane unavailable: {state}")
        acks = json.loads(self.kubectl(
            "exec", self.pod, "--", "python3", "/opt/qwen36-pd/servingrom_run_control.py",
            "status", "--timeout", "30",
        ).stdout)
        components = {row["component"] for row in acks.get("acks", [])}
        required = {"proxy", "prefill", "decode-0", "decode-1", "mooncake", "device"}
        if not required <= components:
            raise RuntimeError(f"run-control capability missing: required={required} actual={components}")

    def sync_ray(self) -> None:
        selected = "scripts servingrom_pipeline servingrom_telemetry configs"
        kubectl = shlex.join(self.kubectl_command)
        command = (
            f"{kubectl} -n {self.ray_namespace} exec {self.ray_pod} -- mkdir -p {self.ray_source} {self.ray_stage}/runs && "
            f"tar -C {self.root} -cf - {selected} | {kubectl} -n {self.ray_namespace} exec -i {self.ray_pod} -- tar -C {self.ray_source} -xf -"
        )
        self.shell(command, timeout=600)

    def run_control(self, action: str, run_id: str) -> dict[str, Any]:
        value = json.loads(self.kubectl(
            "exec", self.pod, "--", "python3", "/opt/qwen36-pd/servingrom_run_control.py",
            action, "--run-id", run_id, "--experiment-id", self.experiment_id,
            "--config-id", self.config["config_id"], "--timeout", "120", timeout=180,
        ).stdout)
        print(f"RUN CONTROL {action}: {run_id} acks={value.get('acknowledged')}", flush=True)
        return value

    def ensure_baseline(self, run_id: str) -> None:
        value = json.loads(self.kubectl(
            "exec", self.pod, "--", "python3",
            "/opt/qwen36-pd/ensure_control_baseline.py",
            "--label", run_id, timeout=60,
        ).stdout)
        if not value.get("accepted"):
            raise RuntimeError(f"failed to restore control baseline: {value}")
        print(
            f"CONTROL BASELINE: {run_id} already={value.get('already_baseline', False)}",
            flush=True,
        )

    def metadata(self, row: dict[str, Any], run_id: str, workload: dict[str, Any]) -> None:
        deployment = self.kubectl("get", "deployment", self.deployment, "-o", "yaml").stdout
        pod_yaml = self.kubectl("get", "pod", self.pod, "-o", "yaml").stdout
        values = {
            "run.json": {
                **{key: row[key] for key in ("plan_id", "workload", "load_fraction", "arrival_rate", "capacity", "excitation", "seed")},
                "run_id": run_id, "pilot_id": self.experiment_id, "config_id": self.config["config_id"],
                "pod": self.pod, "pod_uid": self.pod_uid, "graph_mode": "FULL_DECODE_ONLY",
                "async_scheduling": True, "prefill_tp": 2, "decode_tp": 2,
            },
            "workload.json": {
                **json.loads((self.root / "configs" / "workloads" / f"{row['workload']}.yaml").read_text()),
                "target_arrival_rate": row["arrival_rate"], "load_fraction": row["load_fraction"],
                "excitation": row["excitation"], "seed": row["seed"],
            },
            "workload_result.json": workload,
            "measurement.json": {
                "measurement_start_wall_ns": workload["measurement_start_wall_ns"],
                "measurement_end_wall_ns": workload["measurement_end_wall_ns"],
                "snapshot_period_ms": 200,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            for name, value in values.items(): atomic_json(temp / name, value)
            (temp / "deployment.yaml").write_text(deployment)
            (temp / "pod.yaml").write_text(pod_yaml)
            target = f"/servingrom-results/{self.experiment_id}/{run_id}/metadata"
            self.kubectl("exec", self.pod, "--", "mkdir", "-p", target)
            for path in temp.iterdir(): self.kubectl("cp", str(path), f"{self.pod}:{target}/{path.name}")

    def process(self, run_id: str) -> dict[str, Any]:
        source = f"/servingrom-results/{self.experiment_id}/{run_id}"
        target = f"{self.ray_stage}/runs/{run_id}"
        kubectl = shlex.join(self.kubectl_command)
        self.shell(
            f"{kubectl} -n {self.ray_namespace} exec {self.ray_pod} -- mkdir -p {target} && "
            f"{kubectl} -n {self.namespace} exec {self.pod} -- tar -C {source} -cf - . | "
            f"{kubectl} -n {self.ray_namespace} exec -i {self.ray_pod} -- tar -C {target} -xf -",
            timeout=3600,
        )
        self.ray("bash", f"{self.ray_source}/scripts/run_control_pilot_pipeline.sh", target, timeout=7200)
        quality = json.loads(self.ray("cat", f"{target}/reports/control_pilot_quality.json").stdout)
        if not quality["valid"]:
            raise RuntimeError(f"control quality failed: {quality}")
        self.shell(
            f"{kubectl} -n {self.ray_namespace} exec {self.ray_pod} -- tar -C {target} -cf - derived reports metadata | "
            f"{kubectl} -n {self.namespace} exec -i {self.pod} -- tar -C {source} -xf -",
            timeout=3600,
        )
        return quality

    def execute_one(self, row: dict[str, Any], manifest: dict[str, Any]) -> None:
        run_id = f"sr-control-pilot-{row['plan_id']}-{stamp()}"
        row.update({"run_id": run_id, "status": "RUNNING", "started_at": stamp()})
        self.save(manifest)
        self.run_control("activate", run_id)
        output = f"/tmp/{run_id}-workload.json"
        try:
            self.ray(
                "python3", f"{self.ray_source}/scripts/control_pilot_workload.py",
                "--workload-config", f"{self.ray_source}/configs/workloads/{row['workload']}.yaml",
                "--endpoint", f"http://{self.config['service']}:8080",
                "--tokenize-endpoint", f"http://{self.config['service']}:13700",
                "--run-id", run_id, "--output", output, "--seed", str(row["seed"]),
                "--arrival-rate", str(row["arrival_rate"]), "--load-fraction", str(row["load_fraction"]),
                "--excitation", row["excitation"], "--warmup-seconds", str(self.config["warmup_seconds"]),
                "--measurement-seconds", str(self.config["measurement_seconds"]),
                "--drain-timeout-seconds", str(self.config["drain_timeout_seconds"]),
                timeout=2400,
            )
            workload = json.loads(self.ray("cat", output).stdout)
            if not workload["drain"]["drained"]:
                raise RuntimeError("workload did not drain")
            self.metadata(row, run_id, workload)
        finally:
            rollback_error = None
            try:
                self.ensure_baseline(run_id)
            except Exception as exc:
                rollback_error = exc
            self.run_control("deactivate", run_id)
            if rollback_error is not None:
                raise rollback_error
        quality = self.process(run_id)
        current_uid = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.metadata.uid}").stdout.strip()
        restart = self.kubectl("get", "pod", self.pod, "-o", "jsonpath={.status.containerStatuses[0].restartCount}").stdout.strip()
        if current_uid != self.pod_uid or restart != "0":
            raise RuntimeError(f"frozen Pod changed: uid={current_uid} restart={restart}")
        row.update({"status": "SEALED", "finished_at": stamp(), "quality": quality})
        print(f"SEALED {row['plan_id']} {run_id}", flush=True)

    def execute(self) -> int:
        manifest = self.initialize()
        self.verify_pod()
        self.sync_ray()
        manifest.update({"status": "RUNNING", "pod": self.pod, "pod_uid": self.pod_uid})
        self.save(manifest)
        for row in manifest["runs"]:
            if row["status"] == "SEALED": continue
            try:
                self.execute_one(row, manifest)
                self.save(manifest)
            except Exception as exc:
                row.update({"status": "FAILED", "finished_at": stamp(), "error": repr(exc)})
                manifest["status"] = "STOPPED_FAIL_CLOSED"
                self.save(manifest)
                raise
        manifest["status"] = "PILOT_COLLECTION_COMPLETE"
        self.save(manifest)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="configs/servingrom_control_pilot_v1.json")
    return Pilot(parser.parse_args()).execute()


if __name__ == "__main__":
    raise SystemExit(main())
