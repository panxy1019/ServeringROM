#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
SOURCE_DEPLOYMENT=${SOURCE_DEPLOYMENT:-ray-vllm-pd-decode-ab-qwen36-27b}
CONTROL_DEPLOYMENT=${CONTROL_DEPLOYMENT:-ray-vllm-pd-control-v1-qwen36-27b}
CONTROL_SERVICE=${CONTROL_SERVICE:-qwen36-pd-control-v1}
CONTROL_CONFIGMAP=${CONTROL_CONFIGMAP:-servingrom-control-v1-code}
KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
export KUBECONFIG
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/results/control-plane-deployment}

mkdir -p "$OUTPUT_DIR"
kubectl -n "$NAMESPACE" get deployment "$SOURCE_DEPLOYMENT" -o json >"$OUTPUT_DIR/source-deployment.json"
kubectl -n "$NAMESPACE" get deployment "$SOURCE_DEPLOYMENT" -o yaml >"$OUTPUT_DIR/source-deployment.yaml"

kubectl -n "$NAMESPACE" create configmap "$CONTROL_CONFIGMAP" \
  --from-file=pd_proxy.py="$REPO_ROOT/scripts/pd_proxy.py" \
  --from-file=entrypoint.sh="$REPO_ROOT/scripts/servingrom/pd-worker-entrypoint-control-v1.sh" \
  --from-file=package_init.py="$REPO_ROOT/servingrom_control/__init__.py" \
  --from-file=manager.py="$REPO_ROOT/servingrom_control/manager.py" \
  --from-file=schema.py="$REPO_ROOT/servingrom_control/schema.py" \
  --from-file=safety.py="$REPO_ROOT/servingrom_control/safety.py" \
  --from-file=state.py="$REPO_ROOT/servingrom_control/state.py" \
  --from-file=telemetry.py="$REPO_ROOT/servingrom_control/telemetry.py" \
  --from-file=actuators_init.py="$REPO_ROOT/servingrom_control/actuators/__init__.py" \
  --from-file=routing_ratio.py="$REPO_ROOT/servingrom_control/actuators/routing_ratio.py" \
  --dry-run=client -o yaml | kubectl apply -f -

python3 - "$OUTPUT_DIR/source-deployment.json" "$OUTPUT_DIR/control-deployment.json" \
  "$CONTROL_DEPLOYMENT" "$CONTROL_CONFIGMAP" <<'PY'
import json, sys
source_path, output_path, name, config_map = sys.argv[1:]
value = json.load(open(source_path, encoding="utf-8"))
for key in ("status",):
    value.pop(key, None)
metadata = value["metadata"]
for key in ("uid", "resourceVersion", "generation", "creationTimestamp", "managedFields"):
    metadata.pop(key, None)
metadata["name"] = name
metadata.pop("annotations", None)
metadata["labels"] = dict(metadata.get("labels") or {})
metadata["labels"]["app"] = name
metadata["labels"]["servingrom.openai/control-plane"] = "control-v1"
spec = value["spec"]
spec["replicas"] = 0
spec["selector"]["matchLabels"] = {"app": name}
template = spec["template"]
template["metadata"]["labels"] = dict(template["metadata"].get("labels") or {})
template["metadata"]["labels"]["app"] = name
template["metadata"]["labels"]["servingrom.openai/control-plane"] = "control-v1"
container = template["spec"]["containers"][0]
container["args"] = ["exec /opt/servingrom-control/pd-worker-entrypoint-control-v1.sh"]
env = {row["name"]: row for row in container.get("env", [])}
updates = {
    "PYTHONPATH": "/opt/servingrom-control:/opt/qwen36-pd",
    "SERVINGROM_CONTROL_ENABLED": "true",
    "SERVINGROM_CONTROL_TEST_ENDPOINTS": "true",
    "SERVINGROM_EXPERIMENT_ID": "servingrom-control-v1-smoke",
    "SERVINGROM_RUN_ID": "control-v1-runtime-smoke",
    "SERVINGROM_CONFIG_ID": "qwen36-1p2d-d2-full-decode-only-async-control-v1",
    "SERVINGROM_COMPONENT": "proxy-control-v1",
    "SERVINGROM_OUTPUT_DIR": "/servingrom-results/servingrom-control-v1-smoke/control-v1-runtime-smoke/raw/proxy",
}
for key, val in updates.items():
    env[key] = {"name": key, "value": val}
container["env"] = list(env.values())
container.setdefault("volumeMounts", []).append({
    "name": "servingrom-control-v1-code",
    "mountPath": "/opt/servingrom-control",
    "readOnly": True,
})
template["spec"].setdefault("volumes", []).append({
    "name": "servingrom-control-v1-code",
    "configMap": {
        "name": config_map,
        "defaultMode": 365,
        "items": [
            {"key": "pd_proxy.py", "path": "pd_proxy.py"},
            {"key": "entrypoint.sh", "path": "pd-worker-entrypoint-control-v1.sh"},
            {"key": "package_init.py", "path": "servingrom_control/__init__.py"},
            {"key": "manager.py", "path": "servingrom_control/manager.py"},
            {"key": "schema.py", "path": "servingrom_control/schema.py"},
            {"key": "safety.py", "path": "servingrom_control/safety.py"},
            {"key": "state.py", "path": "servingrom_control/state.py"},
            {"key": "telemetry.py", "path": "servingrom_control/telemetry.py"},
            {"key": "actuators_init.py", "path": "servingrom_control/actuators/__init__.py"},
            {"key": "routing_ratio.py", "path": "servingrom_control/actuators/routing_ratio.py"},
        ],
    },
})
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
PY

kubectl apply -f "$OUTPUT_DIR/control-deployment.json"
python3 - "$OUTPUT_DIR/source-deployment.json" "$OUTPUT_DIR/control-service.json" \
  "$CONTROL_SERVICE" "$CONTROL_DEPLOYMENT" "$NAMESPACE" <<'PY'
import json, sys
_, output_path, name, deployment, namespace = sys.argv[1:]
value = {
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {"name": name, "namespace": namespace},
    "spec": {
        "selector": {"app": deployment},
        "ports": [
            {"name": "openai", "port": 8080, "targetPort": 8080},
            {"name": "prefill", "port": 13700, "targetPort": 13700},
            {"name": "decode-a", "port": 13701, "targetPort": 13701},
            {"name": "decode-b", "port": 13702, "targetPort": 13702},
        ],
    },
}
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2)
    stream.write("\n")
PY
kubectl apply -f "$OUTPUT_DIR/control-service.json"
kubectl -n "$NAMESPACE" get deployment "$CONTROL_DEPLOYMENT" -o yaml >"$OUTPUT_DIR/control-deployment.yaml"
kubectl -n "$NAMESPACE" get configmap "$CONTROL_CONFIGMAP" -o yaml >"$OUTPUT_DIR/control-configmap.yaml"

echo "Prepared $CONTROL_DEPLOYMENT with replicas=0."
echo "Activate only after resource review:"
echo "  kubectl -n $NAMESPACE scale deployment/$SOURCE_DEPLOYMENT --replicas=0"
echo "  kubectl -n $NAMESPACE scale deployment/$CONTROL_DEPLOYMENT --replicas=1"
