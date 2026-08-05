#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NS=${NS:-infra-learning}
DEPLOY=${DEPLOY:-ray-vllm-pd-decode-ab-qwen36-27b}
HOST_RESULTS_ROOT=${HOST_RESULTS_ROOT:-/home/admin/testpanxy/infralearning/qwen36_pd_1p2d/results}
CONFIGMAP=${CONFIGMAP:-servingrom-telemetry-python}

kubectl -n "$NS" create configmap "$CONFIGMAP" \
  --from-file="$ROOT/servingrom_telemetry" \
  --dry-run=client -o yaml | kubectl apply -f -

# This merge patch only adds a Python package mount and an experiment-results
# mount. Engine arguments, NPU bindings, admission, and routing configuration
# remain owned by the frozen D2 Deployment.
PATCH=$(python3 - "$HOST_RESULTS_ROOT" "$CONFIGMAP" <<'PY'
import json, sys
root, configmap = sys.argv[1:]
print(json.dumps({
  "spec": {"template": {"spec": {
    "containers": [{
      "name": "pd-worker",
      "env": [
        {"name": "PYTHONPATH", "value": "/opt/qwen36-pd"},
      ],
      "volumeMounts": [
        {"name": "servingrom-telemetry", "mountPath": "/opt/qwen36-pd/servingrom_telemetry", "readOnly": True},
        {"name": "servingrom-results", "mountPath": "/servingrom-results"},
      ],
    }],
    "volumes": [
      {"name": "servingrom-telemetry", "configMap": {"name": configmap}},
      {"name": "servingrom-results", "hostPath": {"path": root, "type": "DirectoryOrCreate"}},
    ],
  }}}}
))
PY
)
kubectl -n "$NS" patch deployment "$DEPLOY" --type=strategic --patch "$PATCH"

echo "Installed side-band telemetry package and results volume on $NS/$DEPLOY."
echo "Telemetry remains disabled until SERVINGROM_TELEMETRY_ENABLED=true is set."
