#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/common.sh"

required_configmaps=(
  servingrom-telemetry-control-pilot-v1
  servingrom-entrypoint-control-pilot-v1
  qwen36-pd-control-pilot-scripts
  servingrom-control-pilot-v1-code
)
for name in "${required_configmaps[@]}"; do
  kube -n "$NAMESPACE" get configmap "$name" >/dev/null || {
    echo "Missing ConfigMap: $name. Refusing partial deployment." >&2
    exit 6
  }
done

kube apply -f "$ROOT/qwen36-control-pilot-v1.yaml"
kube -n "$NAMESPACE" scale deployment/"$DEPLOYMENT" --replicas=0
"$ROOT/start.sh"
