#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/common.sh"

require_deployment_identity

conflicts=(
  ray-vllm-pd-control-v1-qwen36-27b
  ray-vllm-pd-decode-ab-qwen36-27b
  ray-vllm-pd-servingrom-qwen36-27b
  ray-vllm-pd-worker-qwen36-27b
  ray-vllm-pd-worker-qwen36-27b-1p1d
)
for name in "${conflicts[@]}"; do
  replicas=$(kube -n "$NAMESPACE" get deployment "$name" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 0)
  [[ "${replicas:-0}" == 0 ]] || {
    echo "Refusing start: conflicting deployment $name has replicas=$replicas" >&2
    exit 5
  }
done

kube -n "$NAMESPACE" scale deployment/"$DEPLOYMENT" --replicas=1
kube -n "$NAMESPACE" rollout status deployment/"$DEPLOYMENT" --timeout=90m
"$ROOT/status.sh"
