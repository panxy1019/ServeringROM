#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
DEPLOYMENT=${DEPLOYMENT:-ray-vllm-pd-control-v1-qwen36-27b}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/results/control-runtime-smoke}
KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
export KUBECONFIG

mkdir -p "$OUTPUT_DIR/metadata-post" "$OUTPUT_DIR/runtime-smoke-artifacts"
pod=$(kubectl -n "$NAMESPACE" get pod -l app="$DEPLOYMENT" -o jsonpath='{.items[0].metadata.name}')
OUTPUT_DIR="$OUTPUT_DIR/metadata-post" \
  bash "$REPO_ROOT/scripts/servingrom/capture_control_runtime.sh"
rm -rf "$OUTPUT_DIR/runtime-smoke-artifacts/control-v1-runtime-smoke"
kubectl -n "$NAMESPACE" cp \
  "$pod:/servingrom-results/servingrom-control-v1-smoke/control-v1-runtime-smoke" \
  "$OUTPUT_DIR/runtime-smoke-artifacts/control-v1-runtime-smoke"
kubectl -n "$NAMESPACE" exec -i "$pod" -- /bin/bash -s >"$OUTPUT_DIR/runtime-error-scan.txt" <<'REMOTE'
grep -Ehi 'OOM|out of memory|engine.*(dead|death)|Mooncake.*(fatal|error)|Traceback' \
  /var/log/qwen36-pd/*.log || true
REMOTE
kubectl -n "$NAMESPACE" exec -i "$pod" -- /bin/bash -s >"$OUTPUT_DIR/d2-config-log-evidence.txt" <<'REMOTE'
grep -Eh 'FULL_DECODE_ONLY|async.scheduling|async_scheduling|Starting vLLM server' \
  /var/log/qwen36-pd/decode-*.log | tail -n 100
REMOTE
kubectl -n "$NAMESPACE" exec -i "$pod" -- /bin/bash -s >"$OUTPUT_DIR/engine-lifecycle-counts.txt" <<'REMOTE'
for name in prefill decode-a decode-b; do
  file="/var/log/qwen36-pd/$name.log"
  printf '%s model_load_starts=' "$name"
  grep -c 'Starting to load model' "$file" || true
  printf '%s engine_init_starts=' "$name"
  grep -c 'Initializing a V1 LLM engine' "$file" || true
  printf '%s api_server_starts=' "$name"
  grep -c 'Starting vLLM server' "$file" || true
done
REMOTE
kubectl -n "$NAMESPACE" exec -i "$pod" -- /bin/bash -s >"$OUTPUT_DIR/runtime-fatal-scan.txt" <<'REMOTE'
grep -Ehi '(^|[[:space:]])(ERROR|CRITICAL)([[:space:]]|$)|out of memory|engine (died|death)|Mooncake.*fatal' \
  /var/log/qwen36-pd/*.log || true
REMOTE
kubectl -n "$NAMESPACE" get pod "$pod" -o json >"$OUTPUT_DIR/pod-after.json"
