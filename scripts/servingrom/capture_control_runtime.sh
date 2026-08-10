#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
DEPLOYMENT=${DEPLOYMENT:-ray-vllm-pd-control-v1-qwen36-27b}
OUTPUT_DIR=${OUTPUT_DIR:-results/control-runtime-smoke/metadata}
KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
export KUBECONFIG

mkdir -p "$OUTPUT_DIR"
pod=$(kubectl -n "$NAMESPACE" get pod -l app="$DEPLOYMENT" -o jsonpath='{.items[0].metadata.name}')
kubectl -n "$NAMESPACE" get pod "$pod" -o json >"$OUTPUT_DIR/pod-ready.json"
kubectl -n "$NAMESPACE" get pod "$pod" -o yaml >"$OUTPUT_DIR/pod-ready.yaml"
kubectl -n "$NAMESPACE" get deploy "$DEPLOYMENT" -o yaml >"$OUTPUT_DIR/deployment.yaml"
kubectl -n "$NAMESPACE" exec -i "$pod" -- /bin/bash -s >"$OUTPUT_DIR/runtime-identities.txt" <<'REMOTE'
set -euo pipefail
cat /var/run/qwen36-pd/effective-config.txt
echo '=== devices'
cat /var/run/qwen36-pd/service-device-map.txt
echo '=== mapping'
cat /var/run/qwen36-pd/npu-mapping.json
echo '=== identities'
for name in prefill decode-a decode-b proxy; do
  pid=$(<"/var/run/qwen36-pd/$name.pid")
  start_ticks=$(awk '{print $22}' "/proc/$pid/stat")
  printf '%s pid=%s start_ticks=%s command=' "$name" "$pid" "$start_ticks"
  tr '\0' ' ' <"/proc/$pid/cmdline"
  echo
done
REMOTE
kubectl -n "$NAMESPACE" exec "$pod" -- \
  curl --noproxy '*' -fsS http://127.0.0.1:8080/servingrom/control/state \
  >"$OUTPUT_DIR/initial-control-state.json"
kubectl -n "$NAMESPACE" get pod "$pod" \
  -o jsonpath='{.metadata.uid}{" restart="}{.status.containerStatuses[0].restartCount}{" ready="}{.status.containerStatuses[0].ready}{"\n"}' \
  | tee "$OUTPUT_DIR/pod-identity.txt"
