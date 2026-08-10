#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
CURRENT_DEPLOYMENT=${CURRENT_DEPLOYMENT:-ray-vllm-pd-control-v1-qwen36-27b}
PILOT_DEPLOYMENT=${PILOT_DEPLOYMENT:-ray-vllm-pd-control-pilot-qwen36-27b}
KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export KUBECONFIG

telemetry_files=()
for path in "$REPO_ROOT"/servingrom_telemetry/*.py; do
  telemetry_files+=("--from-file=$path")
done
kubectl -n "$NAMESPACE" create configmap servingrom-telemetry-control-pilot-v1 \
  "${telemetry_files[@]}" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" create configmap servingrom-entrypoint-control-pilot-v1 \
  --from-file=pd-worker-entrypoint-instrumented.sh="$REPO_ROOT/scripts/servingrom/pd-worker-entrypoint-instrumented.sh" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" create configmap qwen36-pd-control-pilot-scripts \
  --from-file=discover_npu_mapping.py="$REPO_ROOT/scripts/discover_npu_mapping.py" \
  --from-file=servingrom_run_control.py="$REPO_ROOT/scripts/servingrom_run_control.py" \
  --from-file=ensure_control_baseline.py="$REPO_ROOT/scripts/servingrom/ensure_control_baseline.py" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" create configmap servingrom-control-pilot-v1-code \
  --from-file=pd_proxy.py="$REPO_ROOT/scripts/pd_proxy.py" \
  --from-file=package_init.py="$REPO_ROOT/servingrom_control/__init__.py" \
  --from-file=manager.py="$REPO_ROOT/servingrom_control/manager.py" \
  --from-file=schema.py="$REPO_ROOT/servingrom_control/schema.py" \
  --from-file=safety.py="$REPO_ROOT/servingrom_control/safety.py" \
  --from-file=state.py="$REPO_ROOT/servingrom_control/state.py" \
  --from-file=telemetry.py="$REPO_ROOT/servingrom_control/telemetry.py" \
  --from-file=actuators_init.py="$REPO_ROOT/servingrom_control/actuators/__init__.py" \
  --from-file=routing_ratio.py="$REPO_ROOT/servingrom_control/actuators/routing_ratio.py" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "$REPO_ROOT/k8s/servingrom/qwen36-control-pilot-v1.yaml"
kubectl -n "$NAMESPACE" scale deployment/"$PILOT_DEPLOYMENT" --replicas=0
kubectl -n "$NAMESPACE" scale deployment/"$CURRENT_DEPLOYMENT" --replicas=0
kubectl -n "$NAMESPACE" wait --for=delete pod -l app="$CURRENT_DEPLOYMENT" --timeout=20m || true
kubectl -n "$NAMESPACE" scale deployment/"$PILOT_DEPLOYMENT" --replicas=1
kubectl -n "$NAMESPACE" rollout status deployment/"$PILOT_DEPLOYMENT" --timeout=90m
