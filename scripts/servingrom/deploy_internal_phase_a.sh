#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
DEPLOYMENT=${DEPLOYMENT:-ray-vllm-pd-servingrom-qwen36-27b}
EXPERIMENT_ID=${EXPERIMENT_ID:-servingrom-internal-phase-a}
RUN_ID=${RUN_ID:-phase-a-$(date -u +%Y%m%dT%H%M%SZ)}
PROJECT_DIR=${PROJECT_DIR:-/home/admin/Desktop/sql/qwen36_pd_1p2d}

kubectl -n "$NAMESPACE" create configmap qwen36-pd-servingrom-d2-scripts \
  --from-file=pd_proxy.py="$PROJECT_DIR/scripts/pd_proxy.py" \
  --from-file=discover_npu_mapping.py="$PROJECT_DIR/scripts/discover_npu_mapping.py" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$PROJECT_DIR/k8s/servingrom/qwen36-servingrom-d2.yaml"
kubectl -n "$NAMESPACE" set env deployment/"$DEPLOYMENT" \
  SERVINGROM_EXPERIMENT_ID="$EXPERIMENT_ID" SERVINGROM_RUN_ID="$RUN_ID"
kubectl -n "$NAMESPACE" scale deployment/ray-vllm-pd-decode-ab-qwen36-27b --replicas=0
kubectl -n "$NAMESPACE" rollout status deployment/ray-vllm-pd-decode-ab-qwen36-27b --timeout=10m
kubectl -n "$NAMESPACE" scale deployment/"$DEPLOYMENT" --replicas=1
kubectl -n "$NAMESPACE" rollout status deployment/"$DEPLOYMENT" --timeout=90m
printf '%s\n' "$EXPERIMENT_ID/$RUN_ID"
