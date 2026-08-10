#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
SOURCE_DEPLOYMENT=${SOURCE_DEPLOYMENT:-ray-vllm-pd-decode-ab-qwen36-27b}
CONTROL_DEPLOYMENT=${CONTROL_DEPLOYMENT:-ray-vllm-pd-control-v1-qwen36-27b}
KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
export KUBECONFIG

kubectl -n "$NAMESPACE" scale deployment/"$CONTROL_DEPLOYMENT" --replicas=0
kubectl -n "$NAMESPACE" rollout status deployment/"$CONTROL_DEPLOYMENT" --timeout=120s || true
kubectl -n "$NAMESPACE" scale deployment/"$SOURCE_DEPLOYMENT" --replicas=1
kubectl -n "$NAMESPACE" rollout status deployment/"$SOURCE_DEPLOYMENT" --timeout=60m
kubectl -n "$NAMESPACE" get deployment "$SOURCE_DEPLOYMENT" "$CONTROL_DEPLOYMENT"
