#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
CURRENT_DEPLOYMENT=${CURRENT_DEPLOYMENT:-ray-vllm-pd-control-v1-qwen36-27b}
PILOT_DEPLOYMENT=${PILOT_DEPLOYMENT:-ray-vllm-pd-control-pilot-qwen36-27b}
KUBECONFIG=${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}
export KUBECONFIG

kubectl -n "$NAMESPACE" scale deployment/"$PILOT_DEPLOYMENT" --replicas=0
kubectl -n "$NAMESPACE" wait --for=delete pod -l app="$PILOT_DEPLOYMENT" --timeout=20m || true
kubectl -n "$NAMESPACE" scale deployment/"$CURRENT_DEPLOYMENT" --replicas=1
kubectl -n "$NAMESPACE" rollout status deployment/"$CURRENT_DEPLOYMENT" --timeout=90m
