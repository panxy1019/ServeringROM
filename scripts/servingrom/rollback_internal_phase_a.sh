#!/usr/bin/env bash
set -euo pipefail
NAMESPACE=${NAMESPACE:-infra-learning}
kubectl -n "$NAMESPACE" scale deployment/ray-vllm-pd-servingrom-qwen36-27b --replicas=0
kubectl -n "$NAMESPACE" rollout status deployment/ray-vllm-pd-servingrom-qwen36-27b --timeout=10m
kubectl -n "$NAMESPACE" scale deployment/ray-vllm-pd-decode-ab-qwen36-27b --replicas=1
kubectl -n "$NAMESPACE" rollout status deployment/ray-vllm-pd-decode-ab-qwen36-27b --timeout=90m
