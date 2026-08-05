#!/usr/bin/env bash
set -euo pipefail

NS=${NS:-infra-learning}
PROD=ray-vllm-pd-worker-qwen36-27b
EXP=ray-vllm-pd-servingrom-qwen36-27b

kubectl -n "$NS" scale deployment/"$EXP" --replicas=0
kubectl -n "$NS" wait --for=delete pod -l app="$EXP" --timeout=900s || true
kubectl -n "$NS" scale deployment/"$PROD" --replicas=1
kubectl -n "$NS" rollout status deployment/"$PROD" --timeout=60m

