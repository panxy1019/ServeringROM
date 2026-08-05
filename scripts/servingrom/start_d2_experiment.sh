#!/usr/bin/env bash
set -euo pipefail

NS=${NS:-infra-learning}
PROD=ray-vllm-pd-worker-qwen36-27b
EXP=ray-vllm-pd-servingrom-qwen36-27b
CONFIG_ID=qwen36-1p2d-d2-full-decode-only-async-v1

[[ $(kubectl -n "$NS" get deployment "$PROD" -o jsonpath='{.spec.replicas}') == 0 ]] || {
  echo "Refusing start: production still owns physical NPU 10-15." >&2
  exit 2
}
[[ $(kubectl -n "$NS" get deployment "$EXP" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="SERVINGROM_CONFIG_ID")].value}') == "$CONFIG_ID" ]] || {
  echo "Refusing start: experiment config ID mismatch." >&2
  exit 3
}
kubectl -n "$NS" scale deployment/"$EXP" --replicas=1
kubectl -n "$NS" rollout status deployment/"$EXP" --timeout=60m

