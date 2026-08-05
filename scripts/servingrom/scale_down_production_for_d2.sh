#!/usr/bin/env bash
set -euo pipefail

NS=${NS:-infra-learning}
PROD=ray-vllm-pd-worker-qwen36-27b
EXP=ray-vllm-pd-servingrom-qwen36-27b
STATE_DIR=${STATE_DIR:-results/phase0b-switch-state}
EXPECTED_CONFIRMATION="scale-down-${PROD}"

if [[ ${CONFIRM_PRODUCTION_INTERRUPTION:-} != "$EXPECTED_CONFIRMATION" ]]; then
  echo "Refusing production interruption." >&2
  echo "Set CONFIRM_PRODUCTION_INTERRUPTION=$EXPECTED_CONFIRMATION after an approved maintenance window." >&2
  exit 2
fi

exp_replicas=$(kubectl -n "$NS" get deployment "$EXP" -o jsonpath='{.spec.replicas}')
[[ "$exp_replicas" == 0 ]] || { echo "$EXP must be scaled to 0 first" >&2; exit 3; }

mkdir -p "$STATE_DIR"
kubectl -n "$NS" get deployment "$PROD" -o yaml >"$STATE_DIR/production-before-scale-down.yaml"
kubectl -n "$NS" get deployment "$PROD" -o jsonpath='{.spec.replicas}' >"$STATE_DIR/production-replicas.txt"
kubectl -n "$NS" scale deployment/"$PROD" --replicas=0
kubectl -n "$NS" wait --for=delete pod -l app="$PROD" --timeout=900s

