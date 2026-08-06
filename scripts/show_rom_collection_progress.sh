#!/usr/bin/env bash
set -euo pipefail
DATASET_ID=${1:-servingrom-qwen36-1p2d-d2-rom-v1}
POD=$(kubectl -n infra-learning get pod -l app=ray-vllm-pd-servingrom-qwen36-27b -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "$POD" ]]; then
  kubectl -n infra-learning exec "$POD" -- cat "/servingrom-results/datasets/$DATASET_ID/collection_progress.json"
else
  kubectl -n infra-learning exec servingrom-rom-results-helper -- cat "/servingrom-results/datasets/$DATASET_ID/collection_progress.json"
fi
