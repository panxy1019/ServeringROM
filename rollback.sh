#!/usr/bin/env bash
set -euo pipefail

KUBECTL=${KUBECTL:-/usr/local/bin/k3s kubectl}
sudo $KUBECTL -n infra-learning scale \
  deployment/ray-vllm-pd-worker-qwen36-27b --replicas=0
sudo $KUBECTL -n infra-learning rollout status \
  deployment/ray-vllm-pd-worker-qwen36-27b --timeout=5m

