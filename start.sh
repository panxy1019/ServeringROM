#!/usr/bin/env bash
set -euo pipefail

KUBECTL=${KUBECTL:-/usr/local/bin/k3s kubectl}

if [[ ${NPU_IDLE_CONFIRMED:-} != YES ]]; then
  echo "Refusing to start: run check_target_npus_idle.py on A3 first." >&2
  echo "Then execute: NPU_IDLE_CONFIRMED=YES ./start.sh" >&2
  exit 1
fi

sudo $KUBECTL -n infra-learning scale \
  deployment/ray-vllm-pd-worker-qwen36-27b --replicas=1
sudo $KUBECTL -n infra-learning rollout status \
  deployment/ray-vllm-pd-worker-qwen36-27b --timeout=40m

