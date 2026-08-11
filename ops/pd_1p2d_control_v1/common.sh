#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
DEPLOYMENT=${DEPLOYMENT:-ray-vllm-pd-control-pilot-qwen36-27b}
SERVICE=${SERVICE:-qwen36-pd-control-pilot}
KUBECONFIG_PATH=${KUBECONFIG_PATH:-/etc/rancher/k3s/k3s.yaml}
EXPECTED_CONFIG_ID=${EXPECTED_CONFIG_ID:-qwen36-1p2d-d2-full-decode-only-async-control-v1}
EXPECTED_IMAGE=${EXPECTED_IMAGE:-110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-servingrom-snapshot-v7}
EXPECTED_PHYSICAL_IDS=${EXPECTED_PHYSICAL_IDS:-Ascend910-10,Ascend910-11,Ascend910-12,Ascend910-13,Ascend910-14,Ascend910-15}

kube() {
  if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    env KUBECONFIG="$KUBECONFIG_PATH" kubectl "$@"
  else
    sudo env KUBECONFIG="$KUBECONFIG_PATH" kubectl "$@"
  fi
}

pod_name() {
  kube -n "$NAMESPACE" get pod -l app="$DEPLOYMENT" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

require_deployment_identity() {
  local config_id image physical_ids
  config_id=$(kube -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
    -o jsonpath='{.spec.template.metadata.labels.servingrom\.openai/config-id}')
  image=$(kube -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
    -o jsonpath='{.spec.template.spec.containers[0].image}')
  physical_ids=$(kube -n "$NAMESPACE" get deployment "$DEPLOYMENT" \
    -o jsonpath='{.spec.template.metadata.annotations.huawei\.com/Ascend910}')
  [[ "$config_id" == "$EXPECTED_CONFIG_ID" ]] || {
    echo "Refusing operation: config ID is $config_id" >&2; return 2;
  }
  [[ "$image" == "$EXPECTED_IMAGE" ]] || {
    echo "Refusing operation: image is $image" >&2; return 3;
  }
  [[ "$physical_ids" == "$EXPECTED_PHYSICAL_IDS" ]] || {
    echo "Refusing operation: NPU annotation is $physical_ids" >&2; return 4;
  }
}
