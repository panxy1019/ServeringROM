#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NS=${NS:-infra-learning}
DEPLOY=ray-vllm-pd-servingrom-qwen36-27b

kubectl -n "$NS" create configmap qwen36-pd-servingrom-d2-scripts \
  --from-file=discover_npu_mapping.py="$ROOT/scripts/discover_npu_mapping.py" \
  --from-file=pd_proxy.py="$ROOT/scripts/pd_proxy.py" \
  --from-file=pd-worker-entrypoint-decode-ab.sh="$ROOT/decode_graph_ab/scripts/pd-worker-entrypoint-decode-ab.sh" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$ROOT/k8s/servingrom/qwen36-servingrom-d2.yaml"
kubectl -n "$NS" scale deployment/"$DEPLOY" --replicas=0
kubectl -n "$NS" get deployment "$DEPLOY" \
  -o custom-columns='NAME:.metadata.name,REPLICAS:.spec.replicas,CONFIG_ID:.spec.template.metadata.labels.servingrom\.openai/config-id,NPUS:.spec.template.metadata.annotations.huawei\.com/Ascend910'

