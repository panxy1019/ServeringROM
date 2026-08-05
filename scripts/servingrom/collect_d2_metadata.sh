#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NS=${NS:-infra-learning}
DEPLOY=ray-vllm-pd-servingrom-qwen36-27b
APP=$DEPLOY
RUN_ID=${RUN_ID:?set RUN_ID for the recovery run}
OUT=${OUT:-$ROOT/results/$RUN_ID/metadata}
mkdir -p "$OUT"

pod=$(kubectl -n "$NS" get pod -l app="$APP" -o jsonpath='{.items[0].metadata.name}')
kubectl -n "$NS" get deployment "$DEPLOY" -o yaml >"$OUT/deployment.yaml"
kubectl -n "$NS" get pod "$pod" -o yaml >"$OUT/pod.yaml"
kubectl -n "$NS" get pod "$pod" -o jsonpath='{.metadata.uid}{"\n"}' >"$OUT/pod_uid.txt"
kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].image}{"\n"}{.status.containerStatuses[0].imageID}{"\n"}' >"$OUT/image_tag_and_digest.txt"
kubectl -n "$NS" get configmap qwen36-pd-servingrom-d2-scripts -o json >"$OUT/configmap.json"
python3 - "$OUT/configmap.json" >"$OUT/configmap_sha256.txt" <<'PY'
import hashlib, json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
data = x.get("data", {})
blob = b"".join(k.encode() + b"\0" + data[k].encode() + b"\0" for k in sorted(data))
print(hashlib.sha256(blob).hexdigest())
PY
kubectl -n "$NS" exec "$pod" -- cat /var/run/qwen36-pd/effective-config.txt >"$OUT/effective-engine-config.txt"
kubectl -n "$NS" exec "$pod" -- cat /var/run/qwen36-pd/service-device-map.txt >"$OUT/logical-npu-binding.txt"
kubectl -n "$NS" exec "$pod" -- sh -lc 'env | sort' >"$OUT/environment.txt"
kubectl -n "$NS" logs "$pod" >"$OUT/startup.log"
kubectl -n "$NS" exec "$pod" -- sh -lc 'git -C /vllm-workspace/vllm rev-parse HEAD; git -C /vllm-workspace/vllm-ascend rev-parse HEAD' >"$OUT/source-commits.txt"
printf 'proxy=8080\nprefill=13700\ndecode_a=13701\ndecode_b=13702\n' >"$OUT/ports.txt"

