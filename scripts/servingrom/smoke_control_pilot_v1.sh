#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-infra-learning}
DEPLOYMENT=${DEPLOYMENT:-ray-vllm-pd-control-pilot-qwen36-27b}
EXPERIMENT_ID=${EXPERIMENT_ID:-servingrom-control-pilot-v1}
CONFIG_ID=${CONFIG_ID:-qwen36-1p2d-d2-full-decode-only-async-control-v1}
RUN_ID=${RUN_ID:-control-pilot-capability-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
KUBECTL=${KUBECTL:-kubectl}

pod=$($KUBECTL -n "$NAMESPACE" get pod -l app="$DEPLOYMENT" -o name | head -1)
if [[ -z "$pod" ]]; then
  echo "Pilot Pod not found" >&2
  exit 1
fi

deactivate() {
  $KUBECTL -n "$NAMESPACE" exec "$pod" -- \
    python3 /opt/qwen36-pd/ensure_control_baseline.py --label "$RUN_ID" || true
  $KUBECTL -n "$NAMESPACE" exec "$pod" -- \
    python3 /opt/qwen36-pd/servingrom_run_control.py deactivate \
      --run-id "$RUN_ID" --experiment-id "$EXPERIMENT_ID" \
      --config-id "$CONFIG_ID" --timeout 120 || true
}
trap deactivate EXIT

$KUBECTL -n "$NAMESPACE" exec "$pod" -- \
  python3 /opt/qwen36-pd/servingrom_run_control.py activate \
    --run-id "$RUN_ID" --experiment-id "$EXPERIMENT_ID" \
    --config-id "$CONFIG_ID" --timeout 120 >/tmp/control-pilot-activate.json

$KUBECTL -n "$NAMESPACE" exec "$pod" -- curl --noproxy '*' -fsS \
  -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-27b-w8a8","messages":[{"role":"user","content":"Reply with exactly: ServingROM pilot ready"}],"temperature":0,"seed":1401,"max_tokens":32}' \
  >/tmp/control-pilot-response.json

sleep 3
deactivate
trap - EXIT

python3 - "$RUN_ID" <<'PY'
import hashlib
import json
import sys

data = json.load(open("/tmp/control-pilot-response.json", encoding="utf-8"))
content = data["choices"][0]["message"]["content"]
print(f"RUN_ID={sys.argv[1]}")
print("HTTP_RESPONSE_OK=1")
print(f"CONTENT={content}")
print(f"OUTPUT_SHA256={hashlib.sha256(content.encode()).hexdigest()}")
print(f"USAGE={json.dumps(data.get('usage'), sort_keys=True)}")
PY

$KUBECTL -n "$NAMESPACE" exec "$pod" -- sh -lc \
  "find /servingrom-results/$EXPERIMENT_ID/$RUN_ID -maxdepth 3 -type f -printf '%P\\n' | sort"
$KUBECTL -n "$NAMESPACE" get "$pod" \
  -o custom-columns=UID:.metadata.uid,READY:.status.containerStatuses[0].ready,RESTARTS:.status.containerStatuses[0].restartCount \
  --no-headers
