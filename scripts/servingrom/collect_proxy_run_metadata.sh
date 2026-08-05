#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NS=${NS:-infra-learning}
DEPLOY=${DEPLOY:-ray-vllm-pd-decode-ab-qwen36-27b}
RESULTS_ROOT=${RESULTS_ROOT:-$ROOT/results}
EXPERIMENT_ID=${EXPERIMENT_ID:?set EXPERIMENT_ID}
RUN_ID=${RUN_ID:?set RUN_ID}
REPOSITORY_COMMIT=${REPOSITORY_COMMIT:?set REPOSITORY_COMMIT}
WORKLOAD=${WORKLOAD:-proxy-integration}
RANDOM_SEED=${RANDOM_SEED:-20260805}
CONFIG_ID=qwen36-1p2d-d2-full-decode-only-async-v1

pod=$(kubectl -n "$NS" get pod -l app="$DEPLOY" -o jsonpath='{.items[0].metadata.name}')
pod_uid=$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.metadata.uid}')
image_tag=$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].image}')
image_digest=$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].imageID}')
pod_ip=$(kubectl -n "$NS" get pod "$pod" -o jsonpath='{.status.podIP}')
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
kubectl -n "$NS" get deployment "$DEPLOY" -o yaml >"$temporary/deployment.yaml"

python3 - "$temporary/run.json" <<PY
import json, pathlib
record = {
  "experiment_id": "$EXPERIMENT_ID",
  "run_id": "$RUN_ID",
  "config_id": "$CONFIG_ID",
  "model": "qwen36-27b-w8a8",
  "tokenizer_revision": "/models/Qwen3.6-27B-w8a8",
  "image_tag": "$image_tag",
  "image_digest": "$image_digest",
  "git_commit": "$REPOSITORY_COMMIT",
  "git": {"commit": "$REPOSITORY_COMMIT", "dirty": False},
  "deployment": "$DEPLOY",
  "pod": "$pod",
  "pod_uid": "$pod_uid",
  "process": {"pod": "$pod", "pod_uid": "$pod_uid", "pod_ip": "$pod_ip"},
  "prefill_endpoints": ["http://127.0.0.1:13700"],
  "decode_endpoints": ["http://127.0.0.1:13701", "http://127.0.0.1:13702"],
  "graph_mode": "FULL_DECODE_ONLY",
  "async_scheduling": True,
  "tp": {"prefill": 2, "decode_a": 2, "decode_b": 2},
  "telemetry": {
    "enabled": True,
    "queue_capacity": int("${SERVINGROM_QUEUE_CAPACITY:-65536}"),
    "batch_size": int("${SERVINGROM_BATCH_SIZE:-1024}"),
    "flush_interval_ms": int("${SERVINGROM_FLUSH_INTERVAL_MS:-250}"),
    "max_file_bytes": int("${SERVINGROM_MAX_FILE_BYTES:-268435456}"),
  },
  "workload": "$WORKLOAD",
  "random_seed": int("$RANDOM_SEED"),
}
pathlib.Path("$temporary/run.json").write_text(json.dumps(record, indent=2) + "\n")
PY

PYTHONPATH="$ROOT" python3 "$ROOT/scripts/prepare_proxy_run.py" \
  --results-root "$RESULTS_ROOT" \
  --run-json "$temporary/run.json" \
  --deployment-yaml "$temporary/deployment.yaml"
