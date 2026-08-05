#!/usr/bin/env bash
set -uo pipefail

STATE_DIR=/var/run/qwen36-eager
LOG_DIR=/var/log/qwen36-eager
MODEL_PATH=${MODEL_PATH:-/models/Qwen3.6-27B-w8a8}
MODEL_NAME=${MODEL_NAME:-qwen36-27b-w8a8-eager}

mkdir -p "$STATE_DIR" "$LOG_DIR"
source /usr/local/Ascend/driver/bin/setenv.bash
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
ulimit -n 65536

{
  date -u --iso-8601=seconds
  echo "=== /proc/1/limits ==="
  cat /proc/1/limits
  echo "=== /proc/1 capabilities ==="
  grep -E '^(CapEff|CapBnd):' /proc/1/status
  echo "=== process memlock ==="
  ulimit -l
  echo "=== allocator ==="
  printf 'PYTORCH_NPU_ALLOC_CONF=%s\n' "$PYTORCH_NPU_ALLOC_CONF"
} >"$LOG_DIR/prestart-container.txt" 2>&1

python3 /opt/qwen36-eager/discover_npu_mapping.py \
  --physical-ids 2,3 \
  --output "$STATE_DIR/npu-mapping.json"

VISIBLE_DEVICES=$(python3 - "$STATE_DIR/npu-mapping.json" <<'PY'
import json
import sys

devices = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
print(f'{devices["2"]["logical_id"]},{devices["3"]["logical_id"]}')
PY
)
printf '%s\n' "$VISIBLE_DEVICES" >"$STATE_DIR/visible-devices.txt"

set +e
env \
  ASCEND_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
  ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES" \
  HCCL_IF_IP="$POD_IP" \
  GLOO_SOCKET_IFNAME=eth0 \
  TP_SOCKET_IFNAME=eth0 \
  HCCL_SOCKET_IFNAME=eth0 \
  OMP_NUM_THREADS=16 \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 13700 \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 2 \
    --quantization ascend \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.88 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --seed 1024 \
    --safetensors-load-strategy eager \
    >"$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "$VLLM_PID" >"$STATE_DIR/vllm.pid"
wait "$VLLM_PID"
STATUS=$?
printf '%s\n' "$STATUS" >"$STATE_DIR/vllm.exit-code"
date -u --iso-8601=seconds >"$STATE_DIR/vllm.exited-at"

# Keep the diagnostic container alive so all evidence can be copied before Pod deletion.
while true; do sleep 3600; done
