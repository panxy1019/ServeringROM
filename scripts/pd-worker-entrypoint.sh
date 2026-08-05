#!/usr/bin/env bash
set -euo pipefail

STATE_DIR=${STATE_DIR:-/var/run/qwen36-pd}
LOG_DIR=${LOG_DIR:-/var/log/qwen36-pd}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3.6-27B-w8a8}
MODEL_NAME=${MODEL_NAME:-qwen36-27b-w8a8}
PHYSICAL_IDS=${PHYSICAL_IDS:-10,11,12,13,14,15}
RAY_ADDRESS=${RAY_ADDRESS:-ray-vllm-lab-head.infra-learning.svc.cluster.local:6379}
PD_STAGE=${PD_STAGE:-1p2d}

mkdir -p "$STATE_DIR" "$LOG_DIR"
source /usr/local/Ascend/driver/bin/setenv.bash
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64:/usr/local/lib64/python3.12/site-packages/mooncake:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,::1,.svc,.svc.cluster.local"
export no_proxy="$NO_PROXY"
ulimit -n 65536

python3 /opt/qwen36-pd/discover_npu_mapping.py \
  --physical-ids "$PHYSICAL_IDS" \
  --output "$STATE_DIR/npu-mapping.json"

IFS=',' read -r -a PHYSICAL_ID_LIST <<<"$PHYSICAL_IDS"
if [[ ${#PHYSICAL_ID_LIST[@]} -ne 6 ]]; then
  echo "PHYSICAL_IDS must contain exactly six IDs for 1P2D: $PHYSICAL_IDS" >&2
  exit 2
fi

logical_pair() {
  python3 - "$STATE_DIR/npu-mapping.json" "$1" "$2" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
print(f'{data[sys.argv[2]]["logical_id"]},{data[sys.argv[3]]["logical_id"]}')
PY
}

wait_http() {
  local name=$1
  local port=$2
  local pid=$3
  local timeout=${4:-${VLLM_STARTUP_TIMEOUT:-3600}}
  local path=${5:-/health}
  local started=$SECONDS
  while (( SECONDS - started < timeout )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited during startup" >&2
      tail -n 200 "$LOG_DIR/$name.log" >&2 || true
      return 1
    fi
    if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port$path" >/dev/null; then
      echo "$name is healthy on port $port"
      return 0
    fi
    sleep 5
  done
  echo "$name health timeout" >&2
  tail -n 200 "$LOG_DIR/$name.log" >&2 || true
  return 1
}

start_vllm() {
  local name=$1
  local visible_devices=$2
  local api_port=$3
  local kv_role=$4
  local kv_port=$5
  local max_batched_tokens=$6
  local max_num_seqs=$7
  local kv_config
  local eager_args=()

  if [[ "$name" == "prefill" && "${PREFILL_ENFORCE_EAGER:-0}" == "1" ]]; then
    eager_args+=(--enforce-eager)
  fi

  kv_config=$(printf \
    '{"kv_connector":"MooncakeConnectorV1","kv_role":"%s","kv_port":"%s","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":%s,"tp_size":2}}}' \
    "$kv_role" "$kv_port" "$DECODE_DP_SIZE")

  nohup env \
    ASCEND_VISIBLE_DEVICES="$visible_devices" \
    ASCEND_RT_VISIBLE_DEVICES="$visible_devices" \
    HCCL_IF_IP="$POD_IP" \
    GLOO_SOCKET_IFNAME=eth0 \
    TP_SOCKET_IFNAME=eth0 \
    HCCL_SOCKET_IFNAME=eth0 \
    OMP_NUM_THREADS=16 \
    HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
    http_proxy= https_proxy= all_proxy= \
    vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$api_port" \
      --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size 2 \
      --quantization ascend \
      --trust-remote-code \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.88 \
    --max-model-len 32768 \
    --max-num-batched-tokens "$max_batched_tokens" \
    --max-num-seqs "$max_num_seqs" \
    "${eager_args[@]}" \
    --seed 1024 \
    --safetensors-load-strategy eager \
    --kv-transfer-config "$kv_config" \
      >"$LOG_DIR/$name.log" 2>&1 &
  local pid=$!
  echo "$pid" >"$STATE_DIR/$name.pid"
  wait_http "$name" "$api_port" "$pid"
}

stop_children() {
  set +e
  for file in "$STATE_DIR"/*.pid; do
    [[ -f "$file" ]] || continue
    kill "$(cat "$file")" 2>/dev/null || true
  done
  ray stop --force >/dev/null 2>&1 || true
}
trap stop_children EXIT TERM INT

case "$PD_STAGE" in
  1p1d)
    DECODE_DP_SIZE=1
    RAY_RESOURCES='{"NPU":4,"PD_PREFILL":1,"PD_DECODE":1,"QWEN36_PD_WORKER":1}'
    ;;
  1p2d)
    # Decode A/B are independent vLLM engines selected by the proxy. They are
    # replicas, not ranks of a single vLLM data-parallel engine.
    DECODE_DP_SIZE=1
    RAY_RESOURCES='{"NPU":6,"PD_PREFILL":1,"PD_DECODE":2,"QWEN36_PD_WORKER":1}'
    ;;
  *)
    echo "Unsupported PD_STAGE: $PD_STAGE (expected 1p1d or 1p2d)" >&2
    exit 2
    ;;
esac

ray start \
  --address="$RAY_ADDRESS" \
  --node-ip-address="$POD_IP" \
  --num-cpus=64 \
  --resources="$RAY_RESOURCES"

PREFILL_DEVICES=$(logical_pair "${PHYSICAL_ID_LIST[0]}" "${PHYSICAL_ID_LIST[1]}")
DECODE_A_DEVICES=$(logical_pair "${PHYSICAL_ID_LIST[2]}" "${PHYSICAL_ID_LIST[3]}")
printf 'stage=%s\nprefill=%s\ndecode_a=%s\n' \
  "$PD_STAGE" "$PREFILL_DEVICES" "$DECODE_A_DEVICES" \
  >"$STATE_DIR/service-device-map.txt"

# Staged startup is intentional: prove 1P1D first, then add the second decoder.
start_vllm prefill "$PREFILL_DEVICES" 13700 kv_producer 36000 8192 16
start_vllm decode-a "$DECODE_A_DEVICES" 13701 kv_consumer 36100 4096 64

DECODER_HOSTS=(127.0.0.1)
DECODER_PORTS=(13701)
WAIT_PIDS=("$(cat "$STATE_DIR/prefill.pid")" "$(cat "$STATE_DIR/decode-a.pid")")
if [[ "$PD_STAGE" == 1p2d ]]; then
  DECODE_B_DEVICES=$(logical_pair "${PHYSICAL_ID_LIST[4]}" "${PHYSICAL_ID_LIST[5]}")
  printf 'decode_b=%s\n' "$DECODE_B_DEVICES" >>"$STATE_DIR/service-device-map.txt"
  start_vllm decode-b "$DECODE_B_DEVICES" 13702 kv_consumer 36200 4096 64
  DECODER_HOSTS+=(127.0.0.1)
  DECODER_PORTS+=(13702)
  WAIT_PIDS+=("$(cat "$STATE_DIR/decode-b.pid")")
fi

PROXY=/opt/qwen36-pd/pd_proxy.py
nohup python3 "$PROXY" \
  --host 0.0.0.0 \
  --port 8080 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-port 13700 \
  --decoder-hosts "${DECODER_HOSTS[@]}" \
  --decoder-ports "${DECODER_PORTS[@]}" \
  --tokenizer "$MODEL_PATH" \
  --max-prefill-inflight-tokens "${MAX_PREFILL_INFLIGHT_TOKENS:-8192}" \
  >"$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
echo "$PROXY_PID" >"$STATE_DIR/proxy.pid"
wait_http proxy 8080 "$PROXY_PID" 120 /openapi.json

touch "$STATE_DIR/READY"
echo "Qwen3.6 PD 1P2D worker is ready"
wait -n "${WAIT_PIDS[@]}" "$PROXY_PID"
exit 1
