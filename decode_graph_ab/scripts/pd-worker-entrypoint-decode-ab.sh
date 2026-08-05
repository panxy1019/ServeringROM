#!/usr/bin/env bash
set -euo pipefail

STATE_DIR=${STATE_DIR:-/var/run/qwen36-pd}
LOG_DIR=${LOG_DIR:-/var/log/qwen36-pd}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3.6-27B-w8a8}
MODEL_NAME=${MODEL_NAME:-qwen36-27b-w8a8}
PHYSICAL_IDS=${PHYSICAL_IDS:-10,11,12,13,14,15}
RAY_ADDRESS=${RAY_ADDRESS:-ray-vllm-lab-head.infra-learning.svc.cluster.local:6379}
DECODE_AB_MODE=${DECODE_AB_MODE:-D0}

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

python3 /opt/qwen36-pd/discover_npu_mapping.py --physical-ids "$PHYSICAL_IDS" \
  --output "$STATE_DIR/npu-mapping.json"
IFS=',' read -r -a PHYSICAL_ID_LIST <<<"$PHYSICAL_IDS"
[[ ${#PHYSICAL_ID_LIST[@]} -eq 6 ]] || { echo "PHYSICAL_IDS must contain six IDs" >&2; exit 2; }

logical_pair() {
  python3 - "$STATE_DIR/npu-mapping.json" "$1" "$2" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))["devices"]
print(f'{data[sys.argv[2]]["logical_id"]},{data[sys.argv[3]]["logical_id"]}')
PY
}

wait_http() {
  local name=$1 port=$2 pid=$3 timeout=${4:-3600} path=${5:-/health}
  local started=$SECONDS
  while (( SECONDS - started < timeout )); do
    kill -0 "$pid" 2>/dev/null || { tail -n 240 "$LOG_DIR/$name.log" >&2 || true; return 1; }
    curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port$path" >/dev/null && return 0
    sleep 5
  done
  tail -n 240 "$LOG_DIR/$name.log" >&2 || true
  return 1
}

decode_args() {
  case "$DECODE_AB_MODE" in
    D0) printf '%s\n' --no-async-scheduling ;;
    D1) printf '%s\n' --no-async-scheduling --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' ;;
    D2) printf '%s\n' --async-scheduling --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' ;;
    *) echo "Unknown DECODE_AB_MODE=$DECODE_AB_MODE" >&2; return 2 ;;
  esac
}

start_vllm() {
  local name=$1 devices=$2 api_port=$3 role=$4 kv_port=$5 max_tokens=$6 max_seqs=$7
  local kv_config
  local -a extra=()
  kv_config=$(printf '{"kv_connector":"MooncakeConnectorV1","kv_role":"%s","kv_port":"%s","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' "$role" "$kv_port")
  if [[ "$name" == prefill ]]; then
    # Frozen production Prefill: eager execution and otherwise unchanged.
    extra+=(--enforce-eager)
  else
    mapfile -t extra < <(decode_args)
  fi
  printf 'mode=%s service=%s extra_args=' "$DECODE_AB_MODE" "$name" >>"$STATE_DIR/effective-config.txt"
  printf '%q ' "${extra[@]}" >>"$STATE_DIR/effective-config.txt"; printf '\n' >>"$STATE_DIR/effective-config.txt"
  nohup env ASCEND_VISIBLE_DEVICES="$devices" ASCEND_RT_VISIBLE_DEVICES="$devices" \
    HCCL_IF_IP="$POD_IP" GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 \
    HCCL_SOCKET_IFNAME=eth0 OMP_NUM_THREADS=16 HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
    http_proxy= https_proxy= all_proxy= \
    vllm serve "$MODEL_PATH" --host 0.0.0.0 --port "$api_port" \
      --served-model-name "$MODEL_NAME" --tensor-parallel-size 2 --quantization ascend \
      --trust-remote-code --no-enable-prefix-caching --gpu-memory-utilization 0.88 \
      --max-model-len 32768 --max-num-batched-tokens "$max_tokens" --max-num-seqs "$max_seqs" \
      "${extra[@]}" --seed 1024 --safetensors-load-strategy eager --kv-transfer-config "$kv_config" \
      >"$LOG_DIR/$name.log" 2>&1 &
  local pid=$!; echo "$pid" >"$STATE_DIR/$name.pid"
  wait_http "$name" "$api_port" "$pid"
}

stop_children() {
  set +e
  for file in "$STATE_DIR"/*.pid; do [[ -f "$file" ]] && kill "$(<"$file")" 2>/dev/null; done
  ray stop --force >/dev/null 2>&1 || true
}
trap stop_children EXIT TERM INT

ray start --address="$RAY_ADDRESS" --node-ip-address="$POD_IP" --num-cpus=64 \
  --resources='{"NPU":6,"PD_PREFILL":1,"PD_DECODE":2,"QWEN36_PD_DECODE_AB":1}'

PREFILL=$(logical_pair "${PHYSICAL_ID_LIST[0]}" "${PHYSICAL_ID_LIST[1]}")
DECODE_A=$(logical_pair "${PHYSICAL_ID_LIST[2]}" "${PHYSICAL_ID_LIST[3]}")
DECODE_B=$(logical_pair "${PHYSICAL_ID_LIST[4]}" "${PHYSICAL_ID_LIST[5]}")
printf 'mode=%s\nprefill=%s\ndecode_a=%s\ndecode_b=%s\n' "$DECODE_AB_MODE" "$PREFILL" "$DECODE_A" "$DECODE_B" >"$STATE_DIR/service-device-map.txt"

start_vllm prefill "$PREFILL" 13700 kv_producer 36000 8192 16
start_vllm decode-a "$DECODE_A" 13701 kv_consumer 36100 4096 64
start_vllm decode-b "$DECODE_B" 13702 kv_consumer 36200 4096 64

nohup python3 /opt/qwen36-pd/pd_proxy.py --host 0.0.0.0 --port 8080 \
  --prefiller-hosts 127.0.0.1 --prefiller-port 13700 \
  --decoder-hosts 127.0.0.1 127.0.0.1 --decoder-ports 13701 13702 \
  --tokenizer "$MODEL_PATH" --max-prefill-inflight-tokens 8192 >"$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!; echo "$PROXY_PID" >"$STATE_DIR/proxy.pid"
wait_http proxy 8080 "$PROXY_PID" 120 /openapi.json
touch "$STATE_DIR/READY"
wait -n "$(<"$STATE_DIR/prefill.pid")" "$(<"$STATE_DIR/decode-a.pid")" "$(<"$STATE_DIR/decode-b.pid")" "$PROXY_PID"
exit 1
