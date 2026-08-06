#!/usr/bin/env bash
set -euo pipefail

STATE_DIR=${STATE_DIR:-/var/run/qwen36-pd}
LOG_DIR=${LOG_DIR:-/var/log/qwen36-pd}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3.6-27B-w8a8}
MODEL_NAME=${MODEL_NAME:-qwen36-27b-w8a8}
PHYSICAL_IDS=${PHYSICAL_IDS:-10,11,12,13,14,15}
RAY_ADDRESS=${RAY_ADDRESS:-ray-vllm-lab-head.infra-learning.svc.cluster.local:6379}
DECODE_AB_MODE=${DECODE_AB_MODE:-D2}
SERVINGROM_RESULTS_ROOT=${SERVINGROM_RESULTS_ROOT:-/servingrom-results}
: "${SERVINGROM_EXPERIMENT_ID:?SERVINGROM_EXPERIMENT_ID is required}"
: "${SERVINGROM_RUN_ID:?SERVINGROM_RUN_ID is required}"
: "${SERVINGROM_CONFIG_ID:?SERVINGROM_CONFIG_ID is required}"
export SERVINGROM_RUN_ROOT="$SERVINGROM_RESULTS_ROOT/$SERVINGROM_EXPERIMENT_ID/$SERVINGROM_RUN_ID"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$SERVINGROM_RUN_ROOT"/{metadata,derived,reports} \
  "$SERVINGROM_RUN_ROOT"/raw/{proxy,prefill,decode-0,decode-1,mooncake,device}
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
    D2) printf '%s\n' --async-scheduling --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' ;;
    *) echo "ServingROM collection requires frozen D2 mode" >&2; return 2 ;;
  esac
}

start_vllm() {
  local name=$1 component=$2 devices=$3 api_port=$4 role=$5 kv_port=$6 max_tokens=$7 max_seqs=$8
  local kv_config
  local -a extra=()
  kv_config=$(printf '{"kv_connector":"MooncakeConnectorV1","kv_role":"%s","kv_port":"%s","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}' "$role" "$kv_port")
  if [[ "$name" == prefill ]]; then extra+=(--enforce-eager); else mapfile -t extra < <(decode_args); fi
  printf 'mode=%s service=%s extra_args=' "$DECODE_AB_MODE" "$name" >>"$STATE_DIR/effective-config.txt"
  printf '%q ' "${extra[@]}" >>"$STATE_DIR/effective-config.txt"; printf '\n' >>"$STATE_DIR/effective-config.txt"
  nohup env ASCEND_VISIBLE_DEVICES="$devices" ASCEND_RT_VISIBLE_DEVICES="$devices" \
    HCCL_IF_IP="$POD_IP" GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 HCCL_SOCKET_IFNAME=eth0 \
    OMP_NUM_THREADS=16 HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
    SERVINGROM_TELEMETRY_ENABLED=true SERVINGROM_COMPONENT="$component" \
    SERVINGROM_ENGINE_ROLE="$component" SERVINGROM_ENGINE_INSTANCE="$component" \
    SERVINGROM_OUTPUT_DIR="$SERVINGROM_RUN_ROOT/raw/$component" \
    SERVINGROM_RUN_ROOT="$SERVINGROM_RUN_ROOT" \
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
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    local alive=0
    for file in "$STATE_DIR"/*.pid; do
      [[ -f "$file" && -d "/proc/$(<"$file")" ]] && alive=1
    done
    (( alive == 0 )) && break
    sleep 1
  done
  ray stop --force >/dev/null 2>&1 || true
}
trap stop_children EXIT TERM INT

ray start --address="$RAY_ADDRESS" --node-ip-address="$POD_IP" --num-cpus=64 \
  --resources='{"NPU":6,"PD_PREFILL":1,"PD_DECODE":2,"QWEN36_PD_SERVINGROM":1}'

PREFILL=$(logical_pair "${PHYSICAL_ID_LIST[0]}" "${PHYSICAL_ID_LIST[1]}")
DECODE_A=$(logical_pair "${PHYSICAL_ID_LIST[2]}" "${PHYSICAL_ID_LIST[3]}")
DECODE_B=$(logical_pair "${PHYSICAL_ID_LIST[4]}" "${PHYSICAL_ID_LIST[5]}")
printf 'mode=%s\nprefill=%s\ndecode_a=%s\ndecode_b=%s\n' "$DECODE_AB_MODE" "$PREFILL" "$DECODE_A" "$DECODE_B" >"$STATE_DIR/service-device-map.txt"

start_vllm prefill prefill "$PREFILL" 13700 kv_producer 36000 8192 16
start_vllm decode-a decode-0 "$DECODE_A" 13701 kv_consumer 36100 4096 64
start_vllm decode-b decode-1 "$DECODE_B" 13702 kv_consumer 36200 4096 64

nohup env SERVINGROM_TELEMETRY_ENABLED=true SERVINGROM_COMPONENT=proxy \
  SERVINGROM_OUTPUT_DIR="$SERVINGROM_RUN_ROOT/raw/proxy" \
  python3 /opt/qwen36-pd/pd_proxy.py --host 0.0.0.0 --port 8080 \
    --prefiller-hosts 127.0.0.1 --prefiller-port 13700 \
    --decoder-hosts 127.0.0.1 127.0.0.1 --decoder-ports 13701 13702 \
    --tokenizer "$MODEL_PATH" --max-prefill-inflight-tokens 8192 \
    --decode-stream-chunk-telemetry >"$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!; echo "$PROXY_PID" >"$STATE_DIR/proxy.pid"
wait_http proxy 8080 "$PROXY_PID" 120 /openapi.json

nohup env SERVINGROM_TELEMETRY_ENABLED=true SERVINGROM_COMPONENT=device \
  SERVINGROM_OUTPUT_DIR="$SERVINGROM_RUN_ROOT/raw/device" \
  python3 /opt/servingrom/scripts/collect_servingrom_device_metrics.py \
    --interval-ms 200 \
    --pid "$(<"$STATE_DIR/prefill.pid")" --pid "$(<"$STATE_DIR/decode-a.pid")" \
    --pid "$(<"$STATE_DIR/decode-b.pid")" --pid "$PROXY_PID" \
    >"$LOG_DIR/device-collector.log" 2>&1 &
echo "$!" >"$STATE_DIR/device-collector.pid"

touch "$STATE_DIR/READY"
wait -n "$(<"$STATE_DIR/prefill.pid")" "$(<"$STATE_DIR/decode-a.pid")" \
  "$(<"$STATE_DIR/decode-b.pid")" "$PROXY_PID"
exit 1
