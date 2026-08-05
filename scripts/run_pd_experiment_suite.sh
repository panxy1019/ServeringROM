#!/usr/bin/env bash
set -uo pipefail

export KUBECONFIG=${KUBECONFIG:-/home/admin/k3s.yaml}
NS=${NS:-infra-learning}
APP=${APP:-ray-vllm-pd-worker-qwen36-27b}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3.6-27B-w8a8}
MODEL_NAME=${MODEL_NAME:-qwen36-27b-w8a8}
RUN_ID=${RUN_ID:-pd-research-$(date -u +%Y%m%dT%H%M%SZ)}
RESULT_ROOT=${RESULT_ROOT:-/home/admin/testpanxy/infralearning/qwen36_pd_1p2d/results/$RUN_ID}
RUN_STEADY=${RUN_STEADY:-1}
STEADY_PROMPTS=${STEADY_PROMPTS:-900}
STEADY_RPS=${STEADY_RPS:-0.5}

POD=$(kubectl -n "$NS" get pod -l app="$APP" -o jsonpath='{.items[0].metadata.name}')
mkdir -p "$RESULT_ROOT" "$RESULT_ROOT/benchmarks" "$RESULT_ROOT/metrics" "$RESULT_ROOT/logs"
printf '%s\n' "$RUN_ID" >"$RESULT_ROOT/RUN_ID"
printf '%s\n' "$POD" >"$RESULT_ROOT/POD"

observer_pid=""
cleanup() {
  if [[ -n "$observer_pid" ]]; then
    kill "$observer_pid" 2>/dev/null || true
    wait "$observer_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

snapshot() {
  local label=$1
  kubectl -n "$NS" exec "$POD" -- sh -c '
    for p in 13700 13701 13702; do
      echo "### PORT=$p"
      curl -fsS "http://127.0.0.1:$p/metrics"
    done
  ' >"$RESULT_ROOT/metrics/$label.prom"
  kubectl -n "$NS" exec "$POD" -- curl -fsS \
    http://127.0.0.1:8080/healthcheck >"$RESULT_ROOT/metrics/$label.proxy.json"
}

wait_idle() {
  local attempt
  for attempt in $(seq 1 120); do
    if kubectl -n "$NS" exec "$POD" -- python3 -c '
import json, urllib.request
ports=(13700,13701,13702)
def value(text,name):
    return sum(float(line.rsplit(None,1)[1]) for line in text.splitlines()
               if line.startswith("vllm:"+name+"{"))
idle=all(value(urllib.request.urlopen(f"http://127.0.0.1:{p}/metrics").read().decode(),
               "num_requests_running") == 0 and
         value(urllib.request.urlopen(f"http://127.0.0.1:{p}/metrics").read().decode(),
               "num_requests_waiting") == 0 for p in ports)
raise SystemExit(0 if idle else 1)
'; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for all engines to become idle" >&2
  return 1
}

run_case() {
  local name=$1 input_len=$2 output_len=$3 prompts=$4 concurrency=$5 request_rate=$6
  local result="$RESULT_ROOT/benchmarks/$name.json"
  local stdout="$RESULT_ROOT/logs/$name.stdout.log"
  local failed="$RESULT_ROOT/logs/$name.failed"

  if [[ -s "$result" ]]; then
    echo "SKIP completed case $name"
    return 0
  fi

  echo "START $name input=$input_len output=$output_len prompts=$prompts concurrency=$concurrency rate=$request_rate"
  wait_idle || return 1
  snapshot "$name-before"
  rm -f "$failed"

  if kubectl -n "$NS" exec "$POD" -- vllm bench serve \
      --backend openai \
      --base-url http://127.0.0.1:8080 \
      --endpoint /v1/completions \
      --model "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --dataset-name random \
      --random-input-len "$input_len" \
      --random-output-len "$output_len" \
      --num-prompts "$prompts" \
      --num-warmups 4 \
      --request-rate "$request_rate" \
      --max-concurrency "$concurrency" \
      --seed 1024 \
      --temperature 0 \
      --ignore-eos \
      --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,95,99 \
      --goodput ttft:2000 tpot:80 e2el:30000 \
      --save-result \
      --result-dir /tmp \
      --result-filename "$name.json" 2>&1 | tee "$stdout"; then
    kubectl -n "$NS" cp "$POD:/tmp/$name.json" "$result"
    snapshot "$name-after"
    echo "PASS $name"
  else
    printf '%s\n' "$(date -u +%FT%TZ)" >"$failed"
    snapshot "$name-failed" || true
    echo "FAIL $name" >&2
  fi
  wait_idle || true
  sleep 3
}

kubectl -n "$NS" get pod "$POD" -o yaml >"$RESULT_ROOT/pod.yaml"
kubectl -n "$NS" exec "$POD" -- sh -c '
  cat /var/run/qwen36-pd/service-device-map.txt
  for f in /var/run/qwen36-pd/*.pid; do echo "$f=$(cat "$f")"; done
  python3 - <<"PY"
from importlib.metadata import version
for package in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "mooncake-transfer-engine"):
    try: print(package, version(package))
    except Exception as exc: print(package, "unavailable", repr(exc))
PY
  npu-smi info
' >"$RESULT_ROOT/runtime-baseline.txt"
snapshot baseline

kubectl -n "$NS" cp \
  /home/admin/testpanxy/infralearning/qwen36_pd_1p2d/scripts/pd_observer.py \
  "$POD:/tmp/pd_observer.py"
kubectl -n "$NS" exec "$POD" -- python3 /tmp/pd_observer.py --interval 1 \
  >"$RESULT_ROOT/observations.jsonl" 2>"$RESULT_ROOT/logs/observer.stderr.log" &
observer_pid=$!

run_case e0_warm_c1             128  32   16  1  inf
run_case e1_balanced_c1         1024 128  8   1  inf
run_case e1_balanced_c2         1024 128  16  2  inf
run_case e1_balanced_c4         1024 128  32  4  inf
run_case e1_balanced_c8         1024 128  48  8  inf
run_case e1_balanced_c16        1024 128  64  16 inf
run_case e1_balanced_c32        1024 128  96  32 inf

run_case e2_prefill_4096_c2     4096 16   8   2  inf
run_case e2_prefill_4096_c4     4096 16   16  4  inf
run_case e2_prefill_4096_c8     4096 16   32  8  inf

run_case e3_decode_512_c2       128  512  8   2  inf
run_case e3_decode_512_c4       128  512  16  4  inf
run_case e3_decode_512_c8       128  512  32  8  inf
run_case e3_decode_512_c16      128  512  48  16 inf
run_case e3_decode_512_c32      128  512  64  32 inf

run_case e4_long_4096_256_c4    4096 256  12  4  inf
run_case e4_long_4096_256_c8    4096 256  24  8  inf

run_case e5_open_0_5rps         512  128  16  64 0.5
run_case e5_open_1rps           512  128  24  64 1
run_case e5_open_2rps           512  128  32  64 2

run_case e6_route_sequential    256  512  4   1  inf
run_case e6_route_concurrent    256  512  8   8  inf

if [[ "$RUN_STEADY" == 1 ]]; then
  run_case e7_steady_30m 512 128 "$STEADY_PROMPTS" 16 "$STEADY_RPS"
fi

snapshot final
for name in prefill decode-a decode-b proxy; do
  kubectl -n "$NS" cp "$POD:/var/log/qwen36-pd/$name.log" \
    "$RESULT_ROOT/logs/$name.log" || true
done
kubectl -n "$NS" exec "$POD" -- npu-smi info >"$RESULT_ROOT/npu-final.txt"
kubectl -n "$NS" exec "$POD" -- curl -fsS \
  http://127.0.0.1:8080/healthcheck >"$RESULT_ROOT/final-health.json"

cleanup
observer_pid=""
echo "RESULT_ROOT=$RESULT_ROOT"

