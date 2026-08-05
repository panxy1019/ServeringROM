#!/usr/bin/env bash
set -uo pipefail

export KUBECONFIG=${KUBECONFIG:-/home/admin/k3s.yaml}
NS=${NS:-infra-learning}
APP=${APP:-ray-vllm-pd-worker-qwen36-27b}
MODEL_PATH=${MODEL_PATH:-/models/Qwen3.6-27B-w8a8}
MODEL_NAME=${MODEL_NAME:-qwen36-27b-w8a8}
RUN_ID=${RUN_ID:-pd-scheduler-$(date -u +%Y%m%dT%H%M%SZ)}
RESULT_ROOT=${RESULT_ROOT:-/home/admin/testpanxy/infralearning/qwen36_pd_1p2d/results/$RUN_ID}

POD=$(kubectl -n "$NS" get pod -l app="$APP" -o jsonpath='{.items[0].metadata.name}')
mkdir -p "$RESULT_ROOT/benchmarks" "$RESULT_ROOT/logs" "$RESULT_ROOT/metrics"
printf '%s\n' "$POD" >"$RESULT_ROOT/POD"
printf '%s\n' "$RUN_ID" >"$RESULT_ROOT/RUN_ID"

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
import urllib.request
ports=(13700,13701,13702)
def counter(text, name):
    return sum(float(line.rsplit(None, 1)[1]) for line in text.splitlines()
               if line.startswith("vllm:" + name + "{"))
raise SystemExit(0 if all(
    counter(urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode(), "num_requests_running") == 0
    and counter(urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode(), "num_requests_waiting") == 0
    for port in ports
) else 1)
'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

bench() {
  local name=$1 input_len=$2 output_len=$3 prompts=$4 concurrency=$5
  wait_idle
  snapshot "$name-before"
  kubectl -n "$NS" exec "$POD" -- vllm bench serve \
    --backend openai \
    --base-url http://127.0.0.1:8080 \
    --endpoint /v1/completions \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --num-prompts "$prompts" \
    --num-warmups 2 \
    --request-rate inf \
    --max-concurrency "$concurrency" \
    --seed 20260804 \
    --temperature 0 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --goodput ttft:2000 tpot:80 e2el:30000 \
    --save-result \
    --result-dir /tmp \
    --result-filename "$name.json" 2>&1 | tee "$RESULT_ROOT/logs/$name.stdout.log"
  kubectl -n "$NS" cp "$POD:/tmp/$name.json" "$RESULT_ROOT/benchmarks/$name.json"
  snapshot "$name-after"
  wait_idle || true
}

kubectl -n "$NS" get pod "$POD" -o yaml >"$RESULT_ROOT/pod.yaml"
kubectl -n "$NS" exec "$POD" -- curl -fsS http://127.0.0.1:8080/healthcheck >"$RESULT_ROOT/health-before.json"
kubectl -n "$NS" cp \
  /home/admin/testpanxy/infralearning/qwen36_pd_1p2d/scripts/pd_observer.py \
  "$POD:/tmp/pd_observer.py"
kubectl -n "$NS" cp \
  /home/admin/testpanxy/infralearning/qwen36_pd_1p2d/scripts/live_pd_validation.py \
  "$POD:/tmp/live_pd_validation.py"
kubectl -n "$NS" cp \
  /home/admin/testpanxy/infralearning/qwen36_pd_1p2d/scripts/mixed_pd_load.py \
  "$POD:/tmp/mixed_pd_load.py"
kubectl -n "$NS" exec "$POD" -- python3 /tmp/pd_observer.py --interval 1 \
  >"$RESULT_ROOT/observations.jsonl" 2>"$RESULT_ROOT/logs/observer.stderr.log" &
observer_pid=$!

snapshot baseline
kubectl -n "$NS" exec "$POD" -- python3 /tmp/live_pd_validation.py sequential-route --count 8 \
  >"$RESULT_ROOT/route-sequential.json"
snapshot route-sequential
kubectl -n "$NS" exec "$POD" -- python3 /tmp/live_pd_validation.py prefill-admission --count 3 \
  >"$RESULT_ROOT/prefill-admission.json"
wait_idle || true
snapshot prefill-admission
kubectl -n "$NS" exec "$POD" -- python3 /tmp/mixed_pd_load.py \
  --per-shape 4 --request-rate 1.0 --max-concurrency 16 \
  >"$RESULT_ROOT/mixed-open-1rps.json"
wait_idle || true
snapshot mixed-open-1rps

bench prefill_4096_c2 4096 16 8 2
bench prefill_4096_c4 4096 16 16 4
bench decode_512_c8 512 512 16 8
bench decode_512_c16 512 512 32 16

snapshot final
for name in prefill decode-a decode-b proxy; do
  kubectl -n "$NS" cp "$POD:/var/log/qwen36-pd/$name.log" "$RESULT_ROOT/logs/$name.log" || true
done
kubectl -n "$NS" exec "$POD" -- npu-smi info >"$RESULT_ROOT/npu-final.txt"
kubectl -n "$NS" exec "$POD" -- curl -fsS http://127.0.0.1:8080/healthcheck >"$RESULT_ROOT/health-final.json"
cleanup
observer_pid=""
echo "RESULT_ROOT=$RESULT_ROOT"
