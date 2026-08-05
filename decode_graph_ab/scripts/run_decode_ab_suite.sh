#!/usr/bin/env bash
set -euo pipefail

export KUBECONFIG=${KUBECONFIG:-/home/admin/k3s.yaml}
NS=${NS:-infra-learning}
PROD_DEPLOY=${PROD_DEPLOY:-ray-vllm-pd-worker-qwen36-27b}
EXP_DEPLOY=${EXP_DEPLOY:-ray-vllm-pd-decode-ab-qwen36-27b}
APP=${APP:-ray-vllm-pd-decode-ab-qwen36-27b}
ROOT=${ROOT:-/home/admin/testpanxy/infralearning/qwen36_pd_1p2d}
LAB_ROOT=$ROOT/decode_graph_ab
RUN_ID=${RUN_ID:-decode-ab-$(date -u +%Y%m%dT%H%M%SZ)}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/results/$RUN_ID}
MODEL_PATH=/models/Qwen3.6-27B-w8a8
MODEL_NAME=qwen36-27b-w8a8
SEED=20260805

mkdir -p "$RESULT_ROOT/baseline"
kubectl -n "$NS" get deploy "$PROD_DEPLOY" -o yaml >"$RESULT_ROOT/baseline/production-deployment.yaml"
kubectl -n "$NS" get pod -l app="$PROD_DEPLOY" -o yaml >"$RESULT_ROOT/baseline/production-pod.yaml"
kubectl -n "$NS" get svc qwen36-pd -o yaml >"$RESULT_ROOT/baseline/production-service.yaml"
PROD_POD=$(kubectl -n "$NS" get pod -l app="$PROD_DEPLOY" -o jsonpath='{.items[0].metadata.name}')
kubectl -n "$NS" exec "$PROD_POD" -- cat /var/run/qwen36-pd/service-device-map.txt >"$RESULT_ROOT/baseline/device-map.txt"
kubectl -n "$NS" exec "$PROD_POD" -- sh -lc \
  'for n in prefill decode-a decode-b; do echo "### $n"; grep -E "Asynchronous scheduling|cudagraph_mode|CUDAGraphMode|enforce_eager" /var/log/qwen36-pd/$n.log | tail -30; done' \
  >"$RESULT_ROOT/baseline/effective-runtime.log" || true
kubectl -n "$NS" exec "$PROD_POD" -- curl -fsS http://127.0.0.1:8080/healthcheck >"$RESULT_ROOT/baseline/health.json"

restore_production() {
  set +e
  kubectl -n "$NS" scale deploy "$EXP_DEPLOY" --replicas=0 >/dev/null 2>&1
  kubectl -n "$NS" wait --for=delete pod -l app="$APP" --timeout=300s >/dev/null 2>&1
  kubectl -n "$NS" scale deploy "$PROD_DEPLOY" --replicas=1 >/dev/null 2>&1
}
trap restore_production EXIT INT TERM

kubectl -n "$NS" scale deploy "$PROD_DEPLOY" --replicas=0
kubectl -n "$NS" wait --for=delete pod -l app="$PROD_DEPLOY" --timeout=600s

wait_idle() {
  local pod=$1
  for _ in $(seq 1 180); do
    if kubectl -n "$NS" exec "$pod" -- python3 -c '
import urllib.request
for port in (13700,13701,13702):
 text=urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",timeout=3).read().decode()
 for name in ("num_requests_running","num_requests_waiting"):
  value=sum(float(line.rsplit(None,1)[1]) for line in text.splitlines() if line.startswith("vllm:"+name+"{"))
  assert value == 0, (port,name,value)
' >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

snapshot_metrics() {
  local pod=$1 path=$2
  kubectl -n "$NS" exec "$pod" -- sh -c '
for port in 13700 13701 13702; do echo "### PORT=$port"; curl -fsS "http://127.0.0.1:$port/metrics"; done
' >"$path"
}

bench() {
  local pod=$1 mode=$2 concurrency=$3 round=$4
  local name="${mode}_c${concurrency}_r${round}"
  wait_idle "$pod"
  snapshot_metrics "$pod" "$RESULT_ROOT/$mode/metrics/$name-before.prom"
  kubectl -n "$NS" exec "$pod" -- vllm bench serve --backend openai \
    --base-url http://127.0.0.1:8080 --endpoint /v1/completions \
    --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" --dataset-name random \
    --random-input-len 512 --random-output-len 512 --num-prompts 32 --num-warmups 2 \
    --request-rate inf --max-concurrency "$concurrency" --seed "$SEED" \
    --temperature 0 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 --goodput ttft:2000 tpot:80 e2el:30000 \
    --save-result --result-dir /tmp --result-filename "$name.json" \
    >"$RESULT_ROOT/$mode/logs/$name.stdout.log" 2>&1
  kubectl -n "$NS" cp "$pod:/tmp/$name.json" "$RESULT_ROOT/$mode/benchmarks/$name.json"
  snapshot_metrics "$pod" "$RESULT_ROOT/$mode/metrics/$name-after.prom"
  wait_idle "$pod"
}

for mode in D0 D1 D2; do
  mkdir -p "$RESULT_ROOT/$mode"/{benchmarks,logs,metrics}
  kubectl -n "$NS" set env deploy/"$EXP_DEPLOY" DECODE_AB_MODE="$mode"
  kubectl -n "$NS" scale deploy "$EXP_DEPLOY" --replicas=1
  kubectl -n "$NS" wait --for=condition=available deploy/"$EXP_DEPLOY" --timeout=3600s
  POD=$(kubectl -n "$NS" get pod -l app="$APP" -o jsonpath='{.items[0].metadata.name}')
  printf '%s\n' "$POD" >"$RESULT_ROOT/$mode/POD"
  kubectl -n "$NS" get pod "$POD" -o yaml >"$RESULT_ROOT/$mode/pod.yaml"
  kubectl -n "$NS" exec "$POD" -- cat /var/run/qwen36-pd/effective-config.txt >"$RESULT_ROOT/$mode/effective-config.txt"
  kubectl -n "$NS" exec "$POD" -- curl -fsS http://127.0.0.1:8080/healthcheck >"$RESULT_ROOT/$mode/health-before.json"
  kubectl -n "$NS" cp "$LAB_ROOT/scripts/decode_ab_observer.py" "$POD:/tmp/decode_ab_observer.py"
  kubectl -n "$NS" cp "$LAB_ROOT/scripts/consistency_probe.py" "$POD:/tmp/consistency_probe.py"
  kubectl -n "$NS" exec "$POD" -- sh -c \
    'nohup python3 /tmp/decode_ab_observer.py --interval 1 >/tmp/decode-ab-observations.jsonl 2>/tmp/decode-ab-observer.err & echo $! >/tmp/decode-ab-observer.pid'

  # Fixed deterministic probe also warms common decode graph sizes.
  kubectl -n "$NS" exec "$POD" -- python3 /tmp/consistency_probe.py --seed "$SEED" --count 8 \
    >"$RESULT_ROOT/$mode/consistency.json"
  for round in 1 2 3; do
    bench "$POD" "$mode" 8 "$round"
    bench "$POD" "$mode" 16 "$round"
  done

  kubectl -n "$NS" exec "$POD" -- sh -c 'kill "$(cat /tmp/decode-ab-observer.pid)" 2>/dev/null || true'
  sleep 2
  kubectl -n "$NS" cp "$POD:/tmp/decode-ab-observations.jsonl" "$RESULT_ROOT/$mode/observations.jsonl"
  kubectl -n "$NS" cp "$POD:/tmp/decode-ab-observer.err" "$RESULT_ROOT/$mode/logs/observer.err"
  for name in prefill decode-a decode-b proxy; do
    kubectl -n "$NS" cp "$POD:/var/log/qwen36-pd/$name.log" "$RESULT_ROOT/$mode/logs/$name.log" || true
  done
  kubectl -n "$NS" exec "$POD" -- npu-smi info >"$RESULT_ROOT/$mode/npu-final.txt"
  kubectl -n "$NS" exec "$POD" -- curl -fsS http://127.0.0.1:8080/healthcheck >"$RESULT_ROOT/$mode/health-after.json"
  kubectl -n "$NS" scale deploy "$EXP_DEPLOY" --replicas=0
  kubectl -n "$NS" wait --for=delete pod -l app="$APP" --timeout=600s
done

python3 "$LAB_ROOT/scripts/analyze_decode_ab.py" "$RESULT_ROOT" | tee "$RESULT_ROOT/analyzer.stdout"
restore_production
trap - EXIT INT TERM
echo "RESULT_ROOT=$RESULT_ROOT"
