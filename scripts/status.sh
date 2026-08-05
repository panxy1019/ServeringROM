#!/usr/bin/env bash
set -euo pipefail

echo "== service health =="
for item in "prefill:13700" "decode-a:13701" "decode-b:13702" "proxy:8080"; do
  name=${item%%:*}
  port=${item##*:}
  printf '%-10s ' "$name"
  curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null \
    && echo healthy || echo unavailable
done

echo "== device mapping =="
cat /var/run/qwen36-pd/service-device-map.txt 2>/dev/null || true
npu-smi info

echo "== Ray resources =="
ray status || true

