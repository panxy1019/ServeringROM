#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://127.0.0.1:8080}
MODEL_NAME=${MODEL_NAME:-qwen36-27b-w8a8}

curl --noproxy '*' -fsS "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a concise assistant.\"},
      {\"role\": \"user\", \"content\": \"用一句中文解释 Prefill 和 Decode 分离。\"}
    ],
    \"temperature\": 0,
    \"max_completion_tokens\": 96
  }" | python3 -m json.tool

