#!/usr/bin/env python3
"""Interactive multi-turn client for the Qwen PD OpenAI-compatible API."""

import json
import os
import sys
from urllib import error, request


API_BASE = os.environ.get("QWEN_API_BASE", "http://127.0.0.1:18080").rstrip("/")
MODEL = os.environ.get("QWEN_MODEL", "qwen36-27b-w8a8")
SYSTEM_PROMPT = os.environ.get("QWEN_SYSTEM_PROMPT", "你是一个严谨、乐于助人的中文助手。")
MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("QWEN_TEMPERATURE", "0.7"))
TIMEOUT = float(os.environ.get("QWEN_REQUEST_TIMEOUT", "300"))


def chat_completion(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    req = request.Request(
        f"{API_BASE}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=TIMEOUT) as response:
            result = json.load(response)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"无法连接 {API_BASE}: {exc.reason}") from exc

    message = result["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning_content") or ""
    return content, result.get("usage", {})


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"已连接模型 {MODEL}（{API_BASE}）")
    print("命令：/reset 清空上下文，/exit 退出。")

    while True:
        try:
            text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话已结束。")
            return 0

        if not text:
            continue
        if text.lower() in {"/exit", "/quit"}:
            print("对话已结束。")
            return 0
        if text.lower() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("上下文已清空。")
            continue

        messages.append({"role": "user", "content": text})
        try:
            content, usage = chat_completion(messages)
        except (RuntimeError, KeyError, ValueError, json.JSONDecodeError) as exc:
            messages.pop()
            print(f"请求失败：{exc}", file=sys.stderr)
            continue

        messages.append({"role": "assistant", "content": content})
        print(f"\n模型> {content}")
        if usage:
            print(
                "[tokens: "
                f"prompt={usage.get('prompt_tokens', '?')}, "
                f"completion={usage.get('completion_tokens', '?')}, "
                f"total={usage.get('total_tokens', '?')}]"
            )


if __name__ == "__main__":
    raise SystemExit(main())
