# ServingROM D2 Deployment Smoke Test

执行时间：2026-08-05 04:45-04:58 UTC

## 基线身份

```text
config_id=qwen36-1p2d-d2-full-decode-only-async-v1
deployment=ray-vllm-pd-decode-ab-qwen36-27b
image=110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730
image_digest=sha256:15c3a3db3772807cc09d9ad37756cd973e3b078956b6a239375ec4ae23317133
pod_uid=ff9e42fb-4d05-4d52-9965-b46102884cfa
```

生产 Deployment `ray-vllm-pd-worker-qwen36-27b` 已缩容到 0；冻结 D2 Deployment 已扩容到 1。本次没有启动 `ray-vllm-pd-servingrom-qwen36-27b`。

## 验收结果

| 检查 | 状态 | 证据 |
|---|---|---|
| Pod Ready | PASS | `Running`, Ready=true |
| restart=0 | PASS | 启动后和请求后均为 0 |
| Prefill | PASS | 13700 `/health` 成功 |
| Decode A | PASS | 13701 `/health` 成功 |
| Decode B | PASS | 13702 `/health` 成功 |
| Proxy | PASS | 8080 `/healthcheck` 返回 status=ok，1P+2D |
| graph mode | PASS | Decode A/B 均记录 `CUDAGraphMode.FULL_DECODE_ONLY` |
| async scheduling | PASS | Decode A/B 均记录 `Asynchronous scheduling is enabled` |
| 固定普通请求 | PASS | HTTP 200，prompt=5 tokens，completion=16 tokens |
| Mooncake/engine/OOM | PASS | 严格 severity 扫描无命中，Pod lastState 为空 |

## Effective config

```text
mode=D2 service=prefill extra_args=--enforce-eager
mode=D2 service=decode-a extra_args=--async-scheduling --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
mode=D2 service=decode-b extra_args=--async-scheduling --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
```

设备映射：

```text
prefill=10,11
decode_a=12,13
decode_b=14,15
```

## 请求结果

请求参数：

```json
{"model":"qwen36-27b-w8a8","prompt":"The capital of France is","max_tokens":16,"temperature":0,"seed":20260805}
```

结果为 HTTP 200，文本以 `Paris.` 开头。Proxy 日志显示请求完整经过 Prefill，并成功路由到 Decode B；同一 smoke 期间的前一次请求成功路由到 Decode A，因此两路 Decode 都实际完成过请求。

## 启动警告说明

Decode B 在首次加载历史 torch compile cache 时出现 `Compiling model again due to a load failure` 警告，随后自动重新编译并成功启动。这不是引擎死亡：

- 日志级别为 `WARNING`；
- 后续 `Asynchronous scheduling is enabled`；
- readiness 成功；
- 推理 HTTP 200；
- restart=0；
- 无 `ERROR`、`CRITICAL`、`EngineDeadError`、OOM 或 Mooncake transfer failure。

本轮只执行 deployment smoke，没有重新运行 C1/C8/C16 或正式输出 SHA256 A/B。

