# ServingROM POC 数据语义与请求标识

本文只定义 Phase 0B 之后的 schema 语义，不实现 telemetry hook。

## 系统变量

| 符号 | 定义 | 当前例子 | 约束 |
|---|---|---|---|
| `mu` | 一次 run 内固定的系统配置 | token budget、chunk size、`max-num-seqs`、graph mode、async scheduling、拓扑 | 写入 run manifest；run 中不得变化 |
| `d_k` | 窗口 `k` 内的外部请求到达 | arrival 数、输入 token、请求类型 | 是外生扰动，不是控制命令 |
| `x_k` | 调度器与资源内部状态 | running/waiting、KV usage、expected remaining token、tie cursor | 是观测状态或系统输出 |
| `u_k` | actuator 在窗口 `k` 下发并确认生效的命令 | 未来的热更新 token budget 或路由目标 | 必须同时记录 command、ack 和 effective window |

当前 Prefill token budget、chunk size、Decode max active sequences 和目标路由比例没有热更新接口，全部属于 `mu`。实际 scheduled token、expected remaining token、tie cursor 和实际路由比例属于 `x_k` 或输出，不得写为 `u_k`。

## 请求标识

```text
external_request_id (可选客户端关联)
        │
        ▼
trace_id (逻辑请求，跨 retry/recompute 固定)
        │
        ├── attempt_id = 0
        │       └── request_id = 物理 attempt ID，贯穿 Proxy/Prefill/Mooncake/Decode
        ├── attempt_id = 1
        │       └── request_id = 新物理 ID
        └── ...
```

- `trace_id`：由 Proxy 首次接收逻辑请求时生成，retry/recompute 不变；
- `attempt_id`：同一 trace 内从 0 单调递增；
- `request_id`：单次 attempt 的物理 ID，必须传入 Prefill、Mooncake 和 Decode；
- `external_request_id`：客户端提供时保存，只用于关联，不承担内部唯一性。

唯一约束为 `(run_id, trace_id, attempt_id)` 和 `(run_id, request_id)`。本阶段不改变现有 retry/recompute 语义。

## Config ID 隔离

- 正式采集 D2：`qwen36-1p2d-d2-full-decode-only-async-v1`；
- 当前生产参考：`qwen36-1p2d-prod-full-piecewise-async-v1`。

不同 config ID 的事件、run 目录和聚合数据不得合并。任何运行时有效配置与注册表不一致的 run 必须 fail-closed。

