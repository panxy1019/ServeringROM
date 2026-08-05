# Proxy 生命周期重建管线

## 数据边界

本管线只重建 Proxy 可直接观察的事实：请求到达、准入结果、Prefill HTTP 边界、P-to-D 路由、Decode HTTP 边界、首字节、响应块、retry、recompute 和逻辑终态。它不把响应块解释为 token，也不推测 vLLM iteration、KV block 或设备执行状态。

事件唯一键为：

```text
(run_id, process_instance_id, event_seq)
```

逻辑请求以 `trace_id` 聚合；完整重算后的物理执行以 `attempt_id` 分组；同一 attempt 内的 Prefill、Mooncake 参数和 Decode 请求共享同一个 `request_id`。

## 处理流程

```text
raw/proxy/*.jsonl
       │
       ├── JSON/Schema 校验
       ├── 按 process_instance_id 校验 event_seq
       ├── 按 trace_id 聚合
       ├── 按 attempt_id 重建物理执行
       └── 执行终态与顺序不变量检查
               │
               ├── derived/trace_lifecycle.parquet
               ├── derived/attempt_lifecycle.parquet
               ├── reports/proxy_lifecycle_quality.json
               └── reports/proxy_lifecycle_quality.md
```

执行命令：

```bash
python3 -m pip install -r requirements-pipeline.txt
python3 scripts/build_proxy_lifecycle.py results/<experiment_id>/<run_id>
python3 scripts/validate_proxy_lifecycle.py results/<experiment_id>/<run_id>
```

## 不变量

验证器检查：到达与唯一终态、attempt 连续性、request ID 不跨 attempt 复用、Prefill submit/complete、路由、Decode submit、首字节和 completion 的因果顺序、拒绝请求不进入 Prefill、每进程事件序列连续、writer 写入计数守恒以及 JSONL 完整性。

复杂的 Decode 路由状态快照写入 `attempt_lifecycle.parquet` 的 `route_snapshot_json`。保留原 JSON 能避免提前固定 A/B backend 列集合，也便于未来增加 Decode 实例。

## Proxy 无法提供的字段

以下字段必须由后续 engine/Mooncake hook 提供，不能由 Proxy 推断：

- Prefill 和 Decode iteration ID、batch membership；
- 真实 scheduled token 与逐 token emission 时间；
- model forward、scheduler、postprocess 分段耗时；
- KV block 使用量、export/import 起止与传输字节数；
- running/waiting engine request 数；
- preemption、swap、generation counter；
- HBM、AICore 和 engine 进程 CPU；
- Proxy 首字节内部的排队、Prefill、KV transfer 与 Decode 首步细分。

因此当前 `ttft_proxy_ns` 是从 Proxy request arrival 到收到 Decode 首字节的端到端边界，不能替代 engine TTFT 分解；`decode_stream_chunk` 也不能用于精确 TPOT 重算。
