# ServingROM 内部遥测 Schema

## 事件归属

内部事件继续使用 `servingrom.telemetry.v1` 公共 envelope。`engine_role`、
`engine_instance`、`tp_rank` 与 `is_driver_rank` 位于 payload；事件唯一键仍为
`(run_id, process_instance_id, event_seq)`。EngineCore iteration 的唯一键为
`(run_id, process_instance_id, engine_iteration_seq)`。

Scheduler/EngineCore 只由 EngineCore driver 写一份。`model_execution_batch` 由每个
TP rank 各写一份。Mooncake 使用独立 component writer，因此不会与 engine writer
共享 JSONL 文件。

## 时间语义

所有 host 区间都同时记录 wall 与 monotonic 时间。单进程 duration 只由 monotonic
时间计算。跨进程关联使用 request ID；跨进程排序仅在 `clock_sync.json` 能力允许时
使用 wall clock。`host_model_submit_duration_ns` 与
`host_execute_duration_ns` 均不是 NPU kernel 时间。

## 不可得字段

- Scheduler 的逐 iteration KV free block 事实没有稳定公开接口，写 `null`。
- Decode 接收侧拿不到 Mooncake 实际 transfer length，`total_bytes` 写 `null`；P 端
  发送事件使用传给 `batch_transfer_sync_write()` 的实际 lengths 求和。
- 设备 exporter 未配置或不暴露某指标时写 `null`，原因进入
  `device_telemetry_capabilities.json`。
- engine output batch 共享一次 EngineCore 时间戳，不伪造逐 token kernel 时间。

## Derived 表

`engine_requests`、`scheduler_iterations`、`scheduler_membership`、
`token_emissions`、`kv_transfers`、`model_execution_batches` 和
`device_metrics` 都由 raw JSONL 单向生成。后处理不得覆盖 raw。

Prefill token 对账会保存 prompt、scheduled 与 delta。占位输出 token 的语义必须
通过当前 vLLM 版本的 Phase A 数据确认，验证器不会用硬编码 `+1` 自动放行。
