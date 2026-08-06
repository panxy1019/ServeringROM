# ServingROM Full-order Snapshot Schema

## 时间与窗口

快照使用统一的墙钟 `ts_wall_ns`，固定周期为 200ms，窗口严格采用半开区间 `[t_k, t_{k+1})`。持续时间只由各进程本地单调时钟计算，绝不跨进程相减。Builder 以事件重放构造状态，不按 JSONL 文件顺序或采样时间分组。

窗口原始产物为 `derived/snapshot_windows.parquet`；质量与覆盖信息为 `derived/snapshot_quality.parquet`。任何缺少所需 engine component 观测的窗口均标为无效，绝不插值。

## 数组产物

- `full_state.npy`：每窗口的请求阶段库存、Prefill/Decode/KV 状态、输入长度和 context 长度直方图。
- `disturbance.npy`：窗口外生到达、接纳、拒绝、Prefill 调度 token 和 scheduler 观测数。
- `output.npy`：输出 token、发射事件和终态计数。
- `next_state.npy`：`full_state[k+1]`，长度为 `N-1`。
- `static_config.json`：固定的 `mu`；无法观测的字段显式为 `null`。

`request_index.json` 仅将稳定 request id 映射到矩阵索引，不保存 prompt 或生成文本。`bin_schema.yaml` 固化高维直方图边界，不能在同一 config_id 下悄悄变化。

## 请求状态机

`ARRIVED -> ADMITTED -> PREFILL_WAITING -> PREFILL_RUNNING -> HANDOFF_WAITING -> KV_QUEUED -> KV_TRANSFERRING -> KV_READY -> DECODE_WAITING -> DECODE_RUNNING -> terminal`。

拒绝路径为 `ARRIVED -> REJECTED`。KV 子状态由 rank 聚合后的 request 级 Parquet 精确定义：有 enqueue 无 start 为 `KV_QUEUED`；有 start 无全部完成为 `KV_TRANSFERRING`；全部预期 TP rank 成功且 `kv_ready_wall_ns` 非空为 `KV_READY`。缺 rank、失败或时间不一致不伪造 ready，而会进入质量违规。

## 数据语义

- `mu`：一次 run 内固定配置，如 token budget、chunk size、graph mode、TP、KV block 参数。
- `d_k`：外部到达、输入长度、预期输出长度、取消与拒绝。
- `x_k`：队列、阶段库存、KV 和 scheduler 内部状态。
- `y_k`：实际输出 token 和终态。

本模块没有 `u_k`，也不写入任何 actuator 或控制命令。
