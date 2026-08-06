# ServingROM Full-order Snapshot Schema

## 1. 样本语义

每条训练样本是固定 200 ms 半开窗口：

```text
(x_k, mu, d_k, y_k, x_{k+1}),  window=[t_k,t_{k+1})
```

- `x_k`：`t_k` 时刻仍在系统中的请求库存；由事件状态机回放得到。
- `mu`：整个 run 固定的引擎、拓扑和 SLO 配置，不伪装成控制量。
- `d_k`：窗口内首次外部到达；retry/recompute 不重复计为外部扰动。
- `y_k`：窗口内完成、token、拒绝、取消、KV 完成及 scheduler 工作量。
- `x_{k+1}`：同一状态函数在 `t_{k+1}` 的值，不通过差分近似。

跨进程排序使用 UTC `ts_wall_ns`；单进程 duration 只使用 monotonic clock。边界事件恰好发生在 `t_{k+1}` 时归入下一窗口。

## 2. 请求状态机

成功路径严格互斥：

```text
ADMITTED -> PREFILL_WAITING -> PREFILL_RUNNING
 -> HANDOFF_WAITING -> KV_QUEUED -> KV_TRANSFERRING
 -> KV_READY -> DECODE_WAITING -> DECODE_RUNNING
 -> COMPLETED/CANCELLED/FAILED
```

拒绝路径为 `ARRIVED -> REJECTED`，不进入 active inventory。Prefill/Decode 的首次 schedule 来自真实 `scheduler_membership`；KV ready 使用 TP rank 聚合后的最后一个 complete。一个 request 在任意边界只能有一个主状态。

## 3. 状态维度

`full_state.npy` 本轮为 1804 维。每一维在 `state_index.json` 中记录名称、block、worker、quantity、unit、bin 上下界和解释。

主要 block：

- 标量库存：active、Prefill、handoff、KV、D1/D2 waiting/running、预计剩余 token 和路由不平衡。
- Prefill waiting：input length × waiting age 的 request count 和 token mass。
- TTFT slack：input length × slack 的 count 和 token mass。
- Prefill running：input length × progress 的 count 和 remaining token mass。
- Mooncake：D1/D2 各自 handoff/queue/inflight/ready-wait 的 age 分布和 bytes。
- Decode：D1/D2 各自 context × TPOT slack、context × generation progress、remaining output 与 context mass。
- 尚无首 token 的 Decode request 使用独立 `first_token_pending`，不伪造 previous-token 时间。

当前没有无扰 per-request free KV block 和硬件 DMA progress；字段在 `static_config.json.unavailable_fields` 声明，不以 0 冒充观测值。KV inflight bytes 在 complete 前保持实际总 bytes，complete 时归零，不做线性传输进度假设。

## 4. 扰动与输出

`disturbance.npy` 为 31 维，包括到达/接纳/拒绝、prompt/output token mass、stream 类型和输入/输出长度直方图。

`output.npy` 为 19 维，包括完成与 goodput、TTFT/TPOT 违规和求和、拒绝/取消/error、D1/D2 emitted token、KV 完成量及 Prefill/Decode scheduled token。

Goodput 固定定义为：请求完成，且 Proxy TTFT 不超过 run 默认 2000 ms，engine token 平均 TPOT 不超过 100 ms。没有请求级 SLO 时来源记为 `run_default`。

## 5. 输出目录

```text
derived/snapshots/
├── full_state.npy
├── disturbance.npy
├── output.npy
├── static_config.npy
├── next_state.npy
├── window_table.parquet
├── snapshot_quality.parquet
├── request_state_inventory.parquet
├── state_index.json
├── disturbance_index.json
├── output_index.json
├── static_config.json
├── bin_schema.yaml
└── snapshot_manifest.json
```

`snapshot_manifest.json` 固定所有标准输入 Parquet 与快照输出的 SHA256。后处理不得覆盖 raw。
