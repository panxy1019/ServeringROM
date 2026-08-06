# ServingROM Mooncake 最小生命周期闭环报告

## 1. 结论

本轮只补齐 Ascend 1P2D 的 Mooncake 最小生命周期遥测，没有进入 Snapshot Builder、
POD、DMDc、MPC 或 actuator，也没有修改 Prefill、Decode、Mooncake、Proxy 的调度和
KV 执行语义。

最终验收状态：**PASS**。

- 13/13 成功 PD attempt 唯一关联一个 request-level KV transfer；
- 26/26 Decode TP rank transfer 成功；
- 每个 request 的 expected/completed rank 均为 2，missing ranks 为 0；
- 13/13 KV ready time 可计算，13/13 actual bytes 大于 0；
- Proxy 路由与 Decode worker 关联 13/13 一致；
- Proxy writer `426=426`，内部 writer `4373=4373`；
- drop=0、JSONL 损坏=0、event_seq 缺口=0、跨层质量违规=0；
- telemetry ON/OFF 固定输出内容 SHA256 完全一致；
- Pod restart=0，fatal runtime error scan=0。

实验完成后 ServingROM 实验 Deployment 已缩容到 0，并恢复冻结的原 D2 Deployment。

## 2. 冻结配置

```text
config_id: qwen36-1p2d-d2-full-decode-only-async-v1
topology: 1P2D
Prefill: TP2, eager
Decode A: TP2, FULL_DECODE_ONLY, async scheduling
Decode B: TP2, FULL_DECODE_ONLY, async scheduling
Mooncake: Decode pull, batch_transfer_sync_read
NPU binding: Prefill 10/11, Decode A 12/13, Decode B 14/15
```

没有改变模型权重、TP、KV connector 参数、Proxy 路由、admission、最大序列、token
budget、图模式或 async scheduling。

## 3. 运行时路径诊断

Ascend 插件实际注册：

```text
MooncakeConnectorV1
  -> vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector
  -> MooncakeConnectorWorker
  -> KVCacheRecvingThread
  -> mooncake.engine.TransferEngine.batch_transfer_sync_read
```

最终 run 产生 4 个一次性 capability marker，覆盖 Decode A/B 的 TP rank 0/1。每个
marker 都记录实际源码、PID、engine ID、TP rank、TransferEngine 类型和调用方法。

旧 hook 未命中是因为它修改了 vLLM 通用模块：

```text
vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector
```

Ascend 插件激活后，请求实际实例化 vLLM-Ascend 自有 connector。旧模块能够正常导入和
构建，但不在当前 Ascend 数据面执行。Decode 日志中的 transfer 消息行号和文本只存在于
vLLM-Ascend 文件，这与 capability marker 相互验证。

更完整的路径证据见 `MOONCAKE_RUNTIME_PATH_DIAGNOSIS.md`。

## 4. 最小 Hook 实现

有效补丁：

```text
patches/vllm_ascend/0002-servingrom-mooncake-transfer-telemetry.patch
```

事件边界：

1. `kv_transfer_enqueued`：`add_request()` 将 pull 任务放入 Decode 队列后；
2. `kv_transfer_started`：真实 `length_list` 构造完成、调用 TransferEngine 前；
3. `kv_transfer_completed`：`batch_transfer_sync_read()` 返回非负值后；
4. `kv_transfer_failed`：TransferEngine 返回负值或真实传输路径抛错后。

每个事件包含 local/remote request ID、source/target engine、Decode TP rank、block 数、
descriptor 数、enqueue/start/complete 双时钟、成功状态和错误码。实际字节严格使用：

```python
actual_bytes = sum(length_list)
```

当前 Mooncake 是 Decode pull：字节在 Decode 进程观测，但来源是 Prefill 注册的 KV 区域。
因此事件保留 `transfer_role=receive` 和 source/target engine，不虚构 Prefill Python send。

hook 没有增加 NPU synchronize、KV buffer 复制、额外 transfer、队列重排、锁等待或
Mooncake 调用顺序变化。主路径 `emit()` 仍只做轻量封装和 `put_nowait()`。

## 5. Writer 终止可靠性修复

诊断镜像 v5 证明 KV hook 正确，但 vLLM TP worker 终止时不会可靠执行 Python `atexit`，
导致只有 Proxy/设备进程生成 close summary。最终 v6 在 writer 后台线程中每 5 秒原子写入
一次 counter checkpoint，业务线程不执行文件 I/O。

正式负载结束后等待 6 秒再缩容，因此所有有事件 writer 均留下稳定计数。最终验证聚合为：

```text
Proxy:   events_written=426,  events_enqueued=426
Internal events_written=4373, events_enqueued=4373
dropped_queue_full=0
dropped_writer_failed=0
serialization/write error=0
```

## 6. 镜像与提交

最终镜像：

```text
110.120.0.3:8889/infra/qwen36-pd-worker:
v0.22.1rc1-a3-ray248-servingrom-kv-tel-v6

manifest digest:
sha256:c238c8d56897d5729ccb7c7e8da4125771acc14d937c5727850bb9314d5c4830
```

镜像固定源码：

```text
vLLM:        0decac0d96c42b49572498019f0a0e3600f50398
vLLM-Ascend: 5f6faa0cb8830f667266f3b8121cd1383606f2a1
repository:  a199b61735315177a2d8b4a02cb718ec10be1d71
```

核心提交：

```text
b6e8126 fix(kv): instrument active Ascend Mooncake transfer path
a199b61 fix(telemetry): checkpoint writer counters before worker exit
```

## 7. 正式运行

```text
experiment_id: servingrom-mooncake-minimal
run_id: mooncake-kv-final-20260806T020712Z
Pod restart: 0
```

场景包括短请求、长 Prompt、C8、主动取消、额外长 Prompt 和固定输出对照。成功请求在
Decode A/B 上按 7/6 分流。超上下文拒绝场景仍沿用既有行为：Prefill 返回 400，Proxy
重试后返回 500；这不是 Mooncake、OOM 或 engine death，本轮没有修改该既有语义。

C8 结果：

| 指标 | 数值 |
|---|---:|
| requests/s | 1.1775 |
| TTFT P50 | 5.916 s |
| TTFT P95 | 5.923 s |
| E2E P50 | 6.779 s |
| E2E P95 | 6.793 s |

诊断 v5 的同一 C8 为 1.1736 requests/s；v6 writer checkpoint 后为 1.1775 requests/s，
没有观察到 checkpoint 引入的回退。与此前冻结 Phase A C8 中位数约 1.2347 requests/s
相比，本次单轮低约 4.6%，但本轮不是正式多重复性能 A/B，启动顺序、warmup 和请求历史
不同，不能将这点差异归因于四个 KV 事件。此前 telemetry ON/OFF 三轮对照为约 +0.86%。

## 8. 输出一致性

固定 temperature=0、seed=1024、64 output tokens：

```text
telemetry OFF content SHA256:
e28e1f6446e8d1102173c7e19c214ed0e65aa46141f26d7baf0b19ec7578bb3c

telemetry ON content SHA256:
e28e1f6446e8d1102173c7e19c214ed0e65aa46141f26d7baf0b19ec7578bb3c
```

HTTP 状态均为 200，output tokens 均为 64。输出内容一致，说明旁路 hook 没有改变生成结果。

## 9. KV 统计

| 指标 | 数值 |
|---|---:|
| request transfer | 13 |
| rank transfer | 26 |
| request/rank success | 13/13、26/26 |
| 每请求 expected/completed rank | 2/2 |
| missing ranks | 0 |
| KV ready 可计算 | 13/13 |
| actual bytes 非空 | 13/13 |
| 实际长度总和 | 4,517,855,232 bytes |
| 单请求 bytes P50/P95/max | 254,607,360 / 757,923,840 / 959,250,432 |
| transfer wall P50/P95/max | 2.202 / 215.879 / 220.910 ms |

长 Prompt 的大 KV payload 拉高了 P95，而多数短/中请求集中在约 2 ms。request wall time
使用所有 TP rank 的 first start 到 last complete，因此包含最慢 rank，适合作为 Decode
真正 KV-ready 的保守边界。

## 10. 派生数据与质量报告

正式结果根目录：

```text
results/servingrom-mooncake-minimal/mooncake-kv-final-20260806T020712Z/
```

关键产物：

```text
derived/kv_transfer_ranks.parquet
derived/kv_transfers.parquet
derived/attempt_lifecycle.parquet
derived/trace_lifecycle.parquet
reports/proxy_lifecycle_quality.json
reports/internal_data_quality.json
reports/kv_transfer_summary.json
metadata/sha256_manifest.json
```

`kv_transfer_ranks.parquet` 每个 request、Decode engine、TP rank 一行；
`kv_transfers.parquet` 每个 Proxy attempt 和目标 Decode engine 一行，计算 first start、
last complete、KV ready、总 bytes、wall time、completed ranks 和 missing ranks。

质量验证结果：

```text
Proxy trace: 14/14 有唯一终态
成功 PD attempt: 13/13
Prefill 关联: 13/13
唯一 Decode 关联: 13/13
Proxy route-worker match: 13/13
跨层 violation: 0
fatal runtime error scan: 0
```

主动取消请求的 Decode token 对账按预期为 null；其余完整请求 Decode token delta=0。
Prefill 仍稳定表现为 `scheduled_tokens=input_tokens-1`，属于已知 PD 首 token 边界语义，
不是本轮 Mooncake 缺口，也没有被伪装成精确对账。

## 11. 仍无法无扰取得的字段

以下字段没有通过推测或填 0 伪造：

1. Prefill 进程中的逐请求 native send 起止时间：当前实现为 Decode pull，Prefill 没有
   对称 Python send 调用；
2. NIC/RDMA 硬件完成时间、链路重传和硬件队列深度：需 Mooncake native/C++ counter；
3. 线速 wire bytes：当前是 TransferEngine descriptor 的实际 requested lengths，不包含
   传输协议头，也无法证明底层是否做合并；
4. native engine 内部每 descriptor 的完成时间和重试细节：Python API 只返回聚合 ret；
5. 远端 Prefill TP rank 的显式 native rank 回调：当前可由 handshake/端口映射推导，但为
   避免把推导当事实，Parquet 只把 Decode 执行 rank 作为已验证 TP rank；
6. 设备侧 DMA 开始/完成硬件时间戳：取得它通常需要 profiler 或同步，会改变实验扰动。

这些字段不是进入 Full-order Snapshot Builder 的硬阻断。当前已有 request、attempt、
source/target engine、Decode TP rank、KV ready、实际 descriptor bytes 和双时钟，可构建
KV-aware 的离散窗口状态；硬件级传输分解应作为后续独立低层实验。

## 12. 停止边界

Mooncake 最小生命周期验收通过后，本轮在此停止。没有实现或启动 Full-order Snapshot
Builder、POD、DMDc、MPC 或 actuator。
