# ServingROM Telemetry Schema v1

本文定义 Step 1 的独立遥测基础库。当前版本没有接入 Proxy、vLLM、vLLM-Ascend 或 Mooncake，也不改变 D2 请求调度、准入、路由和 KV 语义。

## 启用方式

遥测默认关闭。配置从以下环境变量读取：

| 环境变量 | 默认值 | 语义 |
|---|---|---|
| `SERVINGROM_TELEMETRY_ENABLED` | `false` | 总开关 |
| `SERVINGROM_EXPERIMENT_ID` | `unset` | 实验族标识 |
| `SERVINGROM_RUN_ID` | `unset` | 单次运行标识 |
| `SERVINGROM_CONFIG_ID` | `unset` | 固定系统配置标识 |
| `SERVINGROM_COMPONENT` | `unset` | 进程组件，如 proxy/prefill/decode-0 |
| `SERVINGROM_HOST_ID` | hostname | 宿主或 Pod 稳定标识 |
| `SERVINGROM_OUTPUT_DIR` | `results/servingrom/raw` | 当前进程的输出目录 |
| `SERVINGROM_QUEUE_CAPACITY` | `65536` | 非阻塞内存队列容量 |
| `SERVINGROM_BATCH_SIZE` | `1024` | writer 最大批次 |
| `SERVINGROM_FLUSH_INTERVAL_MS` | `250` | 周期 flush 间隔 |
| `SERVINGROM_MAX_FILE_BYTES` | `268435456` | 单个 JSONL 轮转阈值 |

启用时 experiment/run/config/component/host 必须有明确值。`batch_size` 不得超过 queue capacity。

```python
from servingrom_telemetry import TelemetryConfig, create_emitter

emitter = create_emitter(TelemetryConfig.from_env())
```

关闭时工厂立即返回 `NullEmitter`。该路径没有 writer 线程、队列、锁、文件和完整事件对象；`emit()` 恒定返回 `False`，`flush()`/`close()` 返回 `True`。

## 事件 Schema

版本：`servingrom.telemetry.v1`

每一行 JSONL 是一个完整事件，固定包含以下字段：

| 字段 | 类型 | 语义 |
|---|---|---|
| `schema_version` | string | schema 版本 |
| `event_type` | string | 事件类型 |
| `ts_wall_ns` | integer | 事件 `time.time_ns()` |
| `ts_mono_ns` | integer | 事件 `time.monotonic_ns()` |
| `process_start_wall_ns` | integer | emitter 创建时 wall clock |
| `process_start_mono_ns` | integer | emitter 创建时 monotonic clock |
| `host_id` | string | 宿主/Pod 标识 |
| `component` | string | 进程组件 |
| `process_id` | integer | 操作系统 PID |
| `process_instance_id` | string | 当前进程实例的 128-bit 显示 ID |
| `event_seq` | integer | 进程内从 1 开始严格递增 |
| `experiment_id` | string | 实验标识 |
| `run_id` | string | run 标识 |
| `config_id` | string | 固定配置标识 |
| `trace_id` | string/null | 跨 retry/recompute 的逻辑请求 ID |
| `attempt_id` | integer/null | trace 内从 0 开始的 attempt 序号 |
| `request_id` | string/null | 单次 attempt 的物理请求 ID |
| `external_request_id` | string/null | 客户端关联 ID |
| `payload` | object | 事件类型专属字段 |

请求无关事件仍保留四个请求标识字段，并写为 `null`。

事件唯一键：

```text
(run_id, process_instance_id, event_seq)
```

`event_seq` 只保证单进程内有序，不提供分布式全局序号。发生 queue drop 时序列可能出现可解释缺口，必须结合 summary 中的 drop counter 判断；验收 run 要求 drop=0。

## 标识层级

```text
external_request_id (optional)
  └─ trace_id (stable across retry/recompute)
       ├─ attempt_id=0 ─ request_id=A
       ├─ attempt_id=1 ─ request_id=B
       └─ ...
```

`new_trace_id()` 和 `new_request_id()` 生成 UUID。`build_process_instance_id()` 对 host、component、PID、双启动时钟和随机 nonce 做 SHA256，并取前 32 个十六进制字符。`EventSequence.next()` 使用进程内锁保证多线程调用无重复且严格递增。

## 时间语义

- wall time 用于跨组件近似对齐和人类时间展示；
- monotonic time 用于同一进程内的持续时间、队列等待和阶段耗时；
- 任何 duration 都不得由 wall time 相减得到；
- 当前 schema 不声称不同节点的 monotonic clock 可比较；
- 后续 run metadata 需要保存独立 clock sync 信息。

## 写入路径

业务线程执行：

```text
轻量字段封装 → EventSequence.next → queue.put_nowait → return
```

后台线程执行：

```text
批量取队列 → JSON 序列化 → 批量写入 → 周期 flush → 文件轮转
```

文件名：

```text
<process_instance_id>.00000.jsonl
<process_instance_id>.00001.jsonl
...
<process_instance_id>.summary.json
```

每个进程必须使用独立 `process_instance_id` 和文件集，不允许多个进程共享一个 JSONL 文件。单个事件大于轮转阈值时允许该文件超过阈值，以保持事件行不可拆分。

## Proxy run 目录

Proxy 采集使用不可混用的实验与运行标识：

```text
results/<experiment_id>/<run_id>/
├── metadata/
│   ├── run.yaml
│   ├── deployment.yaml
│   ├── git.json
│   ├── image.json
│   ├── process.json
│   ├── telemetry_config.json
│   ├── schema_versions.json
│   └── sha256_manifest.json
├── raw/proxy/
├── derived/
└── reports/
```

`run.yaml` 当前写成 JSON 语法的 YAML 子集，便于只依赖 Python 标准库读取。run 结束并确认 writer drain 后才生成 `sha256_manifest.json`；清单只读取 raw 文件，不改写原始事件。

`scripts/servingrom/install_proxy_telemetry_sidecar.sh` 只为冻结 D2 Deployment 增加 Python ConfigMap 和持久化结果卷。`scripts/servingrom/configure_proxy_telemetry_run.sh` 设置采集身份和 writer 参数；两者均不修改推理引擎参数、NPU 绑定、准入预算或 Decode 路由公式。

## Flush 与关闭

`flush(timeout_s)` 记录调用时已经入队的目标事件数，等待 writer 处理到该目标并 flush。它允许调用方阻塞，但业务 `emit()` 永不等待 writer。

`close(timeout_s)`：

1. 原子停止接受新事件；
2. 通知 writer 排空队列；
3. flush 并 fsync/关闭当前 JSONL；
4. 原子生成 summary JSON；
5. writer 线程退出。

超时或 flush 失败返回 `False`。重复 close 是幂等的。

## 错误隔离与计数

至少输出：

```text
events_attempted
events_enqueued
events_written
events_dropped_queue_full
events_dropped_writer_failed
serialization_errors
event_build_errors
write_errors
flush_errors
queue_depth_current
queue_depth_high_watermark
writer_batches
writer_bytes
writer_write_latency_ns_total
writer_write_latency_ns_max
```

- queue 满时 `put_nowait()` 失败，事件被丢弃并计数，调用线程不阻塞；
- schema/event build 错误只返回 `False`；
- 单事件序列化失败只丢弃该事件；
- batch 写失败丢弃该 batch，并计入 writer failure；
- writer 和 sink 异常不传播到业务线程；
- 正常验收要求 `events_written == events_enqueued` 且所有 drop/error 为 0。

## 本阶段边界

当前库没有：

- 业务事件 hook；
- 多进程聚合或全局排序；
- 跨节点时钟校正；
- crash 后进程内 queue 恢复；
- 压缩、上传或网络 sink；
- telemetry 动态开关热切换。

这些限制不会改变模型输出或 D2 调度，因为当前业务代码没有导入本包。

## Ascend Mooncake 最小生命周期

Ascend 运行时由 vLLM-Ascend 注册 `MooncakeConnectorV1`。Decode TP worker
通过 `TransferEngine.batch_transfer_sync_read()` 从 Prefill 的已注册 KV 区域拉取数据。
每个实际传输单元只记录以下四类旁路事件：

- `kv_transfer_enqueued`：Decode pull 任务进入 `KVCacheRecvingThread` 队列；
- `kv_transfer_started`：真实 `length_list` 构造完成、进入 TransferEngine 前；
- `kv_transfer_completed`：TransferEngine 返回非负状态；
- `kv_transfer_failed`：TransferEngine 返回负值或实际传输路径抛出异常。

事件保存本地与远端 request ID、source/target engine、执行 TP rank、block 数、
descriptor 数和 `sum(length_list)` 得到的真实字节数。字节数来自 Decode pull
描述符，但语义上表示从 Prefill 注册区域读取的源数据量；Prefill 进程不执行
第二次拷贝，也不会为了遥测重复计算或同步 NPU。

运行时初始化另发一次 `kv_transfer_runtime_capability` marker，用于证明 connector、
worker、TransferEngine、方法、源码、PID 和 TP rank 的实际归属。它不是请求生命周期事件。

派生表分为：

- `kv_transfer_ranks.parquet`：每个 request、Decode engine、TP rank 一行；
- `kv_transfers.parquet`：每个 Proxy attempt 和目标 Decode engine 一行，包含 first start、
  last complete、KV ready、总 bytes、wall time、完成 rank 数和 missing ranks。
