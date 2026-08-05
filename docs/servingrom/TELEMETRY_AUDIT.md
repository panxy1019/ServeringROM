# ServingROM POC 第一阶段：1P2D 遥测只读审计

审计时间：2026-08-05 UTC  
审计范围：当前 `infra-learning` 命名空间中的 Qwen3.6 1P2D 服务及其本地工程  
阶段状态：**审计内容完成；阶段门 FAIL；未进入第二阶段**

## 1. 审计约束与运行时事实

本阶段只读取源码、Deployment、运行日志、Prometheus metrics 和容器版本，没有修改 Proxy、vLLM、Mooncake、Deployment 或调度参数，没有重启服务，也没有发送测试请求。

### 1.1 当前运行对象

| 项目 | 当前值 |
|---|---|
| Deployment | `ray-vllm-pd-worker-qwen36-27b` |
| Pod | `ray-vllm-pd-worker-qwen36-27b-6f7d4c4bd7-f5v8r` |
| 生产副本 | 1，Ready，restart=0 |
| 镜像 | `110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730` |
| 镜像 digest | `sha256:15c3a3db3772807cc09d9ad37756cd973e3b078956b6a239375ec4ae23317133` |
| vLLM | `0.22.1+empty`，commit `0decac0d96c42b49572498019f0a0e3600f50398` |
| vLLM-Ascend | `0.22.1rc1`，commit `5f6faa0cb8830f667266f3b8121cd1383606f2a1` |
| Prefill | TP2，逻辑 NPU 10/11，端口 13700，eager |
| Decode A | TP2，逻辑 NPU 12/13，端口 13701 |
| Decode B | TP2，逻辑 NPU 14/15，端口 13702 |
| Proxy | 端口 8080，token-aware admission，双 Decode 公平最小负载路由 |

### 1.2 D2 基线核验结果

当前生产 Decode A/B 的有效运行配置是：

```text
cudagraph_mode = FULL_AND_PIECEWISE
async_scheduling = enabled
```

实验验证通过的 D2 是：

```text
cudagraph_mode = FULL_DECODE_ONLY
async_scheduling = enabled
```

因此，**当前生产运行时不是 D2**。D2 只冻结在 0 副本 Deployment：

```text
ray-vllm-pd-decode-ab-qwen36-27b replicas=0 mode=D2
```

本阶段没有将生产切换到 D2，因为这样会违反只读审计约束。后续开始采集前必须由独立基线阶段明确选择：采集当前生产事实基线，或先受控切换并验证 D2；二者不能在 metadata 中混称为同一 `config_id`。

## 2. 当前 1P2D 请求生命周期

```text
Client
  │ POST /v1/completions 或 /v1/chat/completions
  ▼
Proxy.handle_completions_impl
  │ 解析请求并用真实模型 tokenizer 计算 prompt_tokens
  │ 读取 max_tokens/max_completion_tokens
  ▼
SharedProxyScheduler.begin_request
  │ 检查 max_prefill_inflight_tokens
  │ accepted：预留 Prefill token/KV 压力
  └ rejected：立即返回 429
  ▼
Proxy.assign_instances
  │ 生成 UUID request_id
  │ 以 X-Request-Id 传给 Prefill
  ▼
Prefill OpenAI API
  │ 请求被改写为 stream=false、max_tokens=1
  │ kv_transfer_params.do_remote_decode=true
  ▼
vLLM Scheduler.add_request / schedule
  │ waiting → running
  │ chunked prefill / KV block allocation
  ▼
NPUModelRunner.execute_model
  │ Prefill forward
  ▼
Scheduler.update_from_output / OutputProcessor.process_outputs
  │ Prefill 生成 1 个占位输出并 length-capped 完成
  ▼
MooncakeConnectorScheduler.request_finished
  │ 延迟释放 Prefill KV block
  │ 返回 remote_engine/block/host/port/request 元数据
  ▼
Proxy 收到 Prefill response
  │ pick_decoder(expected_remaining_tokens)
  │ 记录 PD_ROUTE
  ▼
Decode OpenAI API（同一个 X-Request-Id）
  │ Decode Scheduler waiting → running
  ▼
MooncakeConnectorWorker.start_load_kv
  │ KVCacheRecvingThread 入队
  │ batch_transfer_sync_read
  │ transfer complete / done signal
  ▼
Decode iteration
  │ Scheduler.schedule
  │ NPUModelRunner.execute_model
  │ Scheduler.update_from_output
  │ OutputProcessor detokenize/stream
  ▼
Proxy generate_stream
  │ 首字节记录 PD_FIRST_BYTE
  │ 按 usage 或重新 tokenize 累计 completion_tokens
  │ 更新 Decode 预计剩余 token
  ▼
Client receives stream/final response
  │
  ▼
Proxy finally → finish_request → PD_COMPLETE
```

### 2.1 重试和重计算例外

当 Decode 返回 `stop_reason=recomputed` 时，Proxy 调用 `reassign_instances()`，内部再次调用 `assign_instances()`，会生成一个新的 UUID。当前实现没有稳定的跨 attempt `trace_id`，所以重计算前后的两个 `request_id` 无法天然归并。这是第三阶段必须修复的关联缺口，但不得改变重试、路由或 KV 语义。

## 3. 目标事件与源码映射

### 3.1 Proxy 侧

本地文件：`scripts/pd_proxy.py`

| 目标事件 | 类/函数及行号 | 当前可取得字段 | 需要增加的 hook 字段 |
|---|---|---|---|
| `request_arrival` | `handle_completions_impl()` 1092 | 请求体、chat/completion、stream、真实 `prompt_tokens`、期望输出 token | 双时钟、稳定 trace ID、外部 request ID、SLO 类、原始参数摘要 |
| `admission_decision` | `SharedProxyScheduler.begin_request()` 464；429 分支 1108 | 当前 inflight token、请求 token、limit、accepted/rejected、拒绝原因 | decision 前后负载、候选 Prefill、配置 ID |
| `prefill_queue_enter` | `assign_instances()` 1011；`send_request_to_service()` 924 | request ID、Prefill endpoint、prompt token | Proxy 提交时刻、attempt ID |
| `prefill_complete` | `assign_instances()` 1044 | Prefill HTTP 完成、KV transfer params、Prefill wall duration | 后端完成时刻与 Proxy 收包时刻分离、finish reason |
| `p_to_d_route` | `assign_instances()` 1049；现有 `PD_ROUTE` 1057 | request ID、prompt/output token、P/D endpoint、Prefill duration | 候选分数、选前各 Decode load、tie cursor、route reason |
| `decode_queue_enter` | `stream_service_response_with_retry()` 950 | request ID、Decode endpoint、请求体中的 KV params | Decode HTTP 提交时刻、attempt ID |
| `decode_iteration_token`（Proxy emit 边界） | `generate_stream()` 1155–1216 | chunk、累计文本、usage completion token、首字节时刻 | chunk 内 token 数、每 token wall/mono 时刻；只能表示 Proxy 收到/转发时间 |
| `request_complete` | `generate_stream()` finally 1260 | request ID、Decode endpoint、剩余预计 token | 实际 output token、finish reason、HTTP 状态、输出 SHA256 |
| `request_cancel` | `with_cancellation()` 882；`generate_stream()` CancelledError 1244 | disconnect/cancel 路径、request ID（生成后） | cancel source、所处阶段、已生成 token、资源释放结果 |
| `request_error` | Prefill retry 924；Decode retry 950；stream exception 1252 | endpoint、attempt、异常文本、request ID | 结构化 error class、是否可重试、最终 disposition |

### 3.2 vLLM Engine/Scheduler 侧

容器源码基线：`/vllm-workspace/vllm/vllm/`

| 目标事件/数据 | 文件、类和函数 | 当前对象已有字段 | 需要增加的 hook |
|---|---|---|---|
| 请求创建 | `v1/request.py:Request.__init__()` 59 | request ID、arrival time、prompt token IDs/count、max tokens、KV params、状态、输出 token IDs、computed token、preemption | trace/attempt 透传，进程级事件序列 |
| queue enter | `v1/core/sched/scheduler.py:Scheduler.add_request()` 1755 | request、waiting/running 容器、状态 | queue-enter 双时钟、queue depth、worker role |
| iteration scheduling | `Scheduler.schedule()` 329 | running/waiting、token budget、逐请求 `num_scheduled_tokens`、new/resumed/running/preempted、new KV blocks、scheduled timestamp | iteration ID、开始/结束、scheduler overhead、配置值快照、membership 输出 |
| iteration completion | `Scheduler.update_from_output()` 1283 | sampled token IDs、逐请求 scheduled tokens、request state、finish/status、preemption、KV connector output | iteration结束、逐 token索引/context、KV block after、postprocess边界 |
| Engine iteration | `v1/engine/core.py:EngineCore.step()` 428；`step_with_batch_queue()` 469 | scheduler output、model output、async batch queue | schedule/submit/model-complete/update 四个边界时刻 |
| token emission | `v1/engine/output_processor.py:OutputProcessor.process_outputs()` 576 | request ID、new token IDs、finish/stop reason、KV params、detokenizer、request stats | engine token emit wall/mono 时刻、generated index、is_first_token |
| abort/cancel | `EngineCore.abort_requests()` 374；`OutputProcessor.abort_requests()` 450 | request IDs、内部/外部 abort | abort 原因、阶段、资源回收确认 |

### 3.3 Ascend model runner 侧

容器源码：`/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py`

| 数据 | 类/函数 | 当前可取得字段 | 需要增加的 hook |
|---|---|---|---|
| model batch | `NPUModelRunner.execute_model()` 1900 | `SchedulerOutput`、request IDs、逐请求 scheduled tokens、batch request 数、unpadded/padded token、cudagraph mode | forward begin/end、input prepare、sample、postprocess 分段；iteration ID 透传 |
| batch membership | 同上，`input_batch.req_ids` | request IDs、computed token、scheduled token | 单独 membership JSONL，避免复制大对象到 iteration 行 |
| 图执行信息 | `_determine_batch_execution_and_padding()` 调用点 | graph mode、batch descriptor、padding、ubatch | 旁路记录最终执行模式 |

现有 `profiling_chunk_config.need_timing` 会在计时时调用 `_sync_device()`。它能提供较精确的设备阶段时间，但同步会改变流水线重叠和性能，因此不能作为默认无扰 hook。

### 3.4 Mooncake KV transfer 侧

容器源码：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

| 目标事件 | 类/函数 | 当前可取得字段 | 需要增加的 hook |
|---|---|---|---|
| KV export enqueue | `MooncakeConnectorScheduler.request_finished()` 1660 | request ID、prompt blocks、remote engine/host/port、delay-start wall time | 双时钟、block/byte 总量、iteration ID |
| connector metadata | `build_connector_meta()` 1632 | request ID、local/remote blocks、external/computed token、batch membership | metadata emitted 时刻 |
| KV receive enqueue | `MooncakeConnectorWorker.start_load_kv()` 2549 | local/remote request ID、engine、host/port、block mappings | `kv_transfer_start` 的 enqueue 与 actual-start 分离 |
| transfer actual start | `KVCacheTransferThread._handle_request()` 584；`_transfer_kv_cache_all_groups()` 643 | request ID、remote request ID、session、src/dst/length list | wall/mono start、总 bytes、shard/rank |
| transfer actual complete | `_transfer_kv_cache_all_groups()` 826 | request ID、elapsed ms、IP、device/rank、session | wall/mono end、返回码、bytes、success |
| request transfer finished | `KVCacheTaskTracker.update_done_task_count()` 168；`get_finished()` 2108 | finished send/recv request ID set | send/recv role、完成原因、超时强制释放标志 |

## 4. 当前已有字段

### 4.1 Proxy 内存状态和日志

- active request 数；
- Prefill inflight/waiting token；
- admission rejected total；
- 每个 Decode 的预计剩余 output token；
- request ID；
- tokenizer 计算的 prompt token；
- expected output token；
- Prefill/Decode endpoint；
- Prefill HTTP duration；
- Decode first-byte duration；
- completion 时剩余预计 token。

这些字段目前是内存值或非结构化日志，不包含 schema version、双时钟、run/config 标识和连续 event sequence。

### 4.2 vLLM Prometheus 指标

已有聚合指标包括：

- `num_requests_running/waiting/waiting_by_reason`；
- `prompt_tokens_total/generation_tokens_total/iteration_tokens_total`；
- `kv_cache_usage_perc`；
- `num_preemptions_total`；
- request queue/prefill/decode/inference time；
- TTFT、inter-token latency、E2E；
- request prompt/generation/max token histograms；
- external prefix cache query/hit；
- estimated FLOPs/read/write bytes。

这些指标适合对账和运行监控，但没有逐请求 membership，无法独立重建完整生命周期和 TPOT 序列。

## 5. 需要增加的字段与无法直接取得的字段

### 5.1 可通过旁路 hook 直接取得

- 每次 iteration 的 request membership；
- running/waiting 数量；
- scheduled token 总量及逐请求值；
- prompt/context/output/computed token；
- scheduler token budget、max sequence、chunk threshold；
- request status、finish reason、preemption count；
- KV block ID、used/free/ratio；
- Mooncake request/block/byte/session/rank 和传输耗时；
- engine 侧 token ID、token index、first-token；
- Proxy 侧实际 tokenizer input token、准入与路由快照。

### 5.2 当前不能无扰直接取得

下列字段必须记录为 `null` 并在 capability 文件中声明，除非后续实现了经过开销验证的测量方案：

1. **精确 NPU model forward duration**：CPU 提交区间不是设备执行区间；每 iteration 同步 NPU 会破坏 async/D2 行为。
2. **单请求 NPU utilization/HBM 归因**：设备计数器只能按卡采样，不能可靠归因到请求。
3. **单 iteration CPU 使用量**：可做 100 ms 进程采样，但不能精确分摊给某次 iteration。
4. **未来实际 output token**：到 completion 前只能使用 request 上限或估计值。
5. **客户端真正收到 token 的网络时刻**：服务端只能记录 engine emit、Proxy receive 和 Proxy yield；客户端接收需客户端埋点。
6. **KV export 与 receive 的统一单时钟绝对延迟**：同 Pod 可用单调时钟；跨 Pod/节点扩展后需要 clock sync 与 wall clock 误差界。
7. **固定路由目标比例 `rho`**：当前调度器没有该控制参数，只存在基于预计剩余 token 的最小负载选择。

禁止用 0 表示以上不可用字段。

## 6. 拟修改文件列表

本阶段未修改下列文件。后续阶段预计涉及：

### 6.1 新增

```text
servingrom_telemetry/schema.py
servingrom_telemetry/ids.py
servingrom_telemetry/clock.py
servingrom_telemetry/emitter.py
servingrom_telemetry/async_writer.py
servingrom_telemetry/config.py
servingrom_telemetry/tests/
scripts/validate_telemetry_run.py
scripts/run_servingrom_phase_a.sh
docs/servingrom/TELEMETRY_SCHEMA.md
```

### 6.2 Proxy 与部署

```text
scripts/pd_proxy.py
scripts/pd-worker-entrypoint.sh
k8s/qwen36-pd-worker.yaml
Dockerfile
```

### 6.3 需要通过新镜像固定的 vLLM/vLLM-Ascend hook

```text
/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py
/vllm-workspace/vllm/vllm/v1/engine/core.py
/vllm-workspace/vllm/vllm/v1/engine/output_processor.py
/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py
/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
```

不应在运行 Pod 内 `pip install` 或临时编辑源码。hook 应构建为具有新 digest 的小层镜像，并保留当前镜像作为回退。

## 7. 对推理主路径的潜在性能影响

| 风险 | 影响 | 约束方案 |
|---|---|---|
| 每 token Python 对象和 JSON 序列化 | TPOT、CPU 上升 | 主路径只构造小型不可变事件并 `put_nowait`；后台序列化 |
| iteration membership 列表复制 | 高并发时调度开销 | membership 独立事件；只复制 request ID 和必要整数 |
| 队列满时阻塞 | 直接改变调度 | 有界队列，永不等待；丢弃并计数，Phase A 要求 drop=0 |
| 每事件文件写入/fsync | I/O 抖动 | 批量写入，定时 flush；run 结束再 fsync/manifest |
| hook 异常传播 | 请求失败或引擎死亡 | telemetry 异常必须被隔离；仅递增 failure counter |
| NPU synchronize 计时 | 破坏 async 重叠 | 默认禁止；精确设备时间另做 profiling run |
| 每 100 ms 启动 `npu-smi` 子进程 | CPU/进程泄漏与采样抖动 | 禁止；使用常驻 DCMI/驱动接口或已有 exporter |
| 100 ms `/proc` 扫描全机 | host 开销 | 只从已知 PID 文件递归读取该服务 PID 树，不运行 `ps` |
| 高基数 Prometheus label | 内存膨胀 | request ID 不进入 Prometheus，只进入 JSONL |

## 8. 当前可动态修改的控制参数

### 8.1 已有热路径

| 参数/动作 | 当前接口 | 备注 |
|---|---|---|
| 添加/移除 Prefill 或 Decode backend | Proxy `/instances/add`、`/instances/remove` | 会 drain/isolate backend；不是路由比例控制 |
| 每请求 `max_tokens`、sampling 参数、priority | OpenAI 请求参数 | 属于工作负载输入，不应混作系统控制 |
| Reset prefix cache | Proxy `/reset_prefix_cache` | 当前启动参数关闭 prefix cache，POC 不使用 |

### 8.2 当前运行中变化但不是外部可控参数

- Prefill inflight token accounting；
- Decode expected remaining token；
- 公平 tie cursor；
- 基于最小预计 token 的 Decode 选择；
- vLLM 每 iteration 的实际 scheduled token 和 active batch。

这些是状态或策略内部变量，不是当前可由 ServingROM actuator 热更新的 (u_k)。

## 9. 当前需要重启才能修改的参数

| ServingROM 目标控制 | 当前实现状态 |
|---|---|
| Proxy 最大 Prefill inflight token | 仅启动参数 `--max-prefill-inflight-tokens`，无热更新接口 |
| Prefill token budget | `--max-num-batched-tokens=8192`，启动固定 |
| Prefill chunk size/long-prefill threshold | 当前未显式配置，使用 vLLM 配置；无热更新接口 |
| Decode max active sequences | `--max-num-seqs=64`，启动固定 |
| Decode token budget | `--max-num-batched-tokens=4096`，启动固定 |
| D1/D2 目标路由比例 | 当前不存在；需要新增 actuator，但不得改变默认最小负载策略 |
| async scheduling | 当前平台自动开启；启动时决定 |
| cudagraph mode | 启动时决定；当前生产不是 D2 |
| TP、模型、量化、KV connector、KV port | 启动时决定 |
| NPU 绑定、CPU/内存、镜像 | Pod 重建 |

因此，方法论文中的快速控制输入目前尚不全部具备热更新执行面。数据采集阶段可以记录固定配置与实际调度结果，但不能声称已能闭环控制这些参数。

## 10. 保证不改变基线行为的设计约束

1. `SERVINGROM_TELEMETRY_ENABLED=0` 时不创建 writer 线程、不打开文件、不生成事件对象，走 NullEmitter 快路径。
2. hook 放在调度决定和状态更新之后，只读取局部变量，不调整排序、预算、KV block、request status 或返回值。
3. 主路径只使用非阻塞 `put_nowait`；不允许等待磁盘、锁、网络或后台线程。
4. 不在 hook 中调用 tokenizer；`request_arrival` 复用 Proxy 已计算的 token 数。
5. 不在 iteration hook 中执行 NPU synchronize、`npu-smi`、`ps`、网络请求或 JSON dump。
6. 遥测异常不得传播到 Proxy、Scheduler、ModelRunner 或 Mooncake transfer thread。
7. 关闭遥测与开启遥测使用同一模型、seed、请求、Proxy 和调度配置；Phase A 比较输出 SHA256。
8. image digest、ConfigMap、Deployment、环境变量和 effective engine config 写入每个 run metadata。
9. telemetry on/off 分别重复测量 TTFT、TPOT、吞吐、CPU，设置明确开销门。
10. 原生产镜像和 Deployment 保持可一条命令回退；采集使用独立 Deployment/输出目录。
11. 每个进程独立维护 `event_seq`；唯一性键应为 `(run_id, host_id, component, process_start_mono_ns, event_seq)`，不能假设分布式全局序号。
12. Proxy 生成稳定 `trace_id`；Prefill/Decode attempt 使用独立 `request_id`，重计算时保持 trace 不变。

## 11. 建议的事件边界

为避免同名事件语义模糊，第三阶段应固定以下定义：

- `request_arrival`：Proxy 完成请求体解析和 tokenizer 计数后；
- `admission_decision`：Proxy 原子更新 admission 状态后；
- `prefill_queue_enter`：Prefill EngineCore 接收 request 后；
- `prefill_queue_leave`：首次被 Scheduler 纳入执行 batch 时；
- `prefill_iteration_slice`：一次 SchedulerOutput 中该请求的 scheduled prompt token 区间；
- `prefill_complete`：Prefill engine 产生 length-capped output 和 KV params 后；
- `kv_transfer_start`：Mooncake transfer thread 实际进入 `_transfer_kv_cache_all_groups`，另记录 enqueue 时刻；
- `kv_transfer_complete`：同步 transfer 返回且成功后；
- `decode_queue_enter`：Decode EngineCore 接收 request 后；
- `decode_iteration_token`：EngineCore 接收 sampled token ID 后，同时可另记 Proxy yield 时刻；
- `request_complete`：最终 finish reason 确定且 output token 对账完成；
- `request_cancel`：确认 abort/cancel 并记录最后阶段；
- `request_error`：错误改变 request 最终状态时，普通可恢复 retry 另用 attempt/error 字段；
- `p_to_d_route`：Proxy 原子预留 Decode load 后。

## 12. 第一阶段验收与停止原因

| 检查项 | 状态 | 证据/原因 |
|---|---|---|
| 请求生命周期已描述 | PASS | 第 2 节 |
| 目标事件源码位置已定位 | PASS | 第 3 节 |
| 已有字段与 hook 字段已列出 | PASS | 第 3–5 节 |
| 不可直接取得字段已声明 | PASS | 第 5.2 节 |
| 拟修改文件已列出 | PASS | 第 6 节 |
| 性能影响已分析 | PASS | 第 7 节 |
| 动态/重启参数已区分 | PASS | 第 8–9 节 |
| D2 基线身份一致 | **FAIL** | 当前生产为 FULL_AND_PIECEWISE+async，不是 D2 |
| 第一阶段可独立 Git commit | **FAIL** | `/home/admin/Desktop/sql/qwen36_pd_1p2d` 及父目录没有 `.git` |

依据“任何阶段未通过验收门时停止”的要求，本轮在第一阶段结束后停止。未创建 `servingrom_telemetry/`，未实现 writer、schema、hook、验证脚本或 Phase A 测试，也未改变任何运行时配置。

进入第二阶段前必须先解决：

1. 明确并冻结要采集的真实基线：当前生产事实配置，或已验证的 D2；
2. 建立明确的 Git 仓库边界和忽略规则，使每个阶段能够形成独立、可审计 commit；
3. 为后续 patched vLLM/vLLM-Ascend 镜像确定源码 patch 与构建归属。
