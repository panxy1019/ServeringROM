# ServingROM 内部遥测 Phase A 实施报告

## 最终状态

本轮完成了 Step 3、Step 4B、Step 5B 的代码实现和小规模运行验证，但 **Phase A 验收未通过**，
因此按 fail-closed 规则停止，没有进入 Snapshot Builder、POD、DMDc、MPC 或 actuator 开发。

自动跨层验证的唯一剩余违规是：36 个成功 PD 请求发生了真实 Mooncake KV 传输，但
`kv_transfers.parquet` 为 0 行。原始 Decode 日志能看到 TP rank 上的传输耗时，说明数据面正常，
缺失的是 Python hook 对实际 Ascend Mooncake 路径的覆盖。

Phase A 场景门另有两项未满足：预期的快速 429 实际成为 backend retry 后的 500；Prefill
对账对 36 个成功请求都稳定为 `scheduled_tokens = input_tokens - 1`。后者很可能来自 PD 首 token
边界语义，但在用运行时字段证明之前仍标记为 `requires_runtime_semantics_confirmation`，不宣称通过。

## 已实现内容

### EngineCore 与 Scheduler

- 请求进入、终止和 abort 事件；
- scheduler iteration 及 request membership；
- running/waiting、scheduled tokens、KV block 状态和 preemption；
- output batch 及 token emission 索引；
- async scheduling 的 in-flight iteration ID 关联。

### Ascend ModelRunner

- 每个 TP rank 的 model execution batch；
- unpadded/padded token、padding、graph/eager 模式和 host wall time；
- 不调用 `_sync_device()`，避免为遥测引入设备同步。

### Mooncake

- 已实现 enqueue/start/complete/fail schema；
- P 侧代码按实际 transfer `lengths` 计算 bytes；
- D 侧无法直接取得的 bytes 保持 null；
- 已尝试惰性 emitter、Worker config 克隆及 warmup 预初始化。

这些 hook 在当前 Ascend 运行路径中没有产出事件，因此不能把代码存在视为采集成功。

### 设备采样

- 200 ms 常驻采样；
- 只读取显式 PID 的 `/proc` 和可选 Prometheus exporter；
- 不调用 `ps`、`npu-smi` 或周期性 subprocess；
- 本轮未配置 NPU exporter，所以 AICore/HBM 为 null，并在 capability 文件中明确声明。

## 数据产物

正式诊断 run：

```text
results/servingrom-internal-phase-a/phase-a-20260805T085818Z/
```

生成的派生表：

| 表 | 行数 |
|---|---:|
| `engine_requests.parquet` | 72 |
| `scheduler_iterations.parquet` | 1476 |
| `scheduler_membership.parquet` | 1089 |
| `token_emissions.parquet` | 1085 |
| `model_execution_batches.parquet` | 1172 |
| `device_metrics.parquet` | 2775 |
| `kv_transfers.parquet` | 0 |

Proxy 层共 37 个 trace，37 个终态，生命周期重建违规为 0，JSONL 损坏行为 0。
内部关联在识别 vLLM 的 `chatcmpl-<uuid>-<suffix>` 包装后达到：

- 成功 Prefill 关联：36/36；
- 唯一 Decode 关联：36/36；
- Proxy route 与实际 Decode 匹配：36/36；
- Decode A/B 路由：18/18；
- 剩余违规：`kv_telemetry_missing_for_pd_run=1`。

Decode token 对账为 35 个完整请求 delta=0；主动取消请求没有最终 Proxy output token，因此 delta
为 null。Prefill 的 36 个成功请求均为 delta=-1，需要下一轮补充“交给 Decode 的首 token”显式字段。

## Phase A 覆盖情况

- 短输入/短输出：完成；
- 长输入：完成；
- C8：完成；
- D1/D2 均实际处理请求：完成，18/18；
- 主动取消：完成，记录 `request_cancel` 和 engine abort；
- warmup-measurement-drain：完成 Proxy drain；
- 预期 429：未通过，测试请求最终返回 500，并记录两次 Prefill retry 和最终 `request_error`；
- Mooncake 生命周期：未通过，原始 KV 事件缺失；
- NPU 指标：未配置 exporter，按 schema 写 null；
- Worker writer summary：entrypoint 的 3 秒终止窗口未保证所有子进程 summary 落盘。

## 正确性与开销

Telemetry ON/OFF 固定输出 SHA256 一致。C8 三轮中位数显示 requests/s +0.86%，TTFT P95
-1.06%，E2E P95 -0.95%，没有观察到性能回退。Proxy 事件 1425/1425 写入，dropped=0。
详细数字见 `TELEMETRY_OVERHEAD_REPORT.md`。

## 提交与镜像

核心实现提交：

```text
3a33f22 feat(engine): add scheduler and token lifecycle telemetry
0e6122a feat(kv): add Mooncake transfer telemetry
f4a44ad feat(ascend): add model batch and device telemetry
db2cbdd feat(pipeline): reconstruct and validate internal serving telemetry
```

运行中发现初始化时序和 ID 包装问题后增加了独立修复提交。最终实验镜像为：

```text
110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-servingrom-tel-v4
digest: sha256:de935725d588d35db8449cbb767621681560de34ccb8a1678417cab67f46e66e
```

基础镜像、vLLM commit `0decac0...` 和 vLLM-Ascend commit `5f6faa0...` 均保持冻结。

## 下一步所需缺失字段与修复

1. 在实际被 Ascend worker 调用的 KV connector 边界确认运行时类与源码文件，可先增加一次性
   capability marker，再将 hook 下沉到实际 `TransferEngine` 调用前后。
2. 用实际 `lengths` 记录 P 侧 request bytes，并将 TP rank 级 transfer 合并为 request 级事实。
3. 将预期快速拒绝改成可重复的 Proxy token-budget 429 case，避免由 backend 400/500 代替。
4. 给 Worker 增加显式 telemetry drain RPC 或更长的优雅终止过程，确保所有 summary 原子落盘。
5. 配置 Ascend exporter endpoint 后再验证 AICore、HBM 与 iteration 的 200 ms 对齐。

在以上缺口补齐前，当前数据可用于研究 Proxy、Scheduler、Decode membership 和 ModelRunner batch，
但不能用于构建依赖 KV transfer 状态的 Full-order Snapshot。
