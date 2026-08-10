# ServingROM Step 13A：Actuator 可行性审计

## 1. 审计边界

本审计针对冻结的 1P2D D2 配置：Prefill TP2、Decode A/B 各 TP2、Mooncake KV 传输、`FULL_DECODE_ONLY` 和 async scheduling。Dataset v1/v1.1、ServingROM-ROM-v2、200 ms Fast State ROM 与 5 s Slow KPI Head 均未修改。

判定标准不是“Python 属性能否赋值”，而是控制量能否在真实运行中做到：无 Pod/Engine 重启、无模型重载、无 KV 重建、有限时间内可观测生效、可回滚，并保持 graph、async 和输出语义。

## 2. 结论

| ID | 候选 | 分类 | Control-v1 |
|---|---|---|---|
| A1 | Decode A/B routing ratio | `SUPPORTED_WITH_GUARD` | 纳入，`u=rho_A` |
| A2 | Decode scheduler token budget | `DEFERRED` | 不暴露 |
| A3 | Decode max active / max-num-seqs | `NOT_SUPPORTED` | 不暴露 |
| A4 | Prefill scheduling/token budget | `DEFERRED` | 不暴露 |

因此第一版控制维度为一维：

```text
U = [rho_A]
rho_A in [0.2, 0.8]
rho_B = 1 - rho_A
```

## 3. A1：Decode 路由比例

### 3.1 实际对象与生效点

- 源码：`scripts/pd_proxy.py`
- 运行时对象：Proxy scheduler 进程内唯一的 `SharedProxyScheduler`
- 原始入口：`SharedProxyScheduler.pick_decoder()`
- 受控选择器：`RuntimeControlManager.choose_decoder()`
- 确定性算法：`servingrom_control/actuators/routing_ratio.py::weighted_fair_decoder()`
- 调用边界：Prefill HTTP 完成并取得 `kv_transfer_params` 后、Decode HTTP submit 前。

控制关闭或 `control_mode=baseline` 时，代码仍执行冻结的“最小 expected remaining tokens + 公平 tie cursor”算法。受控模式只替换新请求的 decoder 选择，不修改 Prefill、Mooncake 参数、Decode Engine 或已建立的 `InstanceInfo`。

### 3.2 语义

`rho_A` 是未来新 Decode assignment 的长期目标请求比例，而不是已有请求迁移比例。算法以累计 assignment deficit 做确定性选择；同一命令和同一请求序列可复现。若目标 decoder 比另一路高出超过 2048 expected remaining tokens，load guard 临时选择较轻 decoder。

最快生效边界是 `COMMIT` 后的下一次 `pick_decoder()`。`applied_wall_ns` 表示 Proxy 原子交换控制状态的时刻，`effective_from=next_decode_route` 明确区分“已应用”和“被业务请求实际消费”。首个携带相同 `control_command_id` 的 `p_to_d_route` 是观测到的实际生效点。

### 3.3 安全和回退

- 范围：`0.2 <= rho_A <= 0.8`
- 单步变化：`abs(delta rho_A) <= 0.2`
- 最小驻留：5 秒
- CAS：命令 generation 必须严格为当前 generation + 1，且 expected value 匹配
- 健康保护：任一 Decode 缺失或 tainted 时立即进入 `SAFE_BASELINE`
- KV：已有请求不重新路由，不迁移 KV ownership
- 回滚：原子恢复 previous safe value 或冻结 baseline policy

### 3.4 运行时验证

独立实验 Deployment 使用与冻结 D2 相同的镜像、模型、NPU、资源和 vLLM 启动参数，仅将 Proxy 替换为可关闭的 control-v1 版本。实际 smoke 结果见 `RUNTIME_SMOKE_REPORT.md`。运行证据是 A1 晋级为 `SUPPORTED_WITH_GUARD` 的必要条件；若该 smoke 失败，本分类自动降为 `DEFERRED`。

## 4. A2：Decode scheduler token budget

实际字段是 Decode EngineCore 进程中的 `vllm.v1.core.sched.scheduler.Scheduler.max_num_scheduled_tokens`。它在 Scheduler 初始化时从 `scheduler_config.max_num_batched_tokens` 取得，并在每次 `schedule()` 中计算 token budget。

字段的读取频率说明它“可能具备实现热更新的内部条件”，但当前 vLLM 0.22.1 / vLLM-Ascend 0.22.1rc1 没有公开运行时更新 API；Proxy 也不拥有 EngineCore 对象。直接跨进程修改私有字段无法提供命令确认、iteration 生效证据、并发原子性与版本兼容保证。因此本轮不做深层 patch，分类为 `DEFERRED`。

未来晋级至少需要：正式 EngineCore control message、scheduler iteration acknowledgement、旧值回滚、Decode iteration telemetry 中的 configured/effective budget，以及 FULL_DECODE_ONLY 和 async scheduling 回归。

## 5. A3：Decode max active sequences

实际字段是 `Scheduler.max_num_running_reqs`，源于启动参数 `max_num_seqs`。该值不仅限制 Scheduler admission，还参与 KV capacity 与图捕获范围的启动期配置。当前服务没有热更新 API，也没有证明修改后无需 graph recapture 或 KV 重新规划的契约。

因此它不是本版本可安全声明的在线 actuator，分类为 `NOT_SUPPORTED`。未来只有上游提供明确的 dynamic max-active API，并且把逻辑 admission cap 与物理 graph/KV capacity 分离后，才应重新审计。

## 6. A4：Prefill scheduling/token budget

Prefill 使用相同 Scheduler 字段，但还叠加 chunked prefill、Mooncake KV export 和远端 Decode handoff。当前缺少运行时 API和“当前 iteration/下一 iteration”确认事件。私自修改会让 requested 值与真正生效值无法可靠对账，分类为 `DEFERRED`。

## 7. 对 graph、async 与 KV 的影响矩阵

| 候选 | Engine restart | graph recapture | async scheduling | KV manager/已有 KV | 最快边界 |
|---|---:|---:|---|---|---|
| A1 | 否 | 否 | 不接触 | 不接触 | 下一次新 Decode route |
| A2 | 未证明 | 未证明 | 未证明 | 间接改变调度 | scheduler iteration，未建立控制通道 |
| A3 | 需要按启动配置处理 | 可能 | 可能 | 影响容量约束 | 不支持 |
| A4 | 未证明 | Prefill eager，但仍未证明 | 影响调度 | 影响 KV export 节奏 | scheduler iteration，未建立控制通道 |

## 8. 数据语义

只有 `actuator_applied.effective_value` 才能成为后续 Control Dataset 的 `u_k`。目标比例、实际路由比例、waiting、token budget 和 scheduler 输出分别是请求、状态或观测，不得反推为控制输入。

