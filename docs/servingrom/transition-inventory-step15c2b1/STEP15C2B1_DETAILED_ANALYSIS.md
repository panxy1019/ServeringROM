# ServingROM Step 15C-2B.1 三阶段库存、确定性重放与转移流模型报告

## 1. 结论

本轮完全离线执行，没有启动 1P2D、没有重新采集 run、没有读取 Round 14.3 held-out，也没有实现 MPC。输入仅来自已封存的 `servingrom-control-dataset-v1` 及其 36 个 run 对应的只读 lifecycle、scheduler、KV 和 token 派生文件。

最终状态：

- `observed_flow_conservation_pass=true`
- `transition_flow_model_trained=true`
- `control_rom_ready=false`
- train、validation、test 的 handoff、waiting、running 三阶段 request/token replay 均为 `NRMSE=0`、`exact_fraction=100%`
- 当前 sealed telemetry 足以重建三阶段库存，不存在必须重新部署模型或补采事件的缺口

## 2. 三阶段库存定义

对每个已路由到 Decode A/B 的物理 attempt，使用四个可观测边界建立半开区间：

```text
handoff: [p_to_d_route, kv_ready)
waiting: [kv_ready, first_decode_scheduler_membership)
running: [first_decode_scheduler_membership, terminal)
```

每个阶段同时维护两种库存：

```text
request inventory: 当前区间内的请求数
token inventory: expected_output_tokens - 当前边界前已产生的客户端可见 Decode tokens
```

状态按 Decoder A/B 对称构造，因此每个 200 ms 边界的库存张量为：

```text
X_inventory[decoder=A/B, stage=handoff/waiting/running, quantity=requests/tokens]
```

产出的只读 sidecar 共 108,000 行：train、validation、test 各 36,000 行。它不修改 Dataset v1 本体。

## 3. Observed-flow 确定性重放

逐 200 ms 窗口使用真实事件流执行：

```text
handoff[k+1] = handoff[k] + route - kv_ready - handoff_token_service
waiting[k+1] = waiting[k] + kv_ready - decode_admission - waiting_token_service
running[k+1] = running[k] + decode_admission - running_token_service - terminal_residual
```

request 流的 token service 为 0；token 流使用 Decode engine 的 `new_token_count`。请求在 terminal 时仍未生成的 expected token 作为 `terminal_residual` 出库，以覆盖 stop、cancel 或实际输出短于请求上限的情形。

### 3.1 关键语义修正

`token_emissions.parquet` 同时包含 Prefill 的内部 handoff token 和 Decode 的真实输出序列。每个成功请求通常存在：

```text
Prefill: generated_token_index=0，一条内部事件
Decode: generated_token_index=0..N-1，N 条客户端输出事件
```

若两者全部计入 service，会把第一个 token 重复扣减。最终实现只使用所选 Decode engine 的 emissions；本轮显式排除了 11,403 条 Prefill 内部 token 事件，数量与 PD-routed attempts 一致。

另一个边界修正是：若 stage 在窗口中间开始，stage 首个 snapshot 已经包含此前产生 token 后的 remaining 值，不能在同一 snapshot 再扣一次。这一规则由专门单元测试覆盖。

### 3.2 重放结果

| Split | 阶段 | Request NRMSE / exact | Token NRMSE / exact |
|---|---|---:|---:|
| train | handoff | 0 / 100% | 0 / 100% |
| train | waiting | 0 / 100% | 0 / 100% |
| train | running | 0 / 100% | 0 / 100% |
| validation | handoff | 0 / 100% | 0 / 100% |
| validation | waiting | 0 / 100% | 0 / 100% |
| validation | running | 0 / 100% | 0 / 100% |
| test | handoff | 0 / 100% | 0 / 100% |
| test | waiting | 0 / 100% | 0 / 100% |
| test | running | 0 / 100% | 0 / 100% |

36 个 run 中共有 11,403 个 PD-routed attempts，全部具有 trace terminal、KV ready、Decode admission 和合法阶段顺序。325 个没有 PD Decode route 的 attempt 被显式标为 `ignored_without_pd_route`，不进入三阶段库存；它们不是阶段字段缺失。

## 4. Transition/service flow model

守恒门通过后才进行模型训练。test 在 validation 冻结选择前没有被读取。模型使用 1 s 聚合窗口，对三个流分别拟合共享 A/B 参数的非负 ridge 模型：

```text
handoff source + routed inflow -> KV-ready flow
waiting source + KV-ready inflow -> Decode-admission flow
running source + admission inflow -> token-service + terminal outflow
```

共享 A/B 参数保持交换对称性；预测流被限制在当前库存加本窗 inflow 以内，避免产生负库存。ridge 候选为 `1e-4, 1e-2, 1, 10, 100`，validation 冻结 `ridge=1.0`。

### 4.1 A-B differential free rollout

| Split | 阶段 | Request NRMSE | Token NRMSE |
|---|---|---:|---:|
| validation | handoff | 0.9924 | 0.9929 |
| validation | waiting | 0.9892 | 0.9902 |
| validation | running | 0.4912 | 0.2756 |
| test | handoff | 0.9926 | 1.0027 |
| test | waiting | 0.9952 | 0.9914 |
| test | running | 0.4832 | 0.2632 |

### 4.2 技术解释

running 阶段已经能获得有意义的预测，尤其 token differential 的 test NRMSE 为 0.263。原因是 Decode service 在 1 s 尺度上相对连续，当前库存和 admission inflow 对短期 outflow 有较强解释力。

handoff 与 waiting 接近 train-mean baseline。它们是持续时间很短、事件稀疏且受 dwell-time 强烈影响的 transit states；仅使用“当前库存、当前 inflow、全局总量”的无记忆线性 rate model，无法知道单个 cohort 已经在阶段中停留多久，因此无法准确预测下一窗转移。这不是库存不可观测或守恒失败，而是 transition head 缺少 age/dwell-time 状态。

因此当前模型是诊断性 transition/service baseline，不应直接标记为最终 Control-ROM。最可能有效的下一步是保持已闭合的确定性库存骨架，只替换 handoff/waiting 两个失败 head：按 cohort age bins 或 survival/hazard 表示 stage residence time；running service head 可保留为基线。无需重新采集数据，因为 route、KV ready、admission、terminal 的时间戳已经能够离线构造 dwell age。

## 5. 缺失事件与字段

所有阶段已经严格闭合，因此本轮没有必须补采的事件或字段：

```text
missing_trace=0
missing_kv_ready=0
missing_admission=0
invalid_stage_order=0
```

若后续追求更精确的 online hazard model，可从现有时间戳离线构造 `time_since_route`、`time_since_kv_ready` 和 cohort age bins；这属于特征工程，不是 telemetry 缺口。

## 6. 产物与复现

服务端结果目录：

```text
/home/admin/servingrom-results/models/servingrom-transition-inventory-v1/
```

主要产物：

```text
sidecar/stage_inventory_200ms.parquet
models/transition_service_flow_model.npz
OBSERVED_FLOW_REPLAY.json
MISSING_FIELD_AUDIT.json
STAGE_INVENTORY_MANIFEST.json
TRANSITION_FLOW_ABLATION.json
FROZEN_SELECTION_BEFORE_TEST.json
evaluation/final_metrics.json
SHA256_MANIFEST.json
```

复现命令：

```bash
/root/miniconda3/envs/ray-submit/bin/python \
  scripts/run_transition_inventory_step15c2b1.py \
  --dataset-root /home/admin/servingrom-results/datasets/servingrom-control-dataset-v1 \
  --forcing-root .sources/servingrom-control-dataset-v1-control-windows \
  --outflow-root .sources/servingrom-control-dataset-v1-outflow \
  --output-root /home/admin/servingrom-results/models/servingrom-transition-inventory-v1 \
  --config configs/servingrom_transition_inventory_v1.json
```

单元及回归测试结果为 `16 passed`。封存目录中 9 个受 manifest 管理的产物均通过 SHA256 复算，无 mismatch。
