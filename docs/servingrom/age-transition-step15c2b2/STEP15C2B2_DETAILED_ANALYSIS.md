# ServingROM Step 15C-2B.2 详细分析：Age-Structured Semi-Markov Transition ROM

## 1. 实验边界

本轮严格复用 Step 15C-2B.1 已封存的三阶段库存和确定性守恒骨架，完全离线执行：

```text
未启动 1P2D
未重新采集 run
未修改 sealed Control Dataset v1
未读取 Round 14.3 held-out benchmark
未修改 gc12-diff2、12D global/common 或 running service head
未实现 actuator realization 或 MPC
```

train、validation、test 始终按完整 run 隔离。执行顺序为：

```text
读取 train lifecycle
→ 统计 dwell distribution
→ 冻结 AGE_BIN_SCHEMA.json
→ 读取 validation 并选择 smoothing
→ 写入 FROZEN_SELECTION_BEFORE_TEST.json
→ 一次读取 test
```

## 2. Train-only dwell-time audit

两个 transit stage 的 request-level residence time 如下：

| Stage | P50 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| handoff | 25.05 ms | 37.21 ms | 44.57 ms | 47.54 ms | 52.31 ms | 88.02 ms |
| waiting | 33.45 ms | 34.98 ms | 36.66 ms | 37.74 ms | 39.71 ms | 149.06 ms |

关键事实不是 dwell time 很长，而是恰好相反：全部 train handoff/waiting residence time 都短于一个 200 ms telemetry window。也就是说，大多数请求在两个相邻边界之间完成：

```text
route → KV ready → first Decode membership
```

边界库存往往同时看不到其进入和离开。200 ms 快照保留了窗口净结果，却没有保留 transit cohort 在窗口内所处的 phase。

expected output token 与 dwell 的相关性很弱：handoff `-0.0431`，waiting `-0.0269`。A/B 只用于 symmetry audit，参数始终共享；waiting A/B median relative difference 仅 `0.58%`，handoff 为 `13.23%`，没有证据支持为两路 Decode 分别拟合 hazard。

## 3. 冻结的 age schema

train-only schema 为：

```text
[0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, +inf) seconds
```

实现没有把宽 bin 当作一次 transition。内部 cohort 每 200 ms 严格推进一个 micro-step，最后 tail self-loop；对外 sidecar 才聚合到 4 个 age bins。每个 decoder、stage 和 age bin 同时维护 request 与 remaining-token mass。

生成的只读 sidecar：

```text
/home/admin/servingrom-results/models/servingrom-age-transition-v1/
└── sidecar/age_structured_inventory_200ms.parquet
```

共 108,000 行，train、validation、test 各 36,000 行。observed age histogram 聚合回 Step 15C-2B.1 stage inventory 的最大残差为 0，证明 sidecar 构造本身正确。

## 4. H0/H1/H2 决策

### H0：冻结的 memoryless baseline

H0 直接引用 Step 15C-2B.1 的封存结果，不重新调参。其 validation handoff/waiting differential rollout 大约为 `0.99`。

### H1：age-only hazard

使用 train-only exposure/exit count 和 Beta smoothing，validation 在 `0.5/1/2` 中冻结 `smoothing=2.0`。A/B 共享 hazard；request 和 token mass 分别估计。

由于 3,752 个 train lifecycle 的两个 transit dwell 全部落在首个 200 ms age interval，首 bin hazard 退化为：

```text
request hazard = 0.9994675
token hazard   = 0.9999956
```

后续无样本 bins 只保留平滑先验 0.5。这意味着 age histogram 在 200 ms 时钟上几乎没有可供学习的多年龄状态，H1 实际接近“所有新流量在一窗内离开”。

H1 validation：

| Stage | Request differential NRMSE | Token differential NRMSE |
|---|---:|---:|
| handoff | 0.999531 | 0.999996 |
| waiting | 0.999519 | 0.999996 |

H1 相对 H0 的平均改善为 `-0.864%`，即没有实质改善。所有 workload/load 分组均保持约 `0.9995/1.0`，不是某个负载点导致的局部失败。

### H2：未执行

预注册条件要求 H1 先有明显改善但仍未过门，才允许加入少量 common-load correction。实际 H1 没有改善，而且失败来源是 sub-window phase 被 200 ms 边界抹除。对同一首 age bin 增加 total-running 或 common-load scalar，无法恢复请求在该窗口内何时 route、何时 KV ready、何时 admission。

因此：

```text
H2 executed=false
common_load_hazard_needed=false
```

这不是为了省略候选，而是按预注册门控阻止无解释力的额外容量。

## 5. Transition flow 与 running attribution

虽然 stage snapshot rollout 未改善，H1 对窗口聚合 transition flow 仍有一定预测力：

| Split | Flow | Request NRMSE | Token NRMSE |
|---|---|---:|---:|
| validation | KV ready | 0.5098 | 0.5256 |
| validation | admission | 0.7279 | 0.7373 |
| test | KV ready | 0.5036 | 0.5079 |
| test | admission | 0.7140 | 0.7084 |

这说明 residence distribution 能预测一部分聚合 outflow，但不足以恢复稀疏的 A-B transit inventory 快照。

冻结 running head 的 attribution：

| Split | Admission | Running request NRMSE | Running token NRMSE |
|---|---|---:|---:|
| validation | oracle | 0.4774 | 0.2141 |
| validation | predicted | 0.4929 | 0.2778 |
| test | oracle | 0.4668 | 0.2206 |
| test | predicted | 0.4837 | 0.2639 |

predicted admission 接入后仍满足预注册的 `<0.60/<0.40` 门，证明 frozen running service head 没有退化，transition error 传播也没有破坏 running prediction。当前阻断只在 handoff/waiting 状态定义与 200 ms 采样尺度之间。

## 6. 守恒、校准与稳定性

```text
validation conservation residual max = 1.10e-13
test conservation residual max       = 9.40e-14
negative inventory                   = 0
outflow greater than inventory       = 0
zero inventory / zero inflow audit   = PASS
A/B parameter sharing                = PASS
all validation/test runs finite      = PASS
```

train survival curve的 request MAE 为 `1.72e-4`，token-mass MAE 为 `1.43e-6`。这说明 hazard 对 request-level dwell distribution 的拟合本身很好；rollout 失败并非 hazard calibration 差，而是 200 ms snapshot 不能观测 20–50 ms transit phase。

## 7. 四个问题的明确回答

### 1）handoff/waiting 失败是否主要由 residence-time/age state 缺失造成？

不是以“200 ms age histogram”能够表达的方式造成。真实 residence time 很重要，但全部短于一个采样窗口，因此 coarse age state没有新增可观测信息。更准确的缺口是 **sub-window entry phase**，而不是跨多个 200 ms 窗口的长期 memory。

### 2）age-only 是否足够，是否需要 common-load-dependent hazard？

age-only 不足；当前证据也不支持 H2。common load 无法恢复同一窗口内的 route/KV/admission 相位，贸然加入只会增加容量而不解决辨识问题。

### 3）predicted flow 接入 frozen running head 后能否维持能力？

可以。validation 为 `0.4929/0.2778`，test 为 `0.4837/0.2639`，与 Step 15C-2B.1 的约 `0.49/0.28` 一致并通过 running gate。

### 4）是否有足够 oracle plant-side headroom 进入 rho_A → routing forcing？

还没有。running head有 headroom，但完整 route→handoff→waiting→running pipeline 的 transit state gate 未通过，因此不能进入 actuator realization。

## 8. 下一步

本轮应停止，不增加 NN/RNN/Transformer/Koopman，不扩大 history，也不重新采集。下一项最有判别力的离线诊断是利用已经存在的纳秒时间戳测试：

```text
route_phase_in_200ms_window
request expected-token bucket
KV transfer bytes / block count bucket
```

建立 phase-conditioned sub-window transition kernel，直接预测每个窗口的 KV-ready/admission flow，而不是试图在 200 ms 边界保存几乎总为空的 transit inventory。只有该模型在 validation 上显著改善 transition flow，并保持 running propagation，才值得讨论 plant-side pipeline readiness。

## 9. 最终状态

```text
stage_inventory_ready=true
age_transition_ready=false
transition_pipeline_ready=false
control_rom_ready=false
```

服务端封存目录：

```text
/home/admin/servingrom-results/models/servingrom-age-transition-v1/
```

单元测试：`10 passed`。sidecar 行数和 split 隔离正确，11 个 manifest 产物 SHA256 复算无 mismatch。
