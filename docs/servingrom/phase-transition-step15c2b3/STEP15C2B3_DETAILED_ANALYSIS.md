# ServingROM Step 15C-2B.3 详细分析：Phase-Conditioned Sub-window Transition Kernel

## 1. 目标与边界

Step 15C-2B.2 证明 handoff/waiting dwell 全部短于 200 ms，200 ms age histogram 无法保存窗口内部 phase。本轮不重新采集，而是使用 sealed telemetry 中已有的纳秒时间戳，直接预测：

```text
observed route timestamp
→ predicted KV-ready timestamp
→ predicted Decode-admission timestamp
```

预测事件被聚合成 200 ms flow，同时从同一事件链构造 handoff/waiting inventory，因此 conservation by construction。A/B 使用共享参数，特征中没有 signed A-B quantity。

本轮没有启动 1P2D、没有读取 Round 14.3、没有修改 Dataset v1，也没有实现 actuator realization 或 MPC。

## 2. 可用字段

现有 sealed telemetry 已包含：

```text
route_wall_ns
enqueue_wall_ns
kv_ready_wall_ns
first_decode_membership wall time
actual_total_bytes
block_count
input_tokens / expected_output_tokens
workload / load / arrival process
route 时刻的 A+B running request/token inventory
```

train-only delay audit：

| Delay | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|
| route→enqueue | 22.47 ms | 44.74 ms | 49.19 ms | 85.58 ms |
| Mooncake transfer | 2.72 ms | 6.82 ms | 10.67 ms | 28.97 ms |
| route→KV ready | 25.05 ms | 47.54 ms | 52.31 ms | 88.02 ms |
| KV ready→admission | 33.45 ms | 37.74 ms | 39.71 ms | 149.06 ms |

handoff 主要耗时来自 route→enqueue，而不是 Mooncake transfer；waiting 的主体约为稳定的 33–38 ms scheduler admission delay。

## 3. 候选模型

所有模型仅在 train 拟合，validation 选择，test 在 `FROZEN_SELECTION_BEFORE_TEST.json` 落盘后访问一次。

```text
K0 constant delay
K1 phase + request size + KV size + common load log-delay ridge
K2 workload/load/arrival/input-token/block stratified median
K3 route→enqueue + transfer 两物理头分解后求和
```

K1 最终冻结：

```text
feature_set=phase_size_load
ridge=1.0
validation_score=0.746308
```

K2 没有超过 K1。K3 与 K1 的 validation 指标完全等价，说明把 handoff 拆成两个线性头没有消除请求级 route→enqueue jitter。

## 4. 主要结果

### Validation

| Stage/flow | Request NRMSE | Token NRMSE |
|---|---:|---:|
| handoff inventory | 0.5753 | 0.5484 |
| waiting inventory | 0.9234 | 0.9382 |
| KV-ready flow | 0.2917 | 0.2872 |
| admission flow | 0.4001 | 0.4124 |
| running inventory | 0.4825 | 0.2302 |

### Test

| Stage/flow | Request NRMSE | Token NRMSE |
|---|---:|---:|
| handoff inventory | 0.6273 | 0.6259 |
| waiting inventory | 1.0122 | 1.0630 |
| KV-ready flow | 0.3094 | 0.3147 |
| admission flow | 0.4446 | 0.4647 |
| running inventory | 0.4759 | 0.2479 |

与 Step 15C-2B.2 age-only 约 `1.0` 的结果相比，phase conditioning 对 handoff 与 transition flow 是显著成功的；running propagation 也继续保持原有能力。但 waiting inventory validation/test 未过 `<0.70` 门，因此完整 pipeline 仍不能标记 ready。

## 5. Oracle error attribution

### Validation

| Attribution | Handoff req/tok | Waiting req/tok |
|---|---:|---:|
| oracle KV + predicted waiting | 0 / 0 | 0.602 / 0.641 |
| predicted KV + oracle waiting duration | 0.575 / 0.548 | 0.765 / 0.775 |
| oracle KV + oracle waiting duration | 0 / 0 | 0 / 0 |

### Test

| Attribution | Handoff req/tok | Waiting req/tok |
|---|---:|---:|
| oracle KV + predicted waiting | 0 / 0 | 0.654 / 0.703 |
| predicted KV + oracle waiting duration | 0.627 / 0.626 | 0.825 / 0.834 |
| oracle KV + oracle waiting duration | 0 / 0 | 0 / 0 |

解释：waiting delay 模型在 oracle KV 输入下已经接近或达到门槛；主要误差是 predicted KV timestamp 的几十毫秒偏差传播到 waiting interval。对于短于 200 ms 的稀疏库存，一次边界分类错误就会产生接近单位尺度的 differential error，即使 200 ms admission flow 聚合已经相当准确。

## 6. 为什么不继续堆模型

```text
phase linear model: handoff 显著改善
stratified median: 无额外收益
物理分解双头: 无额外收益
oracle attribution: 主要剩余误差是请求级 KV timestamp jitter
```

继续增加 NN、RNN、Transformer 或 unrestricted tree ensemble，会主要拟合不可预测的请求级 jitter，而不是提高可部署 plant model 的可辨识性。当前应该保留 K1 作为 transition-flow head，而不是继续追逐稀疏 boundary inventory 的逐请求命中。

## 7. 守恒与安全

所有 predicted KV-ready 都由 route event 生成，所有 admission 都由 predicted KV-ready 生成；同一 request/token mass 沿单一路径流动：

```text
route → handoff → KV ready → waiting → admission → running
```

模型不独立裁剪三个 head。validation/test 全部 finite，库存非负，A/B 参数共享，conservation residual 保持浮点误差量级。

## 8. 最终判断

```text
phase_transition_ready=false
transition_pipeline_ready=false
control_rom_ready=false
```

但是本轮获得了可复用的积极结果：

```text
handoff inventory headroom: 有
KV-ready flow headroom: 有
admission flow headroom: 有
running propagation headroom: 有
waiting boundary-inventory headroom: 不足
```

最合理的下一研究决策不是继续增加 delay 模型容量，而是重新定义可控 plant state：保留 200 ms 守恒 flow 与 running inventory，把 handoff/waiting 视为 sub-window transport operator，不再要求它们在粗边界上形成可精确 rollout 的显式库存状态。该状态定义变化需要单独预注册，不能在本轮偷偷放宽 readiness。

## 9. 产物

服务端目录：

```text
/home/admin/servingrom-results/models/servingrom-phase-transition-v1/
```

核心文件包括 `SUBWINDOW_DELAY_AUDIT.json`、`PHASE_KERNEL_ABLATION.json`、`TRANSITION_ERROR_ATTRIBUTION.json`、`PIPELINE_ROLLOUT_METRICS.json`、冻结模型与 SHA256 manifest。
