# ServingROM v2 Failure Attribution Deep Dive

## 判定摘要

- 主因：200 ms 完成流的零膨胀和 stock/flow 时间错位。
- 次因：一阶状态缺少方向信息；200 ms 增加记忆仅带来小幅输出收益，但明显改善状态 rollout。
- 非主因：POD rank 不足。5 秒输出头从 r=16 增至 r=64 的收益低于 validation 等价带。
- 结构性结论：不存在同时满足状态与关键输出门的单时钟候选，必须采用多速率结构。

## 时间尺度证据

  - 200 ms: completion nonzero=0.0525, key-output NRMSE=0.7749, state NRMSE=0.8897, rank=64, memory=True
  - 1000 ms: completion nonzero=0.2188, key-output NRMSE=0.6450, state NRMSE=0.9539, rank=64, memory=True
  - 2000 ms: completion nonzero=0.3604, key-output NRMSE=0.5628, state NRMSE=0.9844, rank=64, memory=True
  - 5000 ms: completion nonzero=0.5972, key-output NRMSE=0.4663, state NRMSE=0.9841, rank=64, memory=True

200 ms 下 train completion 非零窗口仅约 5%，多数 Y 行是全零或近零事件流。聚合不是平滑装饰，而是在恢复服务过程对应的可观测时间尺度。

## Rank 证据

  - r=16: key-output NRMSE=0.466693
  - r=32: key-output NRMSE=0.466426
  - r=64: key-output NRMSE=0.466336

高 rank 改善状态重构能量，却没有相应改善控制关键输出，说明无监督 POD 的能量排序与 QoS 可预测方向不一致。

## Held-out 状态动力学

- 200 ms fast-state test state NRMSE：`0.667008`。
- 200 ms fast-state transient state NRMSE：`0.601324`。
- fast-state 谱半径：`0.931535`。
- 5 秒单速率候选 state NRMSE 接近 1，主要表现为快速遗忘初态并回归均值，不能因谱半径小就称为有效动力学。

## 标签与可观测性限制

- goodput 使用统一 TTFT=2000 ms、TPOT=100 ms 坐标，跨 workload 语义已经统一。
- Dataset v1.1 缺少 `tpot_valid_count`；精确 mean TPOT 无法仅凭 201,600 个窗口恢复。
- 当前 5 秒输出头是 failure-attribution benchmark，不是最终 MPC plant。后续模型应从 200 ms latent trajectory 生成 5 秒守恒 KPI。
- 当前数据仍没有真实 u[k]，本阶段不能评价可控性或辨识 B 矩阵。
