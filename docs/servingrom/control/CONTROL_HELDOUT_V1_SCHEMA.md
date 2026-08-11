# ServingROM Control Held-out v1

## 隔离语义

该 benchmark 的全部 run 固定标记为 `test/control-heldout`，不属于 Control
Dataset v1 的 train/validation/test。构建器只读引用冻结训练数据集的质量摘要，
不会修改或重新合并 `servingrom-control-dataset-v1`。

## 10-run 矩阵

- 核心 8 run：`balanced/mixed-bimodal × 75%/92% × interpolation/unseen-composite`；
- robustness 2 run：`mixed-bimodal 92% × slow-ramp/boundary-near`；
- 每个 run：120 秒 warmup、600 秒 measurement、库存清空后结束；
- arrival RNG 与 trajectory RNG 使用独立 seed；
- 拓扑保持 Prefill TP2 + Decode A TP2 + Decode B TP2。

## 控制与坐标

唯一控制输入仍为 `U[1]=actuator_applied.effective_value=rho_A`。每 200 ms
生成 `(X,D,U,X_next)`，每 5 秒生成守恒 KPI。realized request/token ratio、
running/waiting/remaining-token imbalance 都是响应，不是控制输入。

控制族：

- `interpolation`：主控制档位仅为 0.4/0.6；
- `unseen-composite`：混合 0.4/0.6 与已知 0.3/0.5/0.7；
- `slow-ramp`：0.3→0.4→0.5→0.6→0.7 的离散慢斜坡；
- `boundary-near`：经 0.4/0.6 合法过渡至 0.2/0.8。

所有 planned dwell 至少 15 秒，`abs(delta_U)<=0.2`。任何控制拒绝、CAS
错误、safety fallback、writer drop、生命周期守恒失败、Pod/Engine 重启或
健康检查失败都会使当前 run fail-closed。
