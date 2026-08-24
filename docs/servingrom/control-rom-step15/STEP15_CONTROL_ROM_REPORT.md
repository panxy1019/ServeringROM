# ServingROM Step 15 Control-aware ROM Identification

## 结论

- `control_rom_ready=false`
- 最终模型：`linear`，POD rank=16，ridge=100.0
- 增广谱半径：`0.927712`
- 只使用冻结的 `servingrom-control-dataset-v1`；Round 14.3 held-out 数据未读取。
- train/validation 用于拟合和选择；test 仅在模型冻结后执行一次最终评估。
- rank=16 控制状态重构充分性：`False`。

## Fast State 指标

| split | one-step NRMSE | rollout NRMSE | persistence NRMSE | rollout skill |
|---|---:|---:|---:|---:|
| train | 0.690960 | 0.754392 | 1.056736 | 0.286111 |
| validation | 0.698754 | 0.723027 | 1.066384 | 0.321983 |
| test | 0.710884 | 0.738576 | 1.061345 | 0.304114 |

## POD 控制状态重构

validation NRMSE 小于 1 才表示重构优于 train-mean 基线；该门只使用 validation，不参与 test 后调参。

| state | rank 16 validation NRMSE | selected-rank validation NRMSE |
|---|---:|---:|
| waiting_imbalance | 0.058557 | 0.058557 |
| running_imbalance | 1.001375 | 1.001375 |
| remaining_token_imbalance | 1.275442 | 1.275442 |

## 控制方向

以下反事实比较保持状态和扰动不变，只将 `rho_A` 从 0.4 提高到 0.6；正号表示模型预测 A-B 状态随控制方向正确增加。

| state | positive fraction | median delta |
|---|---:|---:|
| waiting_imbalance | 1.000000 | 0.003555 |
| running_imbalance | 1.000000 | 0.001319 |
| remaining_token_imbalance | 1.000000 | 0.113079 |

## Slow KPI Head

- Slow Head ridge：`1.0`
- validation aggregate NRMSE：`0.561347`
- test aggregate NRMSE：`0.547405`
- TTFT/TPOT 使用 5 秒窗口内守恒总量（mean × completed requests）建模，避免无 completion 窗口的 null 均值被伪造成观测值。

## Bilinear 候选

- validation rollout 相对改善：`0.000334`
- 是否保留：`False`

## Readiness Gates

- `spectral_radius`: `True`
- `validation_rollout_skill_positive`: `True`
- `test_rollout_skill_positive`: `True`
- `validation_slow_kpi_beats_mean`: `True`
- `test_slow_kpi_beats_mean`: `True`
- `control_direction`: `True`
- `pod_control_imbalance_reconstruction`: `False`
- `all_metrics_finite`: `True`

## 边界

本轮未读取 held-out actuator benchmark，未重新调参，未实现 actuator 或 MPC。若 readiness 为 false，应停在 Step 15 分析 POD 对控制状态的表达能力或动力学输出头的结构性缺口。
