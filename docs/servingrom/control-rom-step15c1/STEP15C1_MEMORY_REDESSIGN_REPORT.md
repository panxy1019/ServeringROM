# ServingROM Step 15C-1 Control-Dynamics Memory Redesign

## 结论

- `memory_dynamics_ready=false`
- `effective_memory_horizon=None`
- 诊断候选 horizon：`1.0s`
- `effective_forcing_available=true`
- 冻结候选：`exponential`，horizon=1.0s，ridge=10.0。
- representation 固定为 `gc12-diff2`；未读取 held-out、未启动 1P2D、未实现 MPC。
- 未达到 strong gate，因此该候选仅作为失败诊断产物，不是可部署 Control-ROM。

## Validation Ablation

| kind | horizon(s) | dim | running | waiting | remaining | global | slow KPI | radius |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0 | 0 | 0.962735 | 1.000577 | 0.961819 | 0.846307 | 0.379267 | 0.938323 |
| raw_lag | 1.0 | 30 | 0.962524 | 1.001126 | 0.961573 | 0.836290 | 0.360277 | 0.938715 |
| multi_scale | 1.0 | 18 | 0.962614 | 1.000960 | 0.961672 | 0.837136 | 0.358428 | 0.938575 |
| exponential | 1.0 | 6 | 0.963822 | 1.000994 | 0.962893 | 0.836401 | 0.359037 | 0.947827 |
| raw_lag | 2.0 | 60 | 0.964484 | 1.001831 | 0.963422 | 0.827425 | 0.358307 | 0.936855 |
| multi_scale | 2.0 | 36 | 0.964638 | 1.001393 | 0.963592 | 0.828660 | 0.355368 | 0.936183 |
| exponential | 2.0 | 12 | 0.963003 | 1.001331 | 0.962050 | 0.832208 | 0.358458 | 0.952432 |
| raw_lag | 5.0 | 150 | 0.961484 | 1.002815 | 0.960523 | 0.823052 | 0.366926 | 0.936485 |
| multi_scale | 5.0 | 54 | 0.961868 | 1.001725 | 0.960909 | 0.824844 | 0.355736 | 0.934775 |
| exponential | 5.0 | 18 | 0.964153 | 1.001600 | 0.963285 | 0.833330 | 0.359054 | 0.961062 |
| raw_lag | 10.0 | 300 | 0.965735 | 1.003696 | 0.964653 | 0.822543 | 0.370679 | 0.954023 |
| multi_scale | 10.0 | 72 | 0.966241 | 1.001804 | 0.965183 | 0.825139 | 0.347525 | 0.954160 |
| exponential | 10.0 | 24 | 0.964757 | 1.001637 | 0.963819 | 0.834975 | 0.358570 | 0.979469 |
| raw_lag | 20.0 | 600 | 0.967075 | 1.005297 | 0.965943 | 0.823937 | 0.373931 | 0.972064 |
| multi_scale | 20.0 | 90 | 0.966430 | 1.002079 | 0.965273 | 0.825180 | 0.349219 | 0.966256 |
| exponential | 20.0 | 30 | 0.963830 | 1.001702 | 0.962798 | 0.831813 | 0.359486 | 0.990595 |

## Test

- running rollout NRMSE：`0.967794`
- waiting rollout NRMSE：`0.999516`
- remaining-token rollout NRMSE：`0.967122`
- one-step state NRMSE：`0.681088`
- global rollout NRMSE：`0.843960`
- Slow KPI NRMSE：`0.358169`
- control-direction pass fraction：`0.666667`
- augmented spectral radius：`0.947827`

## 失败归因

- 一步预测约为 0.4，但自由 rollout 仍约为 0.97，误差主要在递推中持续累积。
- 1--20s memory 对 running/remaining-token imbalance 的收益很小，有限记忆不是当前主导瓶颈。
- global state 与 Slow KPI 有一定改善，说明历史对总负载有信息，但没有恢复路由产生的 A/B 有效注入。
- 下一步应使用已经存在的 routed request/token-mass imbalance 做显式 forcing；本轮按约束没有进入 Step 15C-2。

## Effective Forcing Audit

- 来源：`sealed run derived/control/control_windows.parquet`
- 对齐：200 ms half-open [start_wall_ns,end_wall_ns); route event selected by ts_wall_ns
- 本轮只准备 Step 15C-2 schema，没有把 forcing 加入 15C-1。
