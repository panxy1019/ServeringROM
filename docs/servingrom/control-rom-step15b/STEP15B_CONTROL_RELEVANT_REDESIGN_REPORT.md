# ServingROM Step 15B Control-Relevant Reduced-State Redesign

## 结论

- `control_representation_ready=true`
- `control_rom_ready=false`
- 冻结表示：`scheme2_common_differential_block_pod` / `gc12-diff2`
- reduced state：global/common rank 12 + differential coordinates 2 = 14
- Round 14.3 held-out actuator 数据未读取；未实现 MPC。
- validation dynamics gate：`False`；诊断 dynamics 不作为可部署模型。

## 三方对照

| representation | dim | val global rollout | val running diff rollout | val remaining diff rollout | val slow KPI | radius |
|---|---:|---:|---:|---:|---:|---:|
| Step15 standard POD | 16 | 0.723027 | n/a | n/a | 0.561347 | 0.927712 |
| Scheme1 core3/r12 | 15 | 0.691604 | 0.952273 | 0.974145 | 0.429297 | 0.927226 |
| Scheme1 core3/r16 | 19 | 0.722939 | 0.952309 | 0.974177 | 0.428922 | 0.927155 |
| Scheme1 core3/r24 | 27 | 0.761680 | 0.952344 | 0.974236 | 0.424469 | 0.927223 |
| Scheme1 core3/r32 | 35 | 0.780727 | 0.952529 | 0.974538 | 0.418646 | 0.926962 |
| Scheme1 all_scalar/r12 | 17 | 0.687987 | 0.951667 | 0.974155 | 0.410987 | 0.906719 |
| Scheme1 all_scalar/r16 | 21 | 0.719778 | 0.951702 | 0.974183 | 0.411146 | 0.906422 |
| Scheme1 all_scalar/r24 | 29 | 0.759024 | 0.951729 | 0.974258 | 0.406735 | 0.906322 |
| Scheme1 all_scalar/r32 | 37 | 0.778373 | 0.951848 | 0.974479 | 0.401428 | 0.906480 |
| Scheme1 selected_binned/r12 | 30 | 0.687997 | 0.951657 | 0.974217 | 0.412544 | 0.930902 |
| Scheme1 selected_binned/r16 | 34 | 0.719784 | 0.951692 | 0.974238 | 0.412659 | 0.930873 |
| Scheme1 selected_binned/r24 | 42 | 0.758971 | 0.951744 | 0.974326 | 0.407815 | 0.930879 |
| Scheme1 selected_binned/r32 | 50 | 0.778375 | 0.951924 | 0.974614 | 0.402208 | 0.930698 |
| Scheme2 gc12-diff2 | 14 | 0.846281 | 0.953173 | 0.971608 | 0.379267 | 0.938323 |

## 表示重构与动力学的分界

- Scheme 2 validation static core NRMSE：`{"running_imbalance": 0.487537834615484, "waiting_imbalance": 0.18273635056992008, "remaining_token_imbalance": 0.6131982598609559}`
- Scheme 1 的 explicit q 坐标在表示层严格可逆，但其自由 rollout 未通过；Scheme 2 用独立 differential rank budget 在不显式携带全部 q 的情况下通过静态表示门。
- `control_representation_ready` 只陈述 reduced coordinate 对控制状态的可观测性；`control_rom_ready` 还要求动态 rollout、global degradation 和控制方向同时通过。

## Test（冻结后单次访问）

- global POD rollout NRMSE：`0.848674`
- running imbalance rollout NRMSE：`0.951877`
- waiting imbalance rollout NRMSE：`0.999304`
- remaining-token imbalance rollout NRMSE：`0.971409`
- Slow KPI NRMSE：`0.379561`
- representation-only core NRMSE：`{"running_imbalance": 0.48278433957738315, "waiting_imbalance": 0.17166431531599832, "remaining_token_imbalance": 0.5977880930933073}`

## 选择逻辑

优先选择同时满足核心 differential rollout <0.7、谱半径 <=1.01、global rollout 与 Slow KPI 相对 Step 15 退化不超过10%的最低维 Scheme 1。只有 Scheme 1 全部失败才允许运行 Scheme 2。若 dynamics 门仍失败，只冻结 validation 选择的最低维可观测 representation，不把诊断模型标为可用 Control-ROM。
