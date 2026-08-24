# ServingROM Step 15C-2B 对称保持差分守恒与服务闭合 ROM

## 最终状态

- `conservation_dynamics_ready=false`
- `control_rom_ready=false`
- validation 冻结候选：`M2_load_dependent_service_closure`，ridge=`0.0001`。
- 12D global/common、原始 gc12-diff2 hidden reference 与 5s Slow KPI Head 均保持冻结。
- 未启动 1P2D、未重新采集、未读取 Round 14.3、未实现 MPC。

## Outflow / Transition Audit

| 守恒关系 | validation residual NRMSE | exact fraction | 结论 |
|---|---:|---:|---|
| `Δq_run = first_decode_schedule - terminal` | 0.000000 | 100.00% | 精确闭合 |
| `Δq_remaining = routed_token_mass - emission - terminal_residual` | 1.446158 | 76.59% | 不闭合 |
| `Δq_remaining = arrival_token_mass - emission - terminal_residual` | 0.006737 | 91.24% | 近乎精确闭合 |

当前 snapshot 的 `decode_d1/d2_expected_remaining_tokens` 会在请求仍处于 Prefill 时，按该 attempt 后续选中的 decoder 提前计入库存。因此它的物理流入边界是 request arrival，不是 `p_to_d_route`。这解释了 routed-token 守恒残差，而不是 token emission 大规模缺失。

## Validation Ablation

| 候选 | ridge | running rollout | remaining rollout | radius max | symmetry | zero bias |
|---|---:|---:|---:|---:|---:|---:|
| M0_inflow_only | 100.0 | 5.711737 | 1.569005 | 1.000000 | True | True |
| M1_minimal_service_closure | 0.0001 | 0.944094 | 1.078121 | 0.904403 | True | True |
| M2_load_dependent_service_closure | 0.0001 | 0.939032 | 1.056614 | 0.990823 | True | True |

## 消融解释

- M1 相比 M0 的平均 rollout 相对改善：`72.23%`。`-Hq` 能抑制 inflow-only 积分漂移。
- M2 相比 M1 的平均 rollout 相对改善：`1.31%`。幅度不足以证明必须加入 total-load correction。
- 最终候选相比 Step 15C-2A：running `-24.14%`，remaining `-41.22%`；负值代表退化。
- M2 虽按 validation 平均 NRMSE 被选中，但没有通过 readiness gate，也不应被视为可部署模型。

## 幅度与相位诊断

- running：median amplitude ratio=`0.3430`，median best lag=`0.10s`，median correlation=`0.3655`。
- remaining：median amplitude ratio=`0.2247`，median best lag=`1.60s`，median correlation=`-0.0028`。
- 由于 200 ms routed forcing 是稀疏随机脉冲而非隔离阶跃，本轮不伪造单一 settling time；低 amplitude ratio 和低 remaining correlation 已表明长期库存响应被严重低估。

## Frozen Test

- running one-step / rollout：`0.327608` / `0.937395`
- remaining one-step / rollout：`0.347746` / `1.076162`
- waiting diagnostic rollout：`0.999227`
- global/common frozen rollout：`0.848496`
- Slow KPI frozen-head regression：`0.717181`

## 五个问题

1. 剩余误差是否主要来自 service/outflow closure：**否；restoring closure 可阻止 M0 积分漂移，但未击败 Step15C-2A，主要残差更符合 arrival/route 状态语义错位与 transition timing**。
2. `-Hq` 是否显著改善长期 rollout：**是**。
3. 是否需要 total-load correction：**否**。
4. telemetry 是否足够构造真实 A/B outflow：**部分：request closure 精确，arrival-aligned token closure 近乎精确；routed-token 与当前 q_remaining 时间语义不一致**。
5. oracle plant-side 是否具有进入 forcing surrogate 的余量：**否**。

## 下一步边界

- `stop: redesign Decode inventory boundary and route-to-admission transition state`
- 应把 Decode-only remaining inventory 从 Prefill/route 前库存中拆开，并显式建模 route→KV-ready→Decode admission transition；在此之前不进入 forcing surrogate。

## Oracle 边界

本轮 actual forcing 和真实 outflow 只用于 plant-side 诊断。未来 counterfactual/MPC 无法直接访问未来 routed forcing，因此无论本轮是否通过，`control_rom_ready` 均保持 false。
