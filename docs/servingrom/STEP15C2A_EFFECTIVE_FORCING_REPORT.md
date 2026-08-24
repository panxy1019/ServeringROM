# ServingROM Step 15C-2A 实际有效强迫项诊断

## 结论

- `actual_forcing_hypothesis_pass=false`
- `effective_forcing_dynamics_ready=false`
- `control_rom_ready=false`
- validation 冻结候选：`actual_forcing_only`，ridge=`100.0`。
- representation 固定为 `gc12-diff2`；global/common dynamics 与 5s Slow KPI Head 均保持冻结。
- 未启动 1P2D、未重新采集、未读取 Round 14.3、未实现 MPC。

## Validation Ablation

| candidate | ridge | running rollout | waiting rollout | remaining rollout | diff POD | global | slow KPI | radius |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| command_only | 100.0 | 0.962863 | 1.000556 | 0.961954 | 0.984061 | 0.846309 | 0.717615 | 0.938417 |
| actual_forcing_only | 100.0 | 0.756434 | 0.951870 | 0.748185 | 0.870470 | 0.846467 | 0.677808 | 0.938018 |
| actual_forcing_plus_command | 100.0 | 0.757985 | 0.951883 | 0.749641 | 0.871055 | 0.846475 | 0.678169 | 0.938313 |

## Frozen Test

- running one-step / rollout：`0.336885` / `0.745119`
- waiting one-step / rollout：`0.954132` / `0.952105`
- remaining one-step / rollout：`0.314050` / `0.736522`
- differential POD rollout：`0.869099`
- global/common rollout：`0.848763`
- Slow KPI regression：`0.674577`
- spectral radius：`0.938018`

## 四个问题

1. 真实 routed forcing 是否显著改善：**是，但未通过严格 readiness 门**。Validation running/remaining 分别相对改善 `21.44%` / `22.22%`；remaining=`0.748185` 通过 `<0.75`，running=`0.756434` 未通过。
2. `f_req` 与 `f_tok` 的主要贡献者：**routed_request_imbalance**。
3. forcing 已知后 `rho_A` 是否仍有实质信息：**否**。
4. 下一步：**explicit conservation/service-outflow model**。

## Oracle 边界

真实未来 routed forcing 在在线 counterfactual/MPC 中不可直接获得。即使本轮通过，也只能证明 effective-forcing dynamics 成立，不能把 oracle forcing 模型标记为可部署 Control-ROM。
