# ServingROM Round 14.1 控制激励 Pilot 方法

## 1. 研究边界

本轮只回答两个问题：`rho_A` 是否形成持续激励，以及它是否对两路 Decode 的内部状态产生方向正确、统计可区分的作用。它不训练 Control-ROM，不实现 MPC，也不把实际路由比例当成控制输入。

唯一控制量为：

```text
u_k = rho_A
rho_B = 1 - rho_A
safe range = [0.2, 0.8]
minimum dwell = 5 s
maximum single-step delta = 0.2
```

`U` 只能由 `actuator_applied.payload.effective_value` 重建，并按命令实际生效时间前向保持到 200 ms fast window。`actual_request_ratio`、`actual_token_ratio` 是 actuator 经请求到达和 token workload 调制后的输出，不能替代 `U`。

## 2. 固定运行环境

- 同一个 Pod UID 完成全部 12 run；
- Prefill TP2、Decode A TP2、Decode B TP2；
- Mooncake KV transfer；
- Decode `FULL_DECODE_ONLY`；
- async scheduling enabled；
- 每个 run 为 120 s warmup、600 s excitation、inventory drain；
- run 切换仅轮转旁路遥测 writer，不重启 Engine、不加载模型。

## 3. 实验矩阵

工作点：

| Workload | 55% capacity | 85% capacity |
|---|---:|---:|
| balanced | 0.275 req/s | 0.425 req/s |
| mixed-bimodal | 0.3575 req/s | 0.5525 req/s |

每个工作点分别运行 PRBS、multilevel random-dwell 和合法 step sequence，共 12 run。PRBS 的 `0.3↔0.7` 变化通过 `0.5` 桥接，避免违反单步变化上限。

## 4. 时间坐标与观测量

fast window 为 200 ms，保存 `X/X_next/D/U`，并增加：

```text
U_prev
delta_U
time_since_control_change
control_command_id
control_generation
```

每 25 个 fast windows 聚合成一个 5 s slow KPI window。控制响应重点分析 A-B 差分：

```text
decode running request imbalance
decode waiting request imbalance
decode expected remaining token imbalance
new-route request imbalance
new-route expected token imbalance
```

吞吐、goodput、TTFT、TPOT、KV 与 scheduler 输出用于解释工作点差异，不作为 `U`。

## 5. 响应辨识

对每个 run 计算：

1. `corr(U_t, state_{t+lag})`，lag 从 0 到 60 s；
2. `rho_A=0.7` 与 `rho_A=0.3` 下状态均值差和 Cohen's d；
3. 每次合法 step 的 20% 响应延迟；
4. 进入最终平台值 ±20% 且保持至少两个 slow windows 的 settling time；
5. 55% 与 85% load 下 effect size 和 lag 的变化。

正方向定义为：提高 `rho_A` 后，A 相对 B 的新请求、token workload、running/waiting 或 expected remaining inventory 增加。KPI 可能因排队与完成事件产生更长滞后，不能用零时延相关替代动态判断。

## 6. 预注册门槛

`persistent_excitation_pass=true` 要求全部 12 run：

- 5 s 序列包含 `0.3/0.5/0.7`；
- `var(U) >= 0.005`；
- 所有命令来自 `actuator_applied`；
- 单步变化、dwell、generation 和 command ID 完整合法。

`control_authority_pass=true` 要求：

- 对 running、waiting、expected remaining 三个直接 Decode 状态，至少一个满足 `high-low > 0` 且 Cohen's d ≥ 0.25；
- 至少 8/12 run 满足上一条；
- 每个 workload/load 工作点的 3 种 excitation 中至少 2 种满足；
- 所有 run 同时通过遥测、守恒、writer、Pod 和 Engine 稳定性质量门。

该门槛在正式结果产生前固定。若控制作用过弱，结论必须为 FAIL，并停在 Round 14.1 分析原因。

## 7. 失败与封存语义

任一 run 失败时，campaign 先幂等回滚 baseline，再 deactivate writer，并停止后续 run。每个成功 run 在 Control 派生产物和质量报告生成后才统一创建 SHA256 manifest。中止或未封存目录保留原样，但不得进入 pilot manifest 的 SEALED 集合。
