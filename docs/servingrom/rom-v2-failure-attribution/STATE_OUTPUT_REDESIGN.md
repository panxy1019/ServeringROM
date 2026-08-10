# ServingROM v2 State / Output Redesign

- 快速状态时钟：`200 ms`；POD rank `16`；短期 Markov 记忆：`True`。
- 慢速 KPI 输出时钟：`5000 ms`，每个输出窗口覆盖 `25` 个状态步。
- stock/flow 分离：X 在快速时钟上保持边界库存；D 保留 200 ms 到达顺序；Y 在慢速窗口内守恒求和。
- 任意聚合、差分和历史特征都在单个 run 内构造，严禁跨 run 泄漏。
- 输出同时保留可对账的原始计数，并派生 throughput、goodput ratio、条件 TTFT/TPOT 与 violation rate。
- 条件指标使用显式 validity mask，不再把无完成请求的窗口伪装成 0 ms 延迟。
- 5 秒单速率状态模型虽稳定但 state NRMSE 接近 1，是均值回归，不作为动力学模型。
- v2 仍然没有 u[k]；队列、scheduler 输出和固定 MU 不会被伪装成 actuator。
- 详细机器可读 schema：`design/state_v2.json` 与 `design/output_v2.json`。
