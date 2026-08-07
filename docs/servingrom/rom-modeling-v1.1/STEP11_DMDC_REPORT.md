# Step 11 DMDc / Reduced Dynamics Identification

- 模型：`z[k+1] = A z[k] + E d[k] + c`，`y[k] = C z[k] + F d[k] + b`。
- MU 已确认固定，不作为控制输入；Dataset v1 不包含 u[k]。
- Ridge 扫描：`[1e-06, 0.0001, 0.01, 1.0, 10.0]`；超参数选择只使用 validation。
- 选定 rank：`16`
- 选定 ridge：`10.0`
- A 谱半径：`0.93938681`
- Validation one-step state NRMSE：`0.710876`
- Validation rollout state NRMSE：`0.741451`
