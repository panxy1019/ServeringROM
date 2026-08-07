# Step 13 Actuator / MPC 准入结论

- 是否允许进入 actuator excitation 与 MPC：`False`
- 自动门：`{"spectral_radius": true, "validation_rollout_finite": true, "test_rollout_finite": true, "transient_rollout_finite": true, "validation_state_nrmse": true, "test_state_nrmse": true, "transient_state_nrmse": true, "validation_output_nrmse": true, "test_output_nrmse": true, "transient_output_nrmse": true, "validation_state_skill": true, "test_state_skill": true, "transient_state_skill": true, "validation_output_skill": true, "test_output_skill": true, "transient_output_skill": true, "key_outputs_observable": true, "key_output_rollout_accuracy": false}`
- 门限：`{"spectral_radius_max": 1.001, "state_nrmse_max": 0.8, "output_nrmse_max": 0.9, "state_skill_min": 0.1, "output_skill_min": 0.1, "key_output_nrmse_max": 0.9}`
- 缺失关键输出：`[]`
- 关键输出 NRMSE 超限：`{"validation": {"completed_output_tokens": 0.9651557975800188, "goodput_output_tokens": 0.9673955910605095, "ttft_sum_ms": 0.9651751068164552, "tpot_sum_ms": 0.9627244536102436}, "test": {"completed_output_tokens": 0.96469787981918, "goodput_output_tokens": 0.9691882113324589, "ttft_sum_ms": 0.9632232418365029, "tpot_sum_ms": 0.9623823470900902}, "test/transient": {"completed_output_tokens": 0.9784684061623666, "goodput_output_tokens": 0.9807585147539765, "ttft_sum_ms": 0.976876121690858, "tpot_sum_ms": 0.9760574186451026}}`
- 当前 Dataset v1 中 MU 为固定配置，且不存在正式运行时 actuator；不会把 scheduler 输出、queue 或 MU 伪装成 u[k]。
- 只有全部门通过，后续才设计可热更新 token budget、max-num-seqs 或路由比例的独立 excitation 数据集。
