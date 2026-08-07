# Step 12 多步 Rollout 与 Held-out Transient 验证

- Validation state/output NRMSE：`0.741451` / `0.795843`
- Test state/output NRMSE：`0.725892` / `0.802494`
- Transient state/output NRMSE：`0.672144` / `0.827855`
- Transient pattern：`{"step-up": {"runs": 3, "state_nrmse_mean": 0.6520008890818229, "output_nrmse_mean": 0.8029450429091272, "all_finite": true}, "step-down": {"runs": 3, "state_nrmse_mean": 0.655184067442606, "output_nrmse_mean": 0.8188768460849668, "all_finite": true}, "ramp-up": {"runs": 3, "state_nrmse_mean": 0.6411772422330523, "output_nrmse_mean": 0.7996188485880557, "all_finite": true}, "held-out-composite": {"runs": 3, "state_nrmse_mean": 0.647900576368348, "output_nrmse_mean": 0.8295030659721355, "all_finite": true}}`
- 不可观测关键输出：`[]`
- 关键输出 NRMSE 超限：`{"validation": {"completed_output_tokens": 0.9651557975800188, "goodput_output_tokens": 0.9673955910605095, "ttft_sum_ms": 0.9651751068164552, "tpot_sum_ms": 0.9627244536102436}, "test": {"completed_output_tokens": 0.96469787981918, "goodput_output_tokens": 0.9691882113324589, "ttft_sum_ms": 0.9632232418365029, "tpot_sum_ms": 0.9623823470900902}, "test/transient": {"completed_output_tokens": 0.9784684061623666, "goodput_output_tokens": 0.9807585147539765, "ttft_sum_ms": 0.976876121690858, "tpot_sum_ms": 0.9760574186451026}}`
- 所有指标按完整 held-out run 自由 rollout 计算，未随机打散窗口。
