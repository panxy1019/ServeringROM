# Step 9 ROM 数据审计与状态预处理

- 数据集 manifest：`e5bc4802c232b53b549c4a1eb003837af378d726af05d98e3da65674d668cf1e`
- Run 隔离：`{"train": 24, "validation": 24, "test": 24, "test/transient": 12}`
- X 恒零/近常量维度：`1360/1804`
- X 有效维度：`444`
- MU 全 Dataset 恒定：`True`
- 拟合策略：只在 train 上执行 log1p、均值/方差估计和常量维检测。
- 尺度策略：各维 z-score 后按物理 block 的有效维数平方根进行平衡，防止 bytes/token mass 和高维 histogram block 支配 POD。
- validation、test、test/transient 仅复用冻结 normalizer，不参与任何拟合。
- Dataset v1 未修改；缺失的 D/Y/MU 索引已作为建模 provenance 单独保存。
- Step 9 结构门：`PASS`
- 变化的 MU 维度：`[]`
- 受 TTFT SLO 定义影响的 X 维度：`312`
- 受 TTFT SLO 定义影响的 Y 维度：`['goodput_request_count', 'goodput_output_tokens', 'ttft_slo_violation_count']`
