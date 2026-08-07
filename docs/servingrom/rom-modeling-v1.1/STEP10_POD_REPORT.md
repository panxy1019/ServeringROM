# Step 10 POD 状态空间降阶

- Train 有效协方差维度：`444`
- 候选 ranks：`[16, 32, 48, 64, 96, 128, 192]`
- Rank 扫描重构误差：
  - r=16: train=0.734297, validation=0.718154, test=0.735208, test/transient=0.716620
  - r=32: train=0.626152, validation=0.620816, test=0.629051, test/transient=0.625359
  - r=48: train=0.540797, validation=0.532197, test=0.547907, test/transient=0.556645
  - r=64: train=0.476321, validation=0.468699, test=0.484520, test/transient=0.488704
  - r=96: train=0.356184, validation=0.351208, test=0.377086, test/transient=0.388736
  - r=128: train=0.275065, validation=0.278937, test=0.313115, test/transient=0.305481
  - r=192: train=0.035906, validation=0.057060, test=0.157541, test/transient=0.038394
- Rank 不按单一累计能量冻结；全部候选进入线性动力学识别。
- 完整谱位于 `pod/spectrum.csv`，物理块模态贡献位于 `pod/mode_block_contributions.json`。
