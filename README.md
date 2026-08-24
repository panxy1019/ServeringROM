# ServingROM

ServingROM 是面向 1P2D 大模型推理系统的可观测性、全阶状态重建、降阶建模和运行时控制实验工程。当前实验平台为 Qwen3.6-27B-w8a8、Prefill TP2、Decode A TP2、Decode B TP2，并通过 Mooncake 完成 KV 传输。

项目已经完成统一遥测、Proxy/Engine/Mooncake 生命周期关联、200 ms Full-order Snapshot、固定配置 ROM、Control-v1 路由执行器、正式 Control Dataset v1，以及 Control-aware ROM 的表示与记忆诊断。当前尚未进入 MPC；下一建模步骤是将实际 routed request/token-mass imbalance 作为显式 effective forcing 引入 Step 15C-2。

## 阅读入口

- [当前完整进展与实验效果](docs/servingrom/CURRENT_PROGRESS_AND_RESULTS_20260824_CN.md)
- [GitHub 发布范围与外部数据资产](docs/servingrom/GITHUB_ARTIFACT_SCOPE_20260824_CN.md)
- [Hugging Face 数据卡](docs/servingrom/HUGGINGFACE_DATASET_CARD.md)
- [Hugging Face 上传清单](docs/servingrom/HUGGINGFACE_UPLOAD_MANIFEST.json)
- [遥测只读审计](docs/servingrom/TELEMETRY_AUDIT.md)
- [Full-order Snapshot Schema](docs/servingrom/FULL_ORDER_SNAPSHOT_SCHEMA.md)
- [Mooncake 最小生命周期报告](docs/servingrom/MOONCAKE_MINIMAL_LIFECYCLE_REPORT.md)
- [Control-v1 Runtime Smoke](docs/servingrom/control/RUNTIME_SMOKE_REPORT.md)
- [Round 14.1 控制激励 Pilot](docs/servingrom/control/CONTROL_EXCITATION_PILOT_REPORT.md)
- [Step 15 Control-aware ROM](docs/servingrom/control-rom-step15/STEP15_CONTROL_ROM_REPORT.md)
- [Step 15B 控制相关状态重设计](docs/servingrom/control-rom-step15b/STEP15B_CONTROL_RELEVANT_REDESIGN_REPORT.md)
- [Step 15C-1 有限记忆诊断](docs/servingrom/control-rom-step15c1/STEP15C1_MEMORY_REDESSIGN_REPORT.md)

## 仓库边界

GitHub 保存代码、测试、配置、Kubernetes YAML、Dockerfile、补丁、Schema、聚合报告和小型 manifest。以下内容不进入 Git：模型权重、原始 telemetry、Parquet/NumPy 数据、完整实验结果、凭据、Kubernetes Secret、构建缓存和 core dump。

正式数据集与模型产物在 `server-00` 上独立封存，并通过报告中的路径和 SHA256 manifest 关联。克隆本仓库不会自动获得这些大体量或受控数据资产。
