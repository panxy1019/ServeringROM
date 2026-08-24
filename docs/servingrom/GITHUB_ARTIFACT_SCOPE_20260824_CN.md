# ServingROM GitHub 发布范围与资产边界

## 1. 发布目标

本仓库推送到 `https://github.com/panxy1019/ServeringROM`，用于保存可复现的工程实现和可公开审计的实验结论。GitHub 不是原始遥测、模型权重或生产数据的备份介质。

## 2. 纳入 GitHub 的内容

| 类别 | 主要目录 | 内容 |
|---|---|---|
| 运行代码 | `scripts/`、`servingrom_*` | Proxy、telemetry、Snapshot、数据集构建、控制器和 ROM 建模 |
| 测试 | `tests/`、`servingrom_telemetry/tests/` | 单元测试、静态 patch 检查和 pipeline 合同测试 |
| 部署 | `k8s/`、`ops/` | 1P2D、D2、Control-v1 的 Deployment 与生命周期脚本 |
| 镜像 | `docker/`、`Dockerfile` | ServingROM 镜像构建入口及 provenance 约束 |
| 补丁 | `patches/` | vLLM、vLLM-Ascend、Mooncake 旁路遥测补丁 |
| 配置 | `configs/` | 冻结 baseline、workload、Control Dataset 和 ROM 参数 |
| 文档 | `docs/servingrom/`、`pd/markdowns/` | 架构、Schema、审计、实验报告和当前状态 |
| 小型证据 | 聚合 JSON、manifest | 不含请求正文、原始 telemetry 和密钥的统计与哈希 |

## 3. 明确排除的内容

以下内容由 `.gitignore` 排除，不推送到 GitHub：

```text
results/
raw/ 和任意 **/raw/
.campaign/
diagnostics/runs/
models/、checkpoints/
*.safetensors、*.onnx、*.pt、*.pth、*.bin
*.npy、*.npz
*.7z、*.zip、*.tar、*.tgz
.env、私钥、credentials、Secret YAML
Docker/Kubernetes 本地认证目录
cache、core dump、日志和 PID 文件
```

本地工作目录约为 2.1 GiB，其中大部分是压力测试 JSONL；Git 对象库仅约 3.9 MiB。推送前检查确认受版本控制历史中不存在超过 GitHub 限制的大 blob，最大已跟踪 blob 约 658 KiB。

## 4. 外部封存资产

以下资产保留在 `server-00`，不会复制到公开仓库：

| 资产 | 路径 | 状态 |
|---|---|---|
| ROM Dataset v1.1 | `/home/admin/servingrom-results/datasets/servingrom-qwen36-1p2d-d2-rom-v1.1-slo2000/` | SEALED，只读派生 |
| Control Dataset v1 | `/home/admin/servingrom-results/datasets/servingrom-control-dataset-v1/` | SEALED、immutable |
| Step 15 模型 | `/home/admin/servingrom-results/models/servingrom-control-rom-v1/` | 诊断产物，不可部署 |
| Step 15B 模型 | `/home/admin/servingrom-results/models/servingrom-control-redesign-v1/` | 表示冻结，动力学未通过 |
| Step 15C-1 模型 | `/home/admin/servingrom-results/models/servingrom-control-memory-v1/` | 失败诊断产物 |
| Held-out campaign | `/home/admin/testpanxy/servingrom-control-dataset-v1-code/.campaign/servingrom-control-heldout-v1/` | 1 FAILED、9 PENDING、未封存 |

GitHub 中保存这些资产的 schema、报告、关键 manifest 和 SHA256 provenance，而非底层数组和 raw telemetry。

## 5. 安全审计

推送前执行了以下检查：

1. 工作树状态和历史 blob 大小检查；
2. 已跟踪文件中的密码、GitHub token、AWS access key 和私钥头扫描；
3. 模型权重、压缩包、raw/results 跟踪状态检查；
4. `.gitignore` 命中验证；
5. 远端仓库所有权和 GitHub CLI 登录身份核验。

当前发布范围没有包含已知密码、访问 token、私钥、模型权重或原始 telemetry。集群地址、Deployment 名称和本地封存路径保留在技术报告中，作为实验 provenance。
