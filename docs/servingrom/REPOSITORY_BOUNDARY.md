# ServingROM POC 仓库边界

审计时间：2026-08-05 UTC

## 纳入版本控制

- `scripts/`：Proxy、Worker entrypoint、实验和验证脚本；
- `k8s/`：当前 1P2D Deployment/Service 定义；
- `decode_graph_ab/`：冻结的 D0/D1/D2 实验定义与分析代码；
- `docker/`：后续 ServingROM 镜像构建归属；
- `patches/`：vLLM、vLLM-Ascend 和 Mooncake 补丁归属；
- `tests/`：后续单元和合同测试；
- `servingrom_telemetry/`：后续遥测基础库；
- `docs/servingrom/`：审计、schema 和实验报告；
- 顶层现有 Dockerfile 与部署、启动、回退脚本。

现有路径没有搬移或重命名，避免改变脚本引用和运行语义。

## 排除清单

| 类别 | 审计发现/规则 |
|---|---|
| 实验结果 | `results/`，包含历史 benchmark、metrics 和日志 |
| 诊断原始数据 | `diagnostics/runs/` |
| 压缩归档 | `diagnostics.7z`、`*.tar`、`*.zip`、`*.7z` 等 |
| 当前大文件 | `npu-smi.log` 约 1.20 MB、`pidstat-compat.jsonl` 约 1.10 MB，均位于已排除诊断目录 |
| 模型权重 | `models/`、`checkpoints/`、`*.safetensors`、`*.bin`、`*.pt`、`*.onnx` 等 |
| 原始遥测 | 任意 `raw/` 目录 |
| 凭据 | `.env*`、私钥、证书、credential 文件、Secret YAML、Docker/Kubernetes 客户端配置 |
| 缓存 | Python、pytest、mypy、ruff、build 和 distribution cache |
| 运行残留 | 日志、PID、socket、core dump、JVM fatal log |

## 敏感信息审计结论

- 未发现符号链接；
- 未发现模型权重文件；
- 未发现 Kubernetes `Secret` 对象或 `secretKeyRef`；
- 未发现密码、token、API key、registry 凭据或私钥正文；
- 发现私有 registry 地址和内部服务地址，它们不是认证凭据，保留用于可复现实验；
- 构建脚本中的 HTTP registry 不包含用户名或密码。

提交前必须运行：

```bash
git status --short
git diff --cached --check
git ls-files | grep -E '(results/|/raw/|\.safetensors$|\.env$)' && exit 1 || true
```

