---
pretty_name: ServingROM 1P2D Full-order and Control Datasets
language:
- en
- zh
tags:
- systems
- serving
- reduced-order-modeling
- control
- vllm
- ascend
- performance
size_categories:
- 100K<n<1M
---

# ServingROM 数据集

该仓库封存 ServingROM 第一代 POC 的两个正式数据集，用于研究 1P2D 大模型推理系统的全阶状态重建、POD/DMDc 降阶建模、运行时路由控制和 Control-aware ROM。

实验系统为 Qwen3.6-27B-w8a8，拓扑为 Prefill TP2 + Decode A TP2 + Decode B TP2，通过 Mooncake 传输 KV。Decode 使用 `FULL_DECODE_ONLY` 图模式和 async scheduling。数据不包含 prompt、对话正文、模型输出正文、模型权重或原始逐事件 telemetry。

配套代码、部署文件、Schema 与实验报告位于 [panxy1019/ServeringROM](https://github.com/panxy1019/ServeringROM)。

## 目录

```text
rom-dataset-v1.1/
    train/{X,X_next,D,Y,MU}.npy
    validation/{X,X_next,D,Y,MU}.npy
    test/{X,X_next,D,Y,MU}.npy
    test/transient/{X,X_next,D,Y,MU}.npy
    dataset_manifest.json
    run_manifest.parquet
    *_index.json
    snapshot_schema.json
    canonicalization_report.json

control-dataset-v1/
    train/{X,X_next,D,U,U_aux}.npy
    validation/{X,X_next,D,U,U_aux}.npy
    test/{X,X_next,D,U,U_aux}.npy
    run_index.parquet
    slow_kpi_windows.parquet
    dataset_manifest.json
    quality_summary.json
    SHA256SUMS.json
```

## ROM Dataset v1.1

路径：`rom-dataset-v1.1/`

这是从已封存 raw telemetry 只读重建的固定配置数据集。v1.1 统一了 TTFT SLO 和 goodput 语义，没有重新采集，也没有修改源 Dataset v1。

| Split | Windows | X/X_next | D | Y | MU | dtype |
|---|---:|---:|---:|---:|---:|---|
| train | 57,600 | 1,804 | 31 | 19 | 12 | float32 |
| validation | 57,600 | 1,804 | 31 | 19 | 12 | float32 |
| test | 57,600 | 1,804 | 31 | 19 | 12 | float32 |
| test/transient | 28,800 | 1,804 | 31 | 19 | 12 | float32 |

每个 fast window 为 200 ms：

- `X[k]`：窗口边界的全阶系统状态；
- `X_next[k]`：下一窗口边界状态；
- `D[k]`：31 维外部 workload disturbance；
- `Y[k]`：19 维守恒输出；
- `MU[k]`：12 维 run 级固定系统配置。

完整 Dataset v1.1 中 `MU` 实际恒定，不是运行时控制输入。不得把 scheduler 输出、queue length 或 `MU` 伪装成 `u[k]`。

## Control Dataset v1

路径：`control-dataset-v1/`

正式矩阵为 `2 workloads × 3 loads × 2 arrival processes × 3 split seeds = 36 runs`。所有 run 均通过请求/阶段库存、KV 生命周期、writer、event sequence、Pod/Engine restart、OOM 和 engine-death 质量门。

| Split | Runs | Fast windows | X/X_next | D | U | U_aux | dtype |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 12 | 36,000 | 1,804 | 31 | 1 | 4 | float64 |
| validation | 12 | 36,000 | 1,804 | 31 | 1 | 4 | float64 |
| test | 12 | 36,000 | 1,804 | 31 | 1 | 4 | float64 |

唯一真实 actuator 为：

```text
U[1] = rho_A
rho_B = 1-rho_A
safe range = [0.2, 0.8]
minimum dwell = 5s
maximum single-step delta = 0.2
```

`U` 只来自 `actuator_applied.effective_value`。actual request ratio、actual token ratio、running/waiting 和 remaining-token imbalance 是系统响应或诊断量，不是控制输入。

训练、验证和测试使用完整 run 隔离：seed 101 为 train，202 为 validation，303 为 test。禁止随机打散窗口后重新划分 split，也禁止跨 run 构造 `X_next`、历史或 rolling feature。

## NumPy 读取示例

```python
from pathlib import Path
import json
import numpy as np

root = Path("control-dataset-v1")
split = "train"

X = np.load(root / split / "X.npy", mmap_mode="r")
X_next = np.load(root / split / "X_next.npy", mmap_mode="r")
D = np.load(root / split / "D.npy", mmap_mode="r")
U = np.load(root / split / "U.npy", mmap_mode="r")

state_index = json.loads((root / "state_index.json").read_text())

print(X.shape, X_next.shape, D.shape, U.shape)
```

大数组建议始终使用 `mmap_mode="r"`，避免一次性加载全部状态矩阵。

## 数据质量与建模边界

- 所有 normalization、常量维检测和 POD basis 必须只使用 train split；
- validation 用于选择 rank、ridge 和模型结构；
- test 只能在模型冻结后访问；
- `test/transient` 用于完整 held-out transient rollout，不参与调参；
- Control Dataset v1 没有包含 Round 14.3 held-out actuator benchmark；
- 当前 `control_rom_ready=false`、`mpc_ready=false`，数据集可用于研究，但不能被解释为已经验证可部署的 MPC 模型。

## 隐私与内容边界

数据集由系统状态、事件计数、延迟聚合、token 数量、KV/queue inventory、workload 标签和控制量组成。发布目录不包含：

- 用户 prompt 或完整对话；
- 模型输出正文；
- 登录凭据或访问 token；
- Kubernetes Secret；
- 模型权重；
- 原始逐请求 JSONL telemetry。

## 校验与溯源

每个数据集目录包含 manifest、Schema 和索引。Control Dataset v1 另有 `SHA256SUMS.json`。下载后应先核对这些文件，再进行建模。GitHub 仓库中的实验报告记录了数据构建、质量门、POD/DMDc、Control-v1 和 Step 15 系列诊断。

## 许可与使用

本数据集当前未声明标准开源许可证。公开发布不自动授予模型权重、基础模型或第三方软件的再分发权。使用者应遵守 Hugging Face 条款、关联软件许可证以及所在机构的数据与安全政策。
