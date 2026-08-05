# Eager TP2 受控修复验证报告

验证日期：2026-08-03  
运行目录：`diagnostics/runs/eager-tp2-20260803T071416Z`  
目标模型：`/models/Qwen3.6-27B-w8a8`  
部署：`qwen36-prefill-tp2-eager-diagnostic`（已删除）

## 结论

**PASS。** 在不改动镜像、模型、CANN/torch-npu/vLLM 版本、Kubernetes 安全上下文、
allocator、设备固定注解或 TP 大小的情况下，仅将权重加载策略改为：

```text
--safetensors-load-strategy eager
```

普通 vLLM TP2 成功加载全部 9 个 safetensors 分片，并在 `13700` 提供健康服务。
这证明 `prefetch` 失败不是模型文件、HCCL、设备映射、HBM 容量或基础版本组合的问题；
它与默认 mmap-backed safetensors 加载路径强相关。`eager` 通过 `read()` 后从内存缓冲区
反序列化，避开了该路径。

该实验**没有**启动 Mooncake、Decode、PD Proxy 或完整 1P2D Deployment。完整
`ray-vllm-pd-worker-qwen36-27b` 在整个过程中始终保持 `replicas=0`。

## 固定条件与唯一变量

| 项目 | 值 |
|---|---|
| 物理 NPU | `Phy-ID 2,3`，容器映射逻辑设备 `2,3` |
| 并行度 | `tensor_parallel_size=2` |
| 镜像 | `110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730` |
| 模型 | `Qwen3.6-27B-w8a8`，9 个 shard |
| NPU allocator | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| Pod 资源 | `cpu=64`、`memory=256Gi`、Ascend910=`2` |
| readiness | `/bin/sh -c curl --noproxy '*' ... /health` |
| 唯一差异 | `--safetensors-load-strategy eager` |

## 关键时间线

| 时间（UTC） | 事件 |
|---|---|
| 07:15:16 | TP0、TP1 开始加载模型 |
| 07:15:16-07:15:29 | TP0 从 `0/9` 到 `9/9`，日志计时 `10.37s` |
| 07:15:38-07:15:39 | 两个 TP rank 均报告 `16.8724 GB` 权重加载完成 |
| 07:18:18 | Engine 初始化完成，`158.98s`，其中图编译 `99.60s` |
| 随后 | API Server 启动完成；Pod IP `10.42.17.142:13700/health` 返回成功 |

加载权重后的延迟主要是首次 `torch.compile`、ACL Graph 捕获、KV cache 创建和 warmup，
不是 weight loader 停滞。启动后每卡约使用 `53.5 GiB` HBM，处于 64 GiB 容量内。

## 与 Prefetch 对照

| 指标 | Prefetch | Eager |
|---|---:|---:|
| safetensors 进度 | 15 分钟仍 `0/9` | `9/9`，TP0 约 10.37 秒 |
| `13700` | 未监听 | 已监听 |
| `/health` | connection refused | 成功 |
| NPU worker HBM | 约 20 GiB 后停滞 | 服务就绪时约 53.5 GiB |
| 启动前内核匹配项 | 无 | 无 |

`prefetch` 的后台读取只预热 Linux page cache，主 loader 仍以 `safe_open()` 产生 mmap-backed
tensor；`eager` 则实际改变主 loader 的 tensor 来源。因此本次 A/B 直接验证的是加载策略，
不是单纯 I/O 预热。

## 内核与资源结论

- 启动至健康检查成功前，采集日志中 `devmm`、`Get_user_pages_fast`、`ret=-14`、SVM、
  SMMU、IOMMU 和 fault 关键字匹配数为 `0`。
- 删除诊断实例前已保存容器检查信息、TP 进程采样、NPU、CANN 日志和内核日志；终止产生的
  日志不参与启动根因判断。
- 删除 Deployment 后，NPU 2、3 的进程表为空；HBM 已回落到约 `3112 MiB` 和 `2888 MiB`
  的设备基础占用。

## 后续建议

1. 将普通 Prefill/PD 前的权重加载默认固定为 `eager`，不要再把 `prefetch` 当作该节点的
   修复方案。
2. 先在相同 TP2 配置下再做两次冷启动复验，再恢复完整 1P2D；复验期间仍保持 Mooncake、
   Decode 与 Proxy 关闭，避免混入新变量。
3. 三次均稳定后，再在 PD Prefill 实例上加入相同 `eager` 参数，最后才逐个启用 Mooncake、
   Decode 与 Proxy。

## 证据位置

```text
diagnostics/eager_tp2/entrypoint.sh
diagnostics/eager_tp2/qwen36-prefill-tp2-eager.yaml
diagnostics/runs/eager-tp2-20260803T071416Z/
```
