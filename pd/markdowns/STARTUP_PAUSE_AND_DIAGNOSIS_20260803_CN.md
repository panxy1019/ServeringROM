# Qwen3.6 PD Worker 启动暂停与诊断记录

更新时间：2026-08-03 01:30 UTC

## 1. 当前结论

本次 PD Worker 启动已按要求暂停。Kubernetes Deployment
`ray-vllm-pd-worker-qwen36-27b` 已缩容到 `replicas=0`，对应 Pod 已删除。

资源释放核验结果：

- 物理 `Phy-ID 2-7` 上没有运行进程；
- 物理 `Phy-ID 2-7` 的 HBM 已回落到约 `2.9-3.1 GiB` 的设备基础占用；
- 物理 `Phy-ID 8-15` 上原有 vLLM 进程和约 `55-59 GiB` HBM 占用保持不变；
- 没有停止、重启或修改其他训练及推理任务；
- PD 镜像、Deployment、Service、ConfigMap 和模型权重均保留，后续可以继续诊断或恢复。

截至暂停时，Prefill TP2 尚未通过健康检查，Decode A、Decode B 和 PD Proxy
均未启动，因此本次不能判定 PD 服务部署成功。

## 2. 本次启动配置

```text
namespace: infra-learning
deployment: ray-vllm-pd-worker-qwen36-27b
image: 110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730
model: /models/Qwen3.6-27B-w8a8
worker resources: 64 CPU / 256 GiB / 6 Ascend910
```

计划中的服务布局：

| 服务 | 物理 NPU | 容器逻辑 NPU | TP | 端口 | Mooncake 角色 |
|---|---:|---:|---:|---:|---|
| Prefill | 2,3 | 2,3 | 2 | 13700 | `kv_producer` |
| Decode A | 4,5 | 4,5 | 2 | 13701 | `kv_consumer` |
| Decode B | 6,7 | 6,7 | 2 | 13702 | `kv_consumer` |
| Proxy | - | - | - | 8080 | 请求分发 |

容器启动后重新执行了设备发现，确认本次物理到逻辑设备映射没有发生重编号。
Ray Worker 也曾成功注册：

```text
CPU=64
NPU=6
PD_PREFILL=1
PD_DECODE=2
QWEN36_PD_WORKER=1
```

## 3. 启动时间线

```text
01:01:59  Pod 创建并进入 Running
01:02:24  Prefill vLLM 进程开始启动
01:03:29  TP rank 0 完成 HCCL 初始化
01:03:39  TP rank 1 完成 HCCL 初始化
01:03:47  两个 TP Worker 开始加载模型
01:03:49  日志停留在 Loading safetensors checkpoint shards: 0/9
01:22-01:29 持续诊断，13700 未监听，Pod 无重启
01:30     按用户要求缩容到 replicas=0 并确认 NPU 释放
```

## 4. 已确认正常的部分

### 4.1 调度和设备绑定

Pod 正确调度到 `a3-server-00`，申请了 6 个 Ascend910 资源。容器内映射为：

```text
physical 2 -> logical 2
physical 3 -> logical 3
physical 4 -> logical 4
physical 5 -> logical 5
physical 6 -> logical 6
physical 7 -> logical 7
```

### 4.2 Ray 和 HCCL

Ray Worker 成功加入现有 Ray Head。Prefill 的两个 rank 建立了 TP=2 的 HCCL
进程组，没有发现 rank 缺失、world size 错误或 HCCL 初始化报错。

### 4.3 模型和量化路径

vLLM 正确识别：

```text
vLLM: 0.22.1
architecture: Qwen3_5ForConditionalGeneration
quantization: ascend / modelslim
checkpoint: 33.92 GiB, 9 safetensors shards
```

两个 TP rank 对应物理卡 2、3 的 HBM 从约 3 GiB 上升到约 20 GiB，说明
NPU 上已经发生了约 17.5 GiB/rank 的模型内存分配。暂停前没有出现 OOM、
NPU health 异常或容器重启。

## 5. 未完成及异常现象

### 5.1 Prefill 未就绪

从 `01:03:49` 到暂停，日志始终停留在：

```text
Loading safetensors checkpoint shards: 0% Completed | 0/9
```

同时：

- `127.0.0.1:13700/health` 返回连接拒绝；
- AICore 利用率为 0%；
- HBM 稳定在约 20 GiB/卡，不再增长；
- 两个 TP 主线程均为 `R` 状态；
- 每个主线程持续消耗约一个 CPU 核，且时间主要累计在内核态；
- Python `VmSize` 显示约 9.9 PB 的巨大虚拟地址空间；
- 读取 `/proc/<pid>/maps` 和使用 `py-spy` 附加都异常缓慢。

这些证据更接近“加载或 Ascend 内存映射阶段陷入内核态忙转”，而不是网络下载、
普通磁盘 I/O、HCCL 等待或 NPU 计算。HBM 已占用不能等同于 9 个分片已经成功
加载，也不能据此认定模型服务可用。

### 5.2 Readiness Probe 存在独立问题

当前 readiness probe 使用：

```bash
/bin/bash -lc 'test -f /var/run/qwen36-pd/READY && curl ...'
```

探针每次约 5 秒超时，而不是在 `READY` 文件不存在时立即返回。容器中的 login
shell 会加载较重的 Ascend 环境，并不断生成 CANN plog。它不是本次模型加载停滞的
唯一证据，但会制造额外噪声和系统开销。下一次重建 Pod 前应改成非 login shell：

```yaml
command: ["/bin/sh", "-c", "test -f /var/run/qwen36-pd/READY && curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null"]
```

### 5.3 Ray 状态短暂残留

Pod 删除后的首次 `ray status` 仍显示 6 NPU 和 PD 自定义资源。这是 Ray 在 Worker
心跳超时前保留的节点记录，不表示 Kubernetes Pod 或 NPU 进程仍然存在。资源是否
真正占用应以 Pod、容器和 `npu-smi info` 为准；恢复前仍应等待 Ray 将旧节点标记为
dead，并再次执行三方核验。

`01:32 UTC` 复查时旧 Worker 已从 Ray 中移除。Ray 当前只剩 Head，资源总量
恢复为 `4 CPU`，不再包含 `NPU`、`PD_PREFILL`、`PD_DECODE` 或
`QWEN36_PD_WORKER`，控制面残留已经清除。

## 6. 暂停后的客观状态

```text
Deployment replicas: 0
PD Pod: 不存在
Prefill/Decode/Proxy: 均未运行
Phy-ID 2-7: 无进程，HBM 为基础占用
Phy-ID 8-15: 原有 vLLM 正常保留
Ray: 仅保留 Head，无 PD Worker 资源
模型权重: 保留
镜像及 Kubernetes 对象: 保留
```

## 7. 建议的恢复顺序

下一轮不要直接再次拉起完整 1P2D，而应先收敛 Prefill 单实例启动问题：

1. 修改 readiness probe 为 `/bin/sh -c`，消除 login shell 噪声。
2. 保持相同模型、TP2、物理卡 2/3 和镜像，只启动一个不带 Mooncake 的普通 vLLM
   对照实例，验证该模型与当前 vLLM/CANN/驱动组合能否健康启动。
3. 如果普通实例仍停在 0/9，优先 A/B：关闭 `expandable_segments`、显式设置经过验证
   的 dtype/Ascend additional config，并采集更细粒度的加载栈。
4. 普通实例通过后，加入 `MooncakeConnectorV1`，只启动 Prefill，再验证健康和一次
   OpenAI API 请求。
5. Prefill 稳定后启动单个 Decode，完成 1P1D KV 传输 smoke。
6. 最后增加 Decode B，形成 1P2D，并执行并发与故障切换测试。

每一步都继续执行物理卡空闲门禁，不直接假设 Kubernetes 的设备资源账本能够覆盖
宿主机 Docker 任务。

## 8. 常用核验命令

在 server-00：

```bash
sudo /usr/local/bin/k3s kubectl -n infra-learning get deploy,pod,svc \
  -l app=ray-vllm-pd-worker-qwen36-27b

sudo /usr/local/bin/k3s kubectl -n infra-learning exec \
  deploy/ray-vllm-lab-head -- ray status
```

在 A3：

```bash
sudo npu-smi info
cd /home/admin/qwen36_pd_1p2d
sudo python3 scripts/check_target_npus_idle.py
```

恢复完整部署前的原入口仍为：

```bash
cd /home/admin/testpanxy/infralearning/qwen36_pd_1p2d
NPU_IDLE_CONFIRMED=YES ./start.sh
```

在完成上述 Prefill 分层诊断前，不建议直接执行该完整入口。
