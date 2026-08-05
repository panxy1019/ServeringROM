# 普通 Prefill TP2 `prefetch` 受控诊断报告

实验日期：2026-08-03  
观察窗口：02:59:08-03:15:52 UTC  
诊断运行 ID：`prefetch-tp2-20260803T025936Z`

## 1. 结论

本轮结果为 **FAIL**：在最长 15 分钟观察期内，模型一直停留在：

```text
Loading safetensors checkpoint shards: 0% Completed | 0/9
```

`13700` 没有监听，`/health` 无法连接，因此不满足“9 个 shard 完成、端口监听、
health 成功”三个联合 PASS 条件。

`prefetch` 的后台 page-cache 预取本身已完成：TP0 完成 5/5 文件，TP1 完成 4/4
文件，但主 safetensors loader 仍停在 0/9。因此，本次结果不支持“默认 lazy mmap
是唯一或主要触发条件”的判断；`prefetch` 只能提前把文件页读入 page cache，没有
消除后续映射、缺页或设备侧 host-memory 注册路径上的停滞。

本轮未启动 Mooncake、Decode、Proxy，也未修改 memlock、THP、
`vm.max_map_count`、SMMU、allocator、镜像、模型或其他 Kubernetes 参数。完整
1P2D Deployment 在实验前后始终保持 `replicas=0`。

按单变量实验顺序，下一轮应只把加载策略由 `prefetch` 改成 `eager`。本报告没有
执行 eager、TP1 或新版官方栈对照。

## 2. 版本门禁

版本兼容性为 **PASS**，依据是 `vLLM-Ascend v0.22.1rc1` 自身的官方 release
matrix，而不是后续 latest 安装页：

| 组件 | 实测版本 | 结论 |
|---|---:|---|
| vLLM | `0.22.1+empty`，源码 tag `v0.22.1` | PASS |
| vLLM-Ascend | `0.22.1rc1`，源码 tag `v0.22.1rc1` | PASS |
| torch | `2.10.0+cpu` | PASS |
| torch-npu | `2.10.0` | PASS |
| CANN | `9.0.0` | PASS |
| triton-ascend | `3.2.1` | PASS |

`+empty` 是 Ascend 插件构建 vLLM 时使用的目标设备形式，不单独构成错误。实验
保持这一整套 release 组合，没有升级或混装 CANN/torch-npu。

## 3. 实验边界

### 3.1 未触碰的生产对象

```text
Deployment: ray-vllm-pd-worker-qwen36-27b
实验前 replicas: 0
实验后 replicas: 0
Mooncake: 未启动
Decode: 未启动
PD Proxy: 未启动
```

物理 NPU 8-15 上的既有 vLLM 服务未修改。实验仅使用物理 NPU 2、3，结束后两张
卡上的诊断进程均已退出，HBM 回落到设备基础占用。

### 3.2 唯一功能变量

诊断对象是一个独立的一次性 Deployment：

```text
qwen36-prefill-tp2-prefetch-diagnostic
```

相对现有普通启动参数仅做以下变化：

```text
移除 Mooncake/PD/KV transfer 参数
不启动 Decode 和 Proxy
--safetensors-load-strategy prefetch
readinessProbe 使用 /bin/sh -c 调用 /health
```

保持不变的关键项：

```text
image: 110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730
model: /models/Qwen3.6-27B-w8a8
tensor-parallel-size: 2
quantization: ascend
max-model-len: 32768
max-num-batched-tokens: 8192
max-num-seqs: 16
gpu-memory-utilization: 0.88
PYTORCH_NPU_ALLOC_CONF: expandable_segments:True
resources: cpu=64, memory=256Gi, huawei.com/Ascend910=2
devices: physical NPU 2,3
/dev/shm: 64Gi
securityContext.privileged: true
```

## 4. 启动时序

| UTC 时间 | 事件 |
|---|---|
| 02:59:08 | Pod 容器启动 |
| 02:59:09 | 保存 `/proc/1/limits`、CapEff、CapBnd 和 allocator |
| 02:59:26 | API Server 输出 vLLM 版本和普通 TP2 参数 |
| 03:00:21 | TP rank 0 完成 HCCL 初始化 |
| 03:00:31 | TP rank 1 完成 HCCL 初始化 |
| 03:00:38 | 两个 rank 开始加载模型 |
| 03:00:40 | 主 loader 显示 0/9；后台 prefetch 启动 |
| 03:00:42 | TP0 的 5/5、TP1 的 4/4 后台文件预取完成 |
| 03:14:32 | 最后一组进程采样，主 loader 仍为 0/9 |
| 03:15:52 | 保存终止前现场，观察结束 |
| 03:22:56 | 删除诊断 Deployment；终止阶段产生 devmm 错误 |

## 5. 验收结果

| 条件 | 结果 | 证据 |
|---|---|---|
| 9 个 shard 加载完成 | FAIL | 15 分钟一直为 0/9 |
| `13700` 监听 | FAIL | 终止前 listener 文件为空 |
| `/health` 成功 | FAIL | `curl` connection refused |
| 三项联合 PASS | **FAIL** | 任一条件失败即失败 |

Readiness 使用 `/bin/sh -c`，所以本轮失败不是 Kubernetes probe 对 shell 语法的
解释差异。

## 6. CPU、内存和文件读取

采集器每秒读取 `/proc`，实际输出受宿主机调度影响，共得到每个 Worker 约 124-125
个有效采样点。两个 TP Worker 的核心统计如下：

| 指标 | TP0，host PID 3299908 | TP1，host PID 3300935 |
|---|---:|---:|
| 有效采样区间 | 03:00:11-03:14:32 | 03:00:18-03:14:32 |
| CPU system 平均 | 97.78% | 98.26% |
| CPU user 平均 | 2.43% | 3.09% |
| minor fault 平均增量/采样 | 205,138 | 163,927 |
| major fault 平均增量/采样 | 0 | 0.04 |
| RSS 峰值 | 3,798,860 KiB | 3,809,180 KiB |
| VmSize 峰值 | 9,679,462,828 KiB | 9,679,239,268 KiB |
| 最终累计 read_bytes | 4,334,481,408 | 5,350,285,312 |

后台预取在数秒内结束后，`read_bytes` 很快停止增长；Worker 随后长期表现为一个核
左右的 system CPU、巨量 minor faults 和约 9.68 PB 虚拟地址空间。这说明停滞点不在
“磁盘持续慢读 34 GiB 权重”上，更接近内存映射/页表/缺页或设备 host-memory 注册
相关路径。该判断是基于现象的定位，不等同于已经证明某一个内核参数是根因。

线程栈快照中长期可见：

```text
futex_wait_queue
vm_mmap_pgoff
lock_mm_and_find_vma
__vm_munmap
hdcdrv_recv_peek_wait
```

大量线程处于 futex 等待，少量关键线程持续位于 mmap/VMA 相关内核路径，与上述
system CPU 和 minor-fault 模式一致。

## 7. NPU 观测

| 物理 NPU | AICore 平均/峰值 | HBM 基线/峰值 |
|---|---:|---:|
| Phy-ID 2 | 0.00% / 0% | 3,113 / 20,565 MiB |
| Phy-ID 3 | 0.02% / 2% | 2,889 / 20,335 MiB |

终止前 `npu-smi` 可见两个 `VLLMWorker_TP`，每个设备进程约 17,502 MiB。HBM 已
分配，但没有进入实际模型算子计算，因此 AICore 基本为 0。CANN plog 表明：

```text
TP0 setup device succeeded: devid=2, hostpid=3299908
TP1 setup device succeeded: devid=3, hostpid=3300935
devmm host mem pool init success
```

这同时确认了两个 rank 与物理 NPU 2、3 的绑定。

## 8. 内核日志的时间隔离

### 8.1 启动与观察阶段

从 Pod 启动到终止前保存现场，实时和完整内核日志中没有以下匹配：

```text
devmm
Get_user_pages_fast
pin_user_pages
ret=-14
SVM/SMMU/IOMMU fault
```

该窗口只记录到 CNI veth 创建事件。因此没有证据表明 `ret=-14` 在模型启动阶段
触发或直接导致 0/9。

### 8.2 终止阶段

删除诊断 Deployment 的 `2026-08-03 03:22:56 UTC`，内核才记录：

```text
devmm_pin_user_pages_fast: Get_user_pages_fast fail
devmm_get_non_svm_addr_pa_list: Get user pages fail (ret=-14)
devmm_make_host_pa_node_list: Get pa list failed
```

日志中的进程正是已经保存的两个 TP host PID。由于错误发生在删除 Pod、终止进程
的时刻，它们必须归类为 teardown 现场，不能倒推为启动阶段的原因。这也验证了
“终止前先保存现场、终止后日志单独归档”的必要性。

## 9. 对 `prefetch` 假设的判断

`prefetch` 已真实生效，且 9 个文件按 TP 分片全部完成后台预取，但主 loader 没有
完成任何一个 shard。因此：

1. 默认 lazy mmap 路径不是当前证据下可确认的主要触发条件。
2. 仅把权重页读入 Linux page cache 不足以绕过停滞。
3. `prefetch` 没有让故障转变为显式 Python/CANN 异常，而是保持长期活锁式状态。
4. 下一步 `eager` 有诊断价值，因为它会进一步改变 safetensors 的实际加载/复制
   方式，而不仅是增加后台 readahead。

下一轮必须继续保持单变量：只改为：

```text
--safetensors-load-strategy eager
```

若 eager 仍失败，再分别进行 TP1 和完整新版官方栈对照；不能同时修改 memlock、
THP、SMMU 或多个软件版本，否则无法归因。

## 10. 现场与复现文件

本地证据目录：

```text
diagnostics/runs/prefetch-tp2-20260803T025936Z/
```

关键文件：

```text
qwen36-prefill-tp2-prefetch.yaml       独立诊断 Deployment
entrypoint.sh                           启动、前置记录和保活逻辑
collect_host.sh                         宿主机采集器
pidstat_compat.py                       无 sysstat 时的 /proc 采样器
extracted/vllm-logs/vllm.log            完整 vLLM 启动日志
extracted/vllm-logs/prestart-container.txt
pidstat-compat.jsonl                    CPU/内存/缺页/I/O 采样
npu-smi.log                             HBM/AICore 连续采样
stacks/                                 20 份线程栈现场
extracted/cann-plog/                    CANN plog
kernel-before-termination.log           启动阶段内核日志
kernel-after-termination-filtered.log   终止阶段 devmm 日志
npu-before-termination.log              终止前设备进程现场
npu-after-termination.log               清理后设备现场
kubernetes-pre-termination.txt          Pod/Deployment/主 PD 配置现场
```

诊断 Deployment 和临时 ConfigMap 已删除；保留的是文件和报告，不存在继续占用
物理 NPU 2、3 的本轮资源。
