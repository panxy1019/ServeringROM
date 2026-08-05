# Qwen3.6 PD 宿主机与镜像兼容性审计报告

审计时间：2026-08-03 02:36-02:43 UTC  
审计范围：A3 宿主机、目标镜像、k3s-agent、Ascend Device Plugin 和已删除
`pd-worker` 的 kubelet 分配记录

## 1. 最终结论

本轮为只读审计，没有启动任何模型、Pod 或诊断服务。Deployment 始终为：

```text
ray-vllm-pd-worker-qwen36-27b replicas=0
对应 Pod 数量=0
```

四项门禁结果：

| 门禁 | 结果 | 核心原因 |
|---|---|---|
| 版本兼容性 | **PASS** | 镜像精确匹配 vLLM-Ascend `v0.22.1rc1` 发布矩阵：CANN 9.0.0、torch/torch-npu 2.10.0/2.10.0 |
| SVM 设备可用性 | **PASS** | `/dev/devmm_svm` 存在，kubelet AllocateResponse 明确注入 Pod |
| 内核启动参数 | **FAIL** | 存在两项 `smmu.bypassdev`；虽然全局 IOMMU 为 Translated/strict，但不能给出无旁路 PASS |
| memlock/页管理限制 | **FAIL** | k3s-agent 和现有容器均只有 64 MiB memlock，且 `vm.max_map_count=65530`、THP=`always` |

版本门禁修正为 **PASS**。其余三项保留当时审计结果；后续普通
TP2 诊断是在明确不改动 memlock、THP、`vm.max_map_count` 和 SMMU 的约束下
单独授权执行，不表示其余三项自动转为 PASS。

## 2. 版本兼容性：PASS

### 2.1 镜像实测版本

目标镜像：

```text
110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730
image ID: sha256:6d454e6d5715ac8792868408e57f9287aa0444867db23747b797ddaae5ff924a
```

`importlib.metadata` 和镜像版本文件结果：

| 组件 | 镜像实测 |
|---|---|
| Python | 3.12.13 |
| torch | 2.10.0+cpu |
| torch-npu | 2.10.0 |
| vLLM | 0.22.1+empty |
| vLLM-Ascend | 0.22.1rc1 |
| CANN | 9.0.0 |
| safetensors | 0.8.0 |
| transformers | 5.5.4 |
| triton-ascend | 3.2.1 |

CANN 的软链接和安装文件一致：

```text
/usr/local/Ascend/ascend-toolkit/latest -> /usr/local/Ascend/cann
resolved path: /usr/local/Ascend/cann-9.0.0
ascend_toolkit_install.info: version=9.0.0
runtime/version.info: Version=9.0.0
```

A3 宿主机驱动：

```text
driver package_version=26.0.rc1
ascendhal_version=7.35.23
Innerversion=V100R001C10SPC001B257
```

宿主机没有 `/usr/local/Ascend/ascend-toolkit/latest`，但 Pod 使用的是镜像内 CANN，
并只读挂载宿主机 driver。因此运行时组合是“宿主机 26.0.rc1 driver + 镜像 CANN
9.0.0”，不是宿主机 toolkit。

### 2.2 vLLM `+empty` 和源码 revision

`0.22.1+empty` 本身不判错。镜像保留了完整 Git 元数据：

```text
/vllm-workspace/vllm
commit: 0decac0d96c42b49572498019f0a0e3600f50398
git describe: v0.22.1

/vllm-workspace/vllm-ascend
commit: 5f6faa0cb8830f667266f3b8121cd1383606f2a1
git describe: v0.22.1rc1
```

vLLM 使用 `VLLM_TARGET_DEVICE=empty` 构建、再由 Ascend 平台插件提供设备后端，
符合硬件插件构建方式。两个源码 revision 与目标 tag 精确对应。

### 2.3 官方文档存在两套时间口径

官方当前安装页列出的组合是：

```text
vLLM            0.22.1
vLLM-Ascend     0.22.1rc1
torch           2.10.0
torch-npu       2.10.0.post2
CANN            9.0.1
```

但官方 Versioning Policy 的 release matrix 对 `v0.22.1rc1` 仍列：

```text
CANN            9.0.0
torch/torch-npu 2.10.0 / 2.10.0
triton-ascend   3.2.1
```

本项审计的对象是已固定的 `v0.22.1rc1` release，因此应以该 release 的官方
matrix 为准，不应用后续 latest 安装页要求倒推已发布镜像为不兼容。镜像内
vLLM、vLLM-Ascend、PyTorch、torch-npu、CANN 和 triton-ascend 形成完整的官方
`v0.22.1rc1` 组合，两个源码 revision 也精确对应 tag，故版本兼容性判
**PASS**。本诊断不升级、不混装 CANN 或 torch-npu。

官方参考：

- [当前 Installation 要求](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)
- [Versioning Policy 发布矩阵](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)
- [v0.22.1rc1 Release 和 commit](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.22.1rc1)

## 3. SVM 设备可用性：PASS

### 3.1 宿主机设备节点

```text
/dev/devmm_svm       char device 511:0, mode 0666
/dev/davinci_manager char device 235:0, mode 0666
/dev/davinci0-15     char device 234:0-15, mode 0666
```

`/dev/devmm_svm` 明确存在且为字符设备。

### 3.2 Device Plugin

```text
DaemonSet: ascend-device-plugin-daemonset
image: 110.120.0.3:8889/ascend-k8sdeviceplugin:v26.1.0
MOUNT_BY_RUNTIME_FOR_DP=true
securityContext.privileged=true
registered huawei.com/Ascend910: 16
```

PD Deployment 申请 `huawei.com/Ascend910: 6`，容器为 privileged。静态 Deployment
YAML 不展开 Device Plugin 的 AllocateResponse，因此又读取了 A3 上的 kubelet
checkpoint。上次 `pd-worker` 的实际注入路径包含：

```text
/dev/davinci_manager
/dev/hisi_hdc
/dev/devmm_svm
/dev/dvpp_cmdlist
/dev/davinci2
/dev/davinci3
/dev/davinci4
/dev/davinci5
/dev/davinci6
/dev/davinci7
```

这比仅检查 DaemonSet 模板更直接，证明 `/dev/devmm_svm` 确实进入了目标容器。
因此 SVM 设备存在性和注入链路判 **PASS**。

checkpoint 中 Resource DeviceIDs 的内部编号与最终 `/dev/davinci2-7` 路径并非
同一表示法；本结论使用实际 AllocateResponse 中的 DeviceSpecs，不用内部 ID
反推物理卡。

## 4. 内核启动参数：FAIL

`/proc/cmdline`：

```text
BOOT_IMAGE=/vmlinuz-6.6.0-72.0.0.76.oe2403sp1.aarch64
...
smmu.bypassdev=0x1000:0x17
smmu.bypassdev=0x1000:0x15
...
```

参数检查：

```text
smmu.bypassdev: PRESENT
iommu.passthrough: ABSENT
iommu=pt: ABSENT
iommu=off: ABSENT
```

内核启动日志同时显示：

```text
iommu: Default domain type: Translated
iommu: DMA domain TLB invalidation policy: strict mode
```

因此不是全局 IOMMU passthrough，也没有关闭 IOMMU；这是有利证据。但两个显式
`smmu.bypassdev` 例外确实存在，且仅凭通用 `lspci` 的 vendor/device ID 无法把
`0x1000:0x17/0x15` 安全映射为“与 NPU 无关”的设备。考虑到此前 devmm
`Get_user_pages_fast fail`，在没有 A3 平台/驱动说明确认这些旁路目标前，不能判
启动参数完全安全，故判 **FAIL**。

本轮没有修改 grub、bootloader 或内核参数。

## 5. memlock 和页管理：FAIL

### 5.1 memlock

```text
审计 root shell ulimit -l: 65536 KiB
k3s-agent LimitMEMLOCK: 67108864 bytes
k3s-agent /proc/<pid>/limits soft/hard: 67108864 / 67108864 bytes
```

A3 上现有 vLLM 进程抽样也一致：

```text
Max locked memory: 67108864 / 67108864 bytes
Max address space: unlimited
```

因此容器运行时链路并没有提供 unlimited memlock；privileged 只放宽设备和
capability 访问，不会自动把继承的 RLIMIT_MEMLOCK 改成 unlimited。对需要大量
host page pinning 的 Ascend/vLLM 场景，64 MiB 不能判为充分。

### 5.2 页管理

```text
vm.max_map_count = 65530
vm.overcommit_memory = 1
vm.overcommit_ratio = 50
transparent_hugepage = [always] madvise never
Mlocked = 0 kB（当前无 PD Worker）
HugePages_Total = 0
```

`overcommit_memory=1` 允许积极虚拟内存承诺，THP 为 always；上次两个 Worker 又
出现约 9.9 PB VmSize、`/proc/<pid>/maps` 极慢和 devmm page pinning 错误。
这些参数不等于已经证明根因，但与 64 MiB memlock 合在一起，无法通过页管理门禁。
故判 **FAIL**。

## 6. Kubernetes 安全上下文

PD Deployment 当前关键配置：

```text
replicas=0
securityContext.privileged=true
requests/limits: cpu=64, memory=256Gi, huawei.com/Ascend910=6
/dev/shm: tmpfs 64Gi
strategy: Recreate
```

没有发现 Pod 级 `runAsNonRoot`、seccomp 或 capability drop 阻止设备访问。主要问题
不在 Kubernetes securityContext，而在 k3s-agent/容器实际继承的 64 MiB memlock。

## 7. 原始证据

```text
markdowns/raw/compat-audit-20260803/
├── qwen36_pd_host_compat_audit.txt
├── qwen36_pd_image_compat_audit.txt
├── qwen36_pd_memlock_page_audit.txt
├── qwen36_pd_device_injection_audit.txt
├── qwen36_pd_deploy.yaml
├── ascend-device-plugin-daemonset.yaml
└── a3-server-00-node.yaml
```

所有采集均为只读。镜像审计使用不挂载 driver 和 `/dev/davinci*` 的临时容器，
只读取文件与 package metadata；容器已经自动删除。

## 8. 后续准入条件（本轮不执行）

申请普通无 Mooncake TP2 复现前，至少需要：

1. 明确选择并构建一套单一官方组合；若按当前 Installation 页，应升级到 CANN
   9.0.1 和 torch-npu 2.10.0.post2，并重新核对 driver/NNAL。
2. 为 k3s-agent 及其容器运行时提供足够的 memlock，验证目标容器
   `/proc/1/limits`，不能只看宿主机交互 shell。
3. 由 A3 平台或驱动文档确认两个 `smmu.bypassdev` 的目标和必要性；在确认前不修改
   启动参数，也不把它判为无影响。
4. 评估 `vm.max_map_count` 和 THP 策略，形成受控 A/B，避免同时修改多个变量。
5. 四项门禁全部 PASS 后，再申请一次普通 TP2、无 Mooncake、无 Decode、无 Proxy
   的受控复现。

当前动作：**保持 `replicas=0`，不启动模型。**
