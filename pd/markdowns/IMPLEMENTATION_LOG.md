# 实施记录

## 2026-08-03 宿主机与镜像只读兼容性审计

- 确认 Deployment `replicas=0` 且无对应 Pod。
- 镜像源码 revision 精确对应 vLLM `v0.22.1` 和 vLLM-Ascend `v0.22.1rc1`，
  `0.22.1+empty` 不判错。
- 镜像为 CANN 9.0.0、torch-npu 2.10.0；按当前 Installation 页要求的
  9.0.1/2.10.0.post2 判版本门禁 FAIL。
- `/dev/devmm_svm` 存在；kubelet checkpoint 证明 Device Plugin 向上次 Worker
  注入 devmm_svm、manager、hisi_hdc 和 davinci2-7，SVM 门禁 PASS。
- `/proc/cmdline` 含两项 `smmu.bypassdev`；全局 IOMMU 为 Translated/strict，
  但旁路目标未得到平台证明，内核参数门禁 FAIL。
- k3s-agent 和现有容器实际 memlock 均为 64 MiB；结合 max_map_count=65530、
  THP=always 和历史 devmm 错误，页管理门禁 FAIL。
- 报告：`markdowns/HOST_IMAGE_COMPATIBILITY_AUDIT_20260803_CN.md`。

## 2026-08-03 Prefill TP2 内核门禁

- 校正 A3 `Asia/Shanghai` 与启动记录 UTC 的时区差异，按绝对 UTC 重新导出
  `01:00-01:31 UTC` 内核日志。
- 捕获 5 条 Ascend devmm 错误，涉及两个 TP PID `3068957/3069323`：
  `Get_user_pages_fast fail`、`ret=-14` 和 `Get pa list failed (pin_flg=3)`。
- 错误时间为 `01:30:02 UTC`，与 Pod 删除同时；按门禁停止，不创建普通 Prefill
  对照实例，也未应用 prefetch/readiness 变更。
- 采集本地 NVMe EXT4 模型存储、64G Pod shm、软件版本和 NPU 基线，并归档原始
  输出及 SHA256。
- 报告：`markdowns/PREFILL_TP2_KERNEL_GATE_REPORT_20260803_CN.md`。

## 2026-08-03 启动暂停与诊断

- 启动前确认物理 `Phy-ID 2-7` 无运行进程，`Phy-ID 8-15` 由既有 vLLM 占用。
- Pod 成功绑定物理 2-7，Ray 注册 64 CPU、6 NPU 和 PD 自定义资源。
- Prefill TP2 完成 HCCL 初始化，模型路径和 Ascend W8A8 量化路径识别正确。
- 两个 rank 的 HBM 增至约 20 GiB，但日志长期停留在 safetensors `0/9`，
  `13700` 未监听，主线程持续内核态忙转，并出现异常巨大的虚拟地址空间。
- 按用户要求将 Deployment 缩容到 0；Pod 和 PD 进程已删除，物理 2-7 已释放，
  未影响物理 8-15 的既有服务。
- 详细记录：`markdowns/STARTUP_PAUSE_AND_DIAGNOSIS_20260803_CN.md`。

## 2026-07-30 只读审计

- A3 节点：`a3-server-00`，驱动 `26.0.rc1`，16 个 `Ascend910` 逻辑资源。
- `NPU 1/2/3` 对应物理 `Phy-ID 2-7`，审计时均无运行进程。
- 目标模型已下载到 `/home/admin/models/Qwen3.6-27B-w8a8`。
- 权重共 9 个 safetensors 分片，索引引用完整，目录约 34GiB。
- 现有 Ray Head 位于 `infra-learning`，Ray 版本为 `2.48.0`。
- 用户确认 Worker 资源固定为 `64 CPU / 256Gi memory`。

## 镜像与网络决策

- 基础镜像：`quay.io/ascend/vllm-ascend:v0.22.1rc1-a3`。
- Quay 直连 manifest 检查成功，约耗时 28 秒。
- `m.daocloud.io` 镜像地址返回 `denied: Read Only`，不采用。
- 初始构建使用 Quay 直连；只有拉层持续停滞时才启用
  `http://110.120.0.3:18080` 代理。
- Python 包使用华为云 PyPI 镜像。

## 设计决定

- 使用一个 Kubernetes Deployment 和一个 Pod，申请 6 个 Ascend 逻辑资源。
- Pod 内运行一个 Prefill TP2、两个 Decode TP2 和一个官方 PD Proxy。
- 使用非 Layerwise `MooncakeConnectorV1`，先验证正确性，再做性能调优。
- 三个模型顺序启动：Prefill、Decode A、Decode B。
- 禁用 prefix cache 和 speculative decoding，避免首轮同时引入额外变量。
- Mooncake `kv_port` 使用 36000、36100、36200，避开 A3 的随机传输端口范围。

## 2026-07-30 构建与推送结果

- Quay 直连拉取时，层 `1e43eb0ddf11` 和 `086b0120f2ec` 连续发生网络重试。
- 为避免修改或重启 Docker daemon，终止直连构建，改用 rootless Podman：

  ```bash
  export HTTP_PROXY=http://110.120.0.3:18080
  export HTTPS_PROXY=http://110.120.0.3:18080
  podman pull quay.io/ascend/vllm-ascend:v0.22.1rc1-a3
  podman save quay.io/ascend/vllm-ascend:v0.22.1rc1-a3 | sudo docker load
  ```

- 基础镜像 manifest：`sha256:c4d766b5f04fe6238a74731d67a215bb6331072ba242c7c5f24a25f99ce36c3b`。
- 派生镜像本地 ID：`sha256:6d454e6d5715ac8792868408e57f9287aa0444867db23747b797ddaae5ff924a`。
- 派生镜像大小：`18270627939` bytes，约 18.27GB。
- 构建门禁通过：
  - vLLM-Ascend 基础版本为 `0.22.1rc1-a3`；
  - Ray 为 `2.48.0`；
  - Python 可以导入 `mooncake`；
  - `/usr/local/bin/mooncake_master` 存在；
  - 官方非 Layerwise PD Proxy 文件存在。
- 私库只提供 HTTP，Docker 首次推送报
  `server gave HTTP response to HTTPS client`。没有修改全局
  `insecure-registries`，而是使用：

  ```bash
  sudo docker save "$IMAGE" | podman load
  podman push --tls-verify=false "$IMAGE"
  ```

- 私库镜像：
  `110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730`
- 私库 digest：
  `sha256:15c3a3db3772807cc09d9ad37756cd973e3b078956b6a239375ec4ae23317133`

## 部署前资源门禁

镜像推送完成后再次检查 A3，发现目标资源状态已发生变化：

```text
NPU 1: 两个 python3.11 训练进程，HBM 约 59GiB / 53GiB
NPU 2: 两个 python3.11 训练进程，HBM 约 43GiB / 40GiB
NPU 3: 两个 python3.11 训练进程，HBM 约 43GiB / 39GiB
```

进程命令：

```text
/usr/local/python3.11.15/bin/python3.11
  -u /app/src/llamafactory/launcher.py
  examples/train_full/qwen3_full_sft_0p15_train.yaml
```

归属 Docker 容器：

```text
name=/llamafactory
image=hiyouga/llamafactory:latest-npu-a3
container=f46ab260f3505e632183729ae369bc0884502f7ccc1481ba9b50a160148de3a6
```

该容器由宿主机 Docker 直接启动，不在 Kubernetes 设备插件的资源账本中。
因此 Kubernetes 即使允许 Pod 调度，也可能把相同物理设备再次分配给 PD Worker。
为避免干扰训练任务，Deployment 使用 `replicas=0` 安全注册，不启动模型。

启动前必须在 A3 执行：

```bash
cd /home/admin/qwen36_pd_1p2d
sudo python3 scripts/check_target_npus_idle.py
```

只有输出 `PASS` 后，才在 server-00 执行：

```bash
cd /home/admin/testpanxy/infralearning/qwen36_pd_1p2d
NPU_IDLE_CONFIRMED=YES ./start.sh
```

安全注册结果：

```text
deployment.apps/ray-vllm-pd-worker-qwen36-27b  replicas=0
service/qwen36-pd                              ClusterIP=10.43.212.138
configmap/qwen36-pd-worker-scripts             created
```

资源已经注册到 `infra-learning`，但没有创建 Pod、没有申请 NPU，也没有启动
vLLM。首次 `deploy.sh` 的最后一个组合 `kubectl get` 写法会将两个名称同时应用
到两种资源类型，导致非破坏性的 `NotFound` 返回；Deployment、Service 和
ConfigMap 均已成功创建，脚本已拆成两条精确查询命令。

## 临时存储清理

构建和推送结束后，原始及派生镜像仍保留在 Docker，并且派生镜像已经进入
私库。rootless Podman 仅用于代理拉取和 HTTP 私库推送，因此执行：

```bash
podman rmi \
  110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730 \
  quay.io/ascend/vllm-ascend:v0.22.1rc1-a3
podman image prune -f
```

清理后 Podman 存储为 `168K`。原始构建日志已经复制到
`markdowns/raw/`。

## 2026-07-31 启动尝试与自动回退

00:58 UTC 启动前，A3 主机门禁返回：

```text
PASS: NPU 1/2/3 均无运行进程，可以启动 PD Worker。
```

随后在 server-00 执行 `NPU_IDLE_CONFIRMED=YES ./start.sh`。私库镜像在
约 35 秒内拉取完成，容器内设备映射验证为：

```text
prefill=2,3
decode_a=4,5
decode_b=6,7
```

Ray Worker 成功注册：

```text
CPU=64
NPU=6
PD_PREFILL=1
PD_DECODE=2
QWEN36_PD_WORKER=1
```

Prefill TP2 完成 9 个权重分片加载，每个 rank 权重约 `16.87 GB`，随后进入
Ascend 首次图/算子编译。为避免首次编译被过短的健康等待窗口终止，
`VLLM_STARTUP_TIMEOUT` 默认值由 1200 秒调整为 3600 秒，并更新了 ConfigMap。

01:21 UTC 再次检查时，发现宿主机 `/llamafactory` 容器已经启动新的 8 卡
Qwen3 全参 SFT：

```text
command=/app/src/llamafactory/launcher.py \
  examples/train_full/qwen3_full_sft_0p15_train.yaml
PIDs=4021280-4021287
physical_npus=0-7
per_npu_hbm=约 34.7-34.8 GiB
```

该任务由宿主机 Docker 直接管理，不受 Kubernetes 设备插件排他调度约束，
因此它在 PD Pod 启动后占用了相同的物理卡 2、3。物理卡 8-15 另有既有
vLLM 进程，当前没有可迁移的空闲卡。

发现资源冲突后立即执行：

```bash
sudo /usr/local/bin/k3s kubectl -n infra-learning scale \
  deployment/ray-vllm-pd-worker-qwen36-27b --replicas=0
```

缩容后我方 Prefill 进程全部退出；`npu-smi` 只保留原训练任务和既有 vLLM
进程。没有停止、重启或修改任何非本项目工作负载。本次未启动 Decode 和
PD Proxy，也未执行推理 smoke。

## 2026-08-03 普通 TP2 `prefetch` 受控诊断

- 将版本门禁按 `vLLM-Ascend v0.22.1rc1` 官方 release matrix 修正为
  PASS；当前镜像的 CANN 9.0.0 和 torch-npu 2.10.0 是该 release 的精确
  组合，未升级或混装。
- 主 PD Deployment 始终保持 `replicas=0`，未启动 Mooncake、Decode 或
  Proxy。
- 创建独立诊断 Deployment，绑定物理 NPU 2、3，普通 vLLM TP2 仅增加
  `--safetensors-load-strategy prefetch`；readiness 使用 `/bin/sh -c`。
- 启动前保存 `/proc/1/limits`、CapEff/CapBnd 和 allocator。容器实测
  memlock 为 64 MiB，capability mask 为 `000001ffffffffff`。
- 两个 rank 完成 HCCL 初始化和 CANN device setup，分别绑定 devid 2、3。
- 后台 page-cache prefetch 在数秒内完成 TP0 5/5 和 TP1 4/4，但主
  safetensors loader 在 15 分钟内一直为 0/9，`13700` 和 `/health` 未就绪。
- 两个 Worker 长期使用约一个 CPU 核的 system time，minor faults 平均约
  20.5 万/s 和 16.4 万/s；AICore 基本为 0，HBM 峰值约 20.6/20.3 GiB。
- 启动观察期间内核日志没有 devmm、`Get_user_pages_fast`、`ret=-14` 或
  page-pinning 错误。删除 Pod 时才出现该类日志，已与启动现场分开归档。
- 诊断 Deployment 和 ConfigMap 已删除；主 PD Deployment 仍为 0。
- 完整现场位于
  `diagnostics/runs/prefetch-tp2-20260803T025936Z/`，报告位于
  `markdowns/PREFETCH_TP2_DIAGNOSTIC_REPORT_20260803_CN.md`。
- 下一个单变量应为 `--safetensors-load-strategy eager`，本次未执行。

## 2026-08-03 失败路径补充分析

- 使用同一镜像不申请 NPU，对同一模型执行只读权重校验：9 个 shard、
  1725 个 tensor、36,423,542,240 bytes 全部可解析。`safe_open` 遍历约
  1.7 秒，逐 shard `read()+safetensors.torch.load()` 约 21.8 秒。
- A3 当前约 1.5 TiB 可用 RAM、CPU 约 98-100% idle、I/O wait 约 0%，排除
  常规节点资源压力。
- 正常运行的 vLLM Worker 同样具有约 9.72 PB VmSize、64 MiB memlock、
  `VmLck=0` 和 `expandable_segments:True`，因此这些单独不是首要根因。
- 官方源码确认 `prefetch` 仅后台读入 page cache，主 loader 仍使用
  `safe_open/get_tensor` mmap 路径；`eager` 才使用 `open().read()` 绕开文件 mmap tensor。
- 当前最可能的路径是 mmap-backed CPU tensor 在 ModelSlim/TP 切片后执行
  `param.data.copy_()` 到 NPU 时发生页故障/VMA 退化或活锁。该结论为高置信
  工程推断，尚需 eager A/B 或逐 tensor copy 追踪确认。
- 详细根因排序和解决矩阵已写入
  `markdowns/PREFETCH_TP2_FAILURE_ROOT_CAUSE_ANALYSIS_20260803_CN.md`。
