# 当前状态

更新时间：2026-08-03 07:24 UTC。

## Eager TP2 修复验证

普通 TP2 对照在物理 `Phy-ID 2,3` 上已使用唯一变量
`--safetensors-load-strategy eager` 完成验证：9 个 shard 全部加载，`13700/health`
成功，启动阶段未出现 devmm/page-pinning/SMMU 关键字错误。诊断 Deployment 与
ConfigMap 已删除，NPU 2、3 已释放到基础 HBM 占用；完整 1P2D Deployment
`ray-vllm-pd-worker-qwen36-27b` 仍是 `replicas=0`。

详细结果：

```text
markdowns/EAGER_TP2_RESOLUTION_REPORT_20260803_CN.md
```

## Prefetch TP2 受控诊断

完整 1P2D Deployment `ray-vllm-pd-worker-qwen36-27b` 在诊断前后均为
`replicas=0`，Mooncake、Decode 和 Proxy 均未启动。一次性普通 TP2 诊断实例
使用物理 NPU 2、3，唯一加载变量为：

```text
--safetensors-load-strategy prefetch
```

15 分钟内后台 prefetch 已完成 TP0 5/5 和 TP1 4/4，但主 loader 仍停在
`0/9`，`13700` 没有监听，`/health` 失败，实验结论为 **FAIL**。
启动观察期间没有 devmm/page-pinning 内核错误；`ret=-14` 仅在删除
诊断 Pod 时出现，已单独归类为终止阶段日志。

诊断 Deployment 和 ConfigMap 已删除，物理 NPU 2、3 上的诊断进程已退出。
下一轮应只把策略改为 `eager`，本轮未执行。

完整报告：

```text
markdowns/PREFETCH_TP2_DIAGNOSTIC_REPORT_20260803_CN.md
markdowns/PREFETCH_TP2_FAILURE_ROOT_CAUSE_ANALYSIS_20260803_CN.md
```

## 2026-08-03 暂停状态

本次启动已按用户要求暂停，Deployment 已缩容到 `replicas=0`，Pod 已删除。
物理 `Phy-ID 2-7` 无进程且 HBM 已回落到设备基础占用；`Phy-ID 8-15` 上的
原有 vLLM 未受影响。Prefill TP2 完成设备映射和 HCCL 初始化后，模型加载长时间
停留在 safetensors `0/9`，端口未监听；Decode 和 Proxy 尚未启动。

完整证据和恢复建议见：

```text
markdowns/STARTUP_PAUSE_AND_DIAGNOSIS_20260803_CN.md
```

### 内核门禁诊断结果

故障窗口中已确认两个 Prefill TP Worker 出现 Ascend devmm
`Get_user_pages_fast fail`、page pinning 失败和 `ret=-14`。错误发生在 Pod 删除时刻，
尚不能证明它是加载停滞的起因，但已经满足停止门禁。本轮没有启动普通 Prefill TP2
对照实例，Deployment 保持 `replicas=0`。

详细报告：

```text
markdowns/PREFILL_TP2_KERNEL_GATE_REPORT_20260803_CN.md
```

### 宿主机与镜像兼容性审计

只读审计四项结果：版本兼容性 PASS、SVM 设备可用性 PASS、内核启动参数 FAIL、
memlock/页管理 FAIL。Deployment 继续保持 `replicas=0`，未启动任何模型实例。

详细报告：

```text
markdowns/HOST_IMAGE_COMPATIBILITY_AUDIT_20260803_CN.md
```

## 已完成

- `Qwen3.6-27B-w8a8` 权重下载与 9 个分片完整性检查。
- 基于 `v0.22.1rc1-a3` 构建 PD Worker 镜像。
- Ray `2.48.0`、Mooncake、`mooncake_master` 和官方 PD Proxy 构建门禁。
- 镜像推送到 HTTP 私库并校验 manifest digest。
- 在 `infra-learning` 注册 ConfigMap、Service 和 `replicas=0` Deployment。
- 将构建日志复制到 `markdowns/raw/`。
- 清理 rootless Podman 临时镜像，存储从约 18GB 降至 168KB。

## 镜像

```text
110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730
sha256:15c3a3db3772807cc09d9ad37756cd973e3b078956b6a239375ec4ae23317133
```

## 当前阻断

2026-07-31 00:58 UTC 启动前门禁曾确认物理 `Phy-ID 2-7` 无进程，
但启动期间宿主机 Docker 容器 `/llamafactory` 开始了新的 8 卡训练：

```text
image: hiyouga/llamafactory:latest-npu-a3
job: examples/train_full/qwen3_full_sft_0p15_train.yaml
container: f46ab260f3505e632183729ae369bc0884502f7ccc1481ba9b50a160148de3a6
processes: 4021280-4021287
```

该训练覆盖物理 `Phy-ID 0-7`，每卡占用约 `34.7-34.8 GiB` HBM。
物理 `Phy-ID 8-15` 同时由既有 vLLM 进程占用约 `55 GiB` HBM。
当前 16 个逻辑 NPU 均没有可安全分配的空位。

检测到冲突后，PD Deployment 已立即缩容为 `replicas=0`。我方
`VLLMWorker_TP` 进程已经退出，没有终止或修改 `/llamafactory` 训练任务，
也没有改动物理 `Phy-ID 8-15` 上的既有 vLLM 服务。

## 卡空闲后的启动命令

在 A3 上确认：

```bash
cd /home/admin/qwen36_pd_1p2d
sudo python3 scripts/check_target_npus_idle.py
```

只有输出 `PASS` 后，在 server-00 执行：

```bash
cd /home/admin/testpanxy/infralearning/qwen36_pd_1p2d
NPU_IDLE_CONFIRMED=YES ./start.sh
```

监控：

```bash
sudo /usr/local/bin/k3s kubectl -n infra-learning get pod -w
sudo /usr/local/bin/k3s kubectl -n infra-learning logs \
  deploy/ray-vllm-pd-worker-qwen36-27b -f
```

停止：

```bash
cd /home/admin/testpanxy/infralearning/qwen36_pd_1p2d
./rollback.sh
```
