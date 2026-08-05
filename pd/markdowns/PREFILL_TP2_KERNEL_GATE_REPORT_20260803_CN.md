# Prefill TP2 内核门禁诊断报告

诊断时间：2026-08-03 02:12-02:18 UTC  
A3 本地时区：Asia/Shanghai（CST，UTC+8）

## 1. 执行结论

本轮诊断命中了用户定义的硬停止条件，**未启动普通 Prefill TP2 对照实例**，
也没有启动 Mooncake、Decode、Proxy 或完整 1P2D。

故障窗口内存在明确的 Ascend devmm 页固定异常：

```text
devmm_pin_user_pages_fast: Get_user_pages_fast fail
devmm_get_non_svm_addr_pa_list: Get user pages fail (ret=-14)
devmm_make_host_pa_node_list: Get pa list failed (pin_flg=3)
```

因此，本轮没有应用 `--safetensors-load-strategy prefetch`，也没有创建新的
readiness probe。现有 PD Deployment 继续保持 `replicas=0`。

## 2. 时间和时区校正

A3 的 journal 以 CST 展示，而上次启动记录使用 UTC。初次直接执行：

```text
--since "2026-08-03 01:00:00"
--until "2026-08-03 01:31:00"
```

会被 A3 按 CST 解析，因此返回 `-- No entries --`。使用带时区的绝对时间重新导出：

```bash
journalctl -k \
  --since "2026-08-03 01:00:00 UTC" \
  --until "2026-08-03 01:31:00 UTC" \
  --no-pager
```

该窗口对应 A3 日志中的 `09:00:00-09:31:00 CST`。

## 3. 内核日志结果

完整窗口共 16 行，关键词匹配 5 行。在错误前还有：

```text
log_to_printk_and_ringbuf: 4 callbacks suppressed
```

因此可见的 5 行可能不是全部事件。

关键日志如下：

```text
Aug 03 09:30:02 [ascend] [devmm] [ERROR]
devmm_pin_user_pages_fast <VLLM::Worker_TP:3068957>
Get_user_pages_fast fail. expected_page_num=512; real_got_page_num=0

Aug 03 09:30:02 [ascend] [devmm] [ERROR]
devmm_pin_user_pages_fast <VLLM::Worker_TP:3069323>
Get_user_pages_fast fail. expected_page_num=512; real_got_page_num=0

Aug 03 09:30:02 [ascend] [devmm] [ERROR]
devmm_get_non_svm_addr_pa_list <VLLM::Worker_TP:3068957>
Get user pages fail. ret=-14; size=0x200000; num=513

Aug 03 09:30:02 [ascend] [devmm] [ERROR]
devmm_get_non_svm_addr_pa_list <VLLM::Worker_TP:3069323>
Get user pages fail. ret=-14; size=0x280000; num=641

Aug 03 09:30:02 [ascend] [devmm] [ERROR]
devmm_make_host_pa_node_list <VLLM::Worker_TP:3068957>
Get pa list failed. pin_flg=3
```

`3068957` 和 `3069323` 正是上次 Prefill 的两个 TP Worker 宿主机 PID。
`ret=-14` 对应 Linux `EFAULT`，说明驱动尝试固定用户态虚拟地址页时，没有取得
预期页。日志明确涉及 devmm、page pinning 和 non-SVM 地址页列表。

本窗口未发现独立的 SMMU 或 IOMMU fault 记录。

## 4. 与加载停滞的时间关系

需要避免过度归因：

```text
01:03:49 UTC  模型日志停在 safetensors 0/9
01:03-01:29  两个 TP 主线程持续内核态忙转，13700 未监听
01:30:02 UTC  执行暂停和 Pod 删除
01:30:02 UTC  内核记录 devmm pin_user_pages 错误
01:30:03 UTC  Pod veth 注销
```

devmm 错误发生在 Pod 终止时刻，而不是日志首次停滞的 `01:03:49 UTC`。目前可以
客观确认：

1. 两个 TP Worker 的 Ascend 页固定操作确实失败；
2. 错误与进程退出/设备内存释放同时发生；
3. 它可能是加载问题暴露出的最终错误，也可能是进程被终止时的释放竞态；
4. 仅凭当前日志，不能证明它是 26 分钟加载停滞的最初原因。

但它满足既定门禁，所以继续重启会掩盖现场并违反本轮约束。

## 5. 模型存储和共享内存

模型宿主机目录：

```text
/home/admin/models/Qwen3.6-27B-w8a8
size: 34G
filesystem: /dev/nvme0n1p5, ext4
mount options: rw,relatime,seclabel,stripe=32
```

映射到目标容器后：

```text
/models/Qwen3.6-27B-w8a8
source: /dev/nvme0n1p5[/admin/models/Qwen3.6-27B-w8a8]
filesystem: ext4
mount options: ro,relatime,seclabel,stripe=32
```

它是本地 NVMe EXT4，不是 NFS、Lustre 或对象存储 FUSE。因此没有证据表明上次
0/9 停滞源于网络文件系统。

共享内存：

```text
A3 host /dev/shm: 1007G total, 约 3.9M used
目标 Pod /dev/shm: 64G total, 0 used（空容器采集）
```

共享内存容量没有显示耗尽，但这不能排除 devmm 对普通匿名页进行 pin 时失败。

## 6. 软件版本

从目标镜像采集：

```text
Python: 3.12.13（镜像路径信息）
torch: 2.10.0+cpu
torch_npu: 2.10.0
safetensors: 0.8.0
transformers: 5.5.4
vLLM: 0.22.1+empty
vLLM-Ascend: 0.22.1rc1
```

Ascend 主机工具：

```text
npu-smi: 26.0.rc1
```

`torch` 的 `+cpu` 是主 PyTorch wheel 的版本标记，Ascend 后端由同版本
`torch_npu 2.10.0` 提供，不表示上次 vLLM 使用 CPU 推理。

本镜像的 `vllm --version` 会加载 Ascend 平台插件，不是纯元数据查询。在无 NPU
设备的只读容器中会因 `libascend_hal.so` 缺失失败；挂载驱动后又会执行设备探测。
本轮避免让版本命令演变为新的服务初始化，最终版本取自同一镜像的 Python package
metadata。曾短暂创建的诊断容器已自动删除，物理 2-7 HBM 保持基础占用且无进程。

## 7. 当前资源状态

```text
PD Deployment: replicas=0
PD Pod: 不存在
Prefill/Decode/Proxy: 未运行
物理 Phy-ID 2-7: 无进程，约 2.9-3.2 GiB 基础 HBM
物理 Phy-ID 8-15: 原有 vLLM 保持运行
```

本轮没有修改原 Deployment 模板，也没有创建 Prefill-only Deployment。

## 8. 原始证据

项目归档目录：

```text
markdowns/raw/diagnosis-20260803/
├── qwen36_pd_kernel_full_utc.log
├── qwen36_pd_kernel_faults.log
├── qwen36_pd_kernel_logs.sha256
├── qwen36_pd_storage.txt
├── qwen36_pd_software_versions.txt
└── qwen36_pd_npu_baseline.txt
```

核心日志 SHA256：

```text
a13283996fb51b8ac97bbce32f9dc33588d825d148c5eeb3486685cfdc2ddb1f  qwen36_pd_kernel_full_utc.log
2dfedca667339cf28fcdd791d0ad1bc6e3ac7b0012199b3a7072356f43fbb396  qwen36_pd_kernel_faults.log
```

A3 上的只读副本仍保存在：

```text
/tmp/qwen36_pd_kernel_full_utc.log
/tmp/qwen36_pd_kernel_faults.log
/tmp/qwen36_pd_kernel_logs.sha256
/tmp/qwen36_pd_storage.txt
/tmp/qwen36_pd_software_versions.txt
/tmp/qwen36_pd_npu_baseline.txt
```

## 9. 后续建议（本轮不执行）

在下一次启动前，建议先围绕 devmm pinning 做专项确认：

1. 核对驱动 `26.0.rc1`、CANN、`torch_npu 2.10.0` 和 vLLM-Ascend
   `0.22.1rc1` 的官方兼容矩阵及已知 devmm 问题。
2. 检查宿主机 locked-memory、memlock、透明大页、NUMA 和容器内存锁限制。
3. 获取驱动侧更完整的 devmm/plog，确认被 suppression 隐藏的 4 条回调。
4. 明确错误是否只在强制终止时出现，再决定是否允许一次普通 TP2 对照。
5. 只有重新通过内核门禁后，才应用非 login readiness probe 和
   `--safetensors-load-strategy prefetch`。

当前建议状态：**保持暂停，不重启。**
