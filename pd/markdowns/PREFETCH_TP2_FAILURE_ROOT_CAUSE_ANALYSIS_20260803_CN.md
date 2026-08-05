# Prefetch TP2 失败根因分析与解决路线

分析日期：2026-08-03  
关联运行：`prefetch-tp2-20260803T025936Z`  
关联报告：`PREFETCH_TP2_DIAGNOSTIC_REPORT_20260803_CN.md`

## 1. 最简结论

### 2026-08-03 eager 对照后的更新

后续保持所有条件不变、仅使用 `--safetensors-load-strategy eager` 的 TP2 实例已完成
`9/9` shard、Engine 初始化和 `13700/health`。因此 H1 已从“高置信假设”提升为：

```text
此节点上的默认 mmap-backed safetensors 主加载路径会阻断启动；
eager 内存缓冲区加载可以稳定绕开该阻断路径。
```

这仍不能单独定位为某一行 torch-npu、CANN 或 ModelSlim 的源码 defect；它证明的是可复现
的路径级差异。完整证据见 `EAGER_TP2_RESOLUTION_REPORT_20260803_CN.md`。

当前最可能的故障点不是模型下载、磁盘读取、Mooncake、HCCL 初始化、HBM 不足或
版本混装，而是：

```text
safetensors mmap-backed CPU tensor
        ↓
vLLM / ModelSlim 权重切片与 param.data.copy_()
        ↓
torch-npu / CANN 将 CPU 页复制或注册到 NPU
        ↓
持续 minor page fault / mmap-VMA 活动，首个 shard 无法完成
```

`prefetch` 只把文件读进 Linux page cache，随后仍使用 `safe_open()` 和
`get_tensor()` 的 mmap-backed tensor。它没有改变 CPU tensor 到 NPU 参数的复制
方式，所以这次失败并没有真正绕开可疑路径。

第一优先解决办法是进行单变量 `eager` 对照：

```text
--safetensors-load-strategy eager
```

这会用 `open(...).read()` 读取单个 shard，再由 `safetensors.torch.load()` 从内存
缓冲区构造 tensor，绕开默认 `safe_open()` 的文件 mmap tensor。当前 256 GiB Pod
内存足以支撑两个 TP rank 各自按单 shard eager 加载。

## 2. 已经证明的事实

### 2.1 服务卡在权重阶段，不是健康探针假象

两个 TP rank 均完成：

```text
HCCL world_size=2 初始化
物理 NPU 2、3 设备打开
CANN client/server version check
devmm host memory pool 初始化
模型结构和 W8A8 ModelSlim 配置创建
```

之后 15 分钟一直停在：

```text
Loading safetensors checkpoint shards: 0% Completed | 0/9
```

`13700` 没有监听，`/health` connection refused。readiness 已改成 `/bin/sh -c`，
因此不是 login shell 探针拖慢或误判。

### 2.2 权重文件完整且 CPU eager 解析正常

使用同一派生镜像、只读挂载同一模型目录、不申请 NPU，完成了两种检查：

| 检查 | 结果 |
|---|---:|
| `safe_open` 遍历 9 shard/1725 tensors | 通过，约 1.7 秒 |
| 逐 shard `read()` + `safetensors.torch.load()` | 通过，约 21.8 秒 |
| 解析 tensor 数据量 | 36,423,542,240 bytes |

所有 shard 均可读取和反序列化，没有 header、dtype、shape 或文件截断错误。模型位于
本地 NVMe EXT4，不是 NFS、Lustre 或 FUSE。

因此“权重损坏”“网络存储慢”“safetensors 不能解析 bfloat16”不是主要候选。

### 2.3 节点没有 CPU、RAM 或磁盘压力

诊断后的只读复查：

```text
MemAvailable: 约 1.5 TiB
CPU idle: 约 98-100%
I/O wait: 约 0%
model filesystem: local NVMe EXT4
```

诊断期间两个 Worker 各自只占约一个 CPU 核，不是 64 CPU quota 被耗尽。没有
OOM、memory.events、NPU health 异常或 Pod 重启。

### 2.4 卡住进程的特征是持续页故障，不是持续磁盘读取

| 指标 | TP0 | TP1 |
|---|---:|---:|
| system CPU 平均 | 97.78% | 98.26% |
| user CPU 平均 | 2.43% | 3.09% |
| minor fault 平均增量/采样 | 205,138 | 163,927 |
| RSS 峰值 | 3.80 GiB | 3.81 GiB |
| HBM 峰值 | 20.57 GiB | 20.34 GiB |
| AICore | 基本 0% | 基本 0% |

线程现场反复出现：

```text
vm_mmap_pgoff
lock_mm_and_find_vma
__vm_munmap
futex_wait_queue
```

以 4 KiB 页粗略换算，十几万到二十万 minor faults/s 已远高于只顺序触碰 34 GiB
checkpoint 一次所需的页数；持续十几分钟更像重复映射、解除映射、页表查找或某个
CPU→NPU copy 路径上的活锁/退化，而不是正常权重加载。

### 2.5 超大 VmSize 不是独立根因

卡住 Worker 的 VmSize 约 9.68 PB，但节点上正常运行的 8 个 vLLM Worker 同样约
9.72 PB，并且：

```text
maps 数约 6,100-6,300
VmLck=0
Max locked memory=64 MiB
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

正常 Worker 空闲抽样的 minor faults 为 0。由此可得：

1. PB 级虚拟地址保留是当前 Ascend expandable allocator 的可见特征；
2. `vm.max_map_count=65530` 尚未接近耗尽；
3. 64 MiB memlock 和 `VmLck=0` 在同一宿主机上并未阻止其他 vLLM 正常运行；
4. 真正异常的是卡住实例持续制造 page faults，而不是虚拟地址数值本身。

memlock、THP、`vm.max_map_count` 和 SMMU 仍可作为后续兼容性研究对象，但不应在
当前阶段优先修改，否则会引入多变量且缺乏针对性。

### 2.6 派生镜像没有关键依赖漂移

官方基础镜像与派生镜像逐项对比结果一致：

```text
torch-npu       2.10.0
vllm            0.22.1+empty
vllm-ascend     0.22.1rc1
safetensors     0.8.0
transformers    5.5.4
numpy           1.26.4
tokenizers      0.22.2
grpcio          1.81.1
protobuf        7.35.1
```

安装 Ray 的派生层没有升级这些权重加载依赖，因此不需要先重建一个“去 Ray”镜像。

### 2.7 终止时的 devmm 错误不是启动起因

启动到终止前的完整内核窗口没有：

```text
Get_user_pages_fast fail
ret=-14
page pinning error
SMMU/IOMMU fault
```

这些错误只在删除 Pod、终止两个 TP Worker 的同一秒出现。它们说明 teardown 时
驱动仍试图处理已经变化的用户虚拟地址，但无法证明 15 分钟前就是它导致了 0/9。

## 3. `prefetch` 为什么没有解决问题

vLLM 0.22.1 的实现逻辑是：

```python
if strategy == "prefetch":
    # 后台线程顺序 read()，只填充 OS page cache
    _prefetch_all_checkpoints(files)

for shard in files:
    if strategy == "eager":
        state_dict = load(open(shard, "rb").read())
    else:
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)
```

本轮日志显示 TP0 的 5/5 和 TP1 的 4/4 后台预读在数秒内完成，但主 loader 继续
停在 0/9。这符合源码行为：prefetch 和主 mmap loader 并行，prefetch 完成后不会
把主 loader 改成 eager。

所以正确表述是：

```text
已排除“page cache 未预热/磁盘慢”
尚未排除“mmap-backed tensor 到 NPU copy 的问题”
```

## 4. 根因假设排序

### H1：mmap-backed tensor 与 Ascend 权重 copy 路径发生退化或活锁

置信度：**高**。

支持证据：

- CPU eager 读取 36.42 GB 仅约 21.8 秒；
- NPU 服务却在首 shard 内停留 15 分钟；
- system CPU 接近一核，minor faults 持续极高；
- AICore 为 0，说明尚未进入模型算子；
- vLLM 最终通过 `param.data.copy_(loaded_weight)` 把 CPU tensor 写入 NPU 参数；
- `prefetch` 没有改变 `safe_open/get_tensor` 的 mmap-backed tensor。

H1 仍是“最符合证据的路径级根因”，不是已经定位到 torch-npu 或驱动中的某一行。
要把它升级为确定根因，需要 eager A/B 或逐 tensor copy 追踪。

### H2：两个 TP rank 并发加载放大 mmap/VMA 或 NPU host-copy 竞争

置信度：**中**。

两个 rank 同时进入首 shard、同时约一核 system CPU，并在不同物理 NPU 上出现相同
模式。HCCL 已初始化，但加载阶段仍可能在 TP 切片、同步或共享主机内存路径上相互
影响。TP1 对照可以验证。

### H3：`prefetch` 后台线程与主 loader 的并发使问题更容易触发

置信度：**中低**。

prefetch 确实在每个 rank 启动最多 8 个后台读取线程，并立即让主 loader 开始 mmap。
不过后台读取数秒后完成，停滞仍持续十几分钟；此前默认策略也出现过 0/9。因此它
可能是放大因素，不像唯一根因。

### H4：模型/ModelSlim 的特定首 shard tensor 触发 torch-npu copy 问题

置信度：**中**。

第一 shard 包含 478 个 tensor，progress bar 只有整 shard 完成后才从 0 变成 1，
所以当前无法知道卡在第一个 tensor，还是 shard 内的后续 tensor。W8A8 权重映射和
TP 切片发生在 copy 前，某个 shape/dtype/stride 组合可能触发问题。逐 tensor 日志
能够直接确认。

### H5：版本、文件、资源或 Kubernetes 注入错误

置信度：**低**。

完整 release matrix 匹配、CANN client/server version check 成功、设备映射正确、
文件可完整 eager 解析、内存/CPU 充足，也没有 OOM、HCCL 或 Device Plugin 错误。

## 5. 推荐解决顺序

### 第一步：TP2 eager，最高优先级

保持本轮所有条件不变，只改：

```bash
--safetensors-load-strategy eager
```

不要同时加 `--enforce-eager`。这两个 “eager” 含义不同：

- `--safetensors-load-strategy eager` 控制权重文件加载，正对当前 0/9；
- `--enforce-eager` 控制模型执行阶段的图捕获，模型尚未加载完时没有帮助。

验收仍应为：9/9、13700 监听、health 成功。若通过，连续冷启动 3 次，确认不是一次
偶然成功，再把 eager 固化为该模型的加载默认值。之后才逐层加入 Mooncake、Decode
和 Proxy。

预期内存风险可控：vLLM eager 是逐 shard 处理；最大 shard 约 4 GiB。两个 rank
同时加载时会增加 CPU bytes/state_dict 临时内存，但距离 256 GiB limit 很远。

### 第二步：若 eager 仍失败，加入逐 tensor 边界日志

不要立即改内核参数。先在诊断入口中对权重 loader 增加：

```text
rank
shard
tensor name
source shape/dtype/stride/contiguous
目标 param shape/dtype/device
copy 开始时间
copy 返回时间
torch_npu.synchronize 返回时间
```

这样可以回答：

```text
卡在 safe_open/get_tensor？
卡在 TP narrow/reshape？
卡在 param.data.copy_？
还是卡在 NPU synchronize？
```

同时注册 `faulthandler`，定时保存 Python 全线程栈；宿主机继续采集 perf/pidstat、
wchan 和 CANN plog。现有内核栈只能看到 VMA 活动，不能给出 Python tensor 名称。

### 第三步：TP1 eager 对照

如果 TP2 eager 仍失败，在一张空闲卡上运行相同模型 TP1：

- TP1 通过、TP2 失败：重点检查 TP sharding、HCCL 后加载同步和双 rank 并发；
- TP1 也失败：重点检查单 rank ModelSlim tensor copy、torch-npu/driver 页处理；
- TP1 卡在相同 tensor：可以直接构造该 tensor 的最小 NPU copy 复现。

### 第四步：同步 copy 诊断

在已经获得具体 tensor 名称后，下一轮只增加一个同步变量：

```text
ASCEND_LAUNCH_BLOCKING=1
```

或明确关闭 task queue，再在每次关键 copy 后执行 `torch_npu.npu.synchronize()`。
目标不是长期性能配置，而是把异步队列中的挂起变成可定位的同步调用或显式异常。

### 第五步：allocator 单变量对照

只有 eager、逐 tensor 和同步 copy 仍无法解释时，再测试：

```text
移除 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

其优先级较低，因为同一节点正常 vLLM Worker 正在使用相同配置和 PB 级 VmSize。

### 第六步：完整新官方栈对照

若最小 TP1 copy 在当前官方组合上仍可稳定复现，再构造独立的新官方栈镜像。必须
整体匹配新的 vLLM、vLLM-Ascend、torch、torch-npu、CANN 和 triton-ascend，不能
只升级 CANN 或 torch-npu。保留当前镜像作为 A/B 基线，不覆盖生产标签。

## 6. 当前不建议做的操作

暂不建议：

```text
修改 SMMU/IOMMU 启动参数
修改 THP
提高 vm.max_map_count
把 memlock 直接改成 unlimited
重置仍可能被其他任务使用的 NPU
同时升级 CANN 和修改加载参数
直接恢复完整 1P2D
```

原因不是这些参数永远无关，而是现有正常 Worker 已证明当前宿主机参数可以承载
vLLM；在尚未完成 eager 和 TP1 归因前修改它们，收益小且会破坏单变量实验。

## 7. 建议的实验判定矩阵

| 实验 | 唯一变化 | 结果解释 |
|---|---|---|
| E1 | TP2 + eager | 通过则确认 mmap-backed load 是关键条件 |
| E2 | TP2 eager + tensor tracing | 确定具体阻塞 tensor/调用 |
| E3 | TP1 + eager | 区分单卡 copy 与 TP2 并发问题 |
| E4 | 对具体 tensor 同步 copy | 区分 Python loader 与 torch-npu/CANN |
| E5 | allocator 关闭 expandable segments | 检查虚拟地址分配交互 |
| E6 | 完整新版官方矩阵 | 判断当前 release 特定缺陷 |

每轮都应保存终止前现场，并把终止阶段 devmm 日志单独归档。

## 8. 官方实现依据

- vLLM 0.22.1 safetensors loader：
  <https://docs.vllm.ai/en/v0.22.1/api/vllm/model_executor/model_loader/weight_utils/>
- vLLM LoadConfig 对 lazy/eager/prefetch 的定义：
  <https://docs.vllm.ai/en/stable/api/vllm/config/load/>
- vLLM-Ascend v0.22.1rc1 release：
  <https://github.com/vllm-project/vllm-ascend/releases/tag/v0.22.1rc1>

这些来源支持的是加载策略语义和版本边界；关于当前节点的根因排序来自本次现场
证据，是工程推断，不冒充上游已确认缺陷。
