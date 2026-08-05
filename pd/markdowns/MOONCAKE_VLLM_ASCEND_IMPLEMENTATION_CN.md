# Mooncake Transfer Engine、vLLM 与 Ascend 实现深析

> 本文从内存地址、传输描述符和 vLLM Scheduler/Worker 接口解释当前
> Ascend 910 PD 实现。概念与完整系统架构见：
> [PD 分离与 Mooncake 系统架构](./PD_DISAGGREGATION_AND_MOONCAKE_ARCHITECTURE_CN.md)

## 0. 版本和证据范围

当前实验环境：

```text
Model        Qwen3.6-27B-w8a8
vLLM        0.22.1
vLLM-Ascend 0.22.1rc1-a3
Ray         2.48.0
Driver      26.0.rc1
CANN        9.0.0（由启动日志确认路径）
Image       110.120.0.3:8889/infra/
            qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730
```

本文的 vLLM 调用链以 vLLM `v0.22.1` 的
`MooncakeConnectorV1` 为主要证据。Mooncake 和 vLLM-Ascend `main`
文档用于解释设计方向；上游主分支可能比实验镜像包含更多能力，不能把
主分支新增功能直接当成本镜像已支持功能。

## 1. Transfer Engine：从虚拟地址到远端写入

### 1.1 Segment 不是一块预分配的大内存

Mooncake Transfer Engine 的第一抽象是 Segment。对 RAM Segment，可将其
理解为：

```text
Segment = 一个进程可对外发布的逻辑地址空间
Buffer  = Segment 中真正注册、允许传输的连续地址区间
```

一个进程启动 Transfer Engine 后拥有一个以 `local_server_name` 标识的
RAM Segment。它逻辑上可以覆盖进程地址空间，但远端不能任意访问该进程；
只有通过 `registerLocalMemory()` 或批量接口注册的 Buffer 才会出现在
可传输元数据中。

Buffer 元数据至少要表达：

```text
virtual address
length
memory device / NUMA affinity
transport permissions
remote key or transport-specific handle
preferred network path
```

这使 Transfer Engine 可以统一处理 CPU DRAM、GPU/NPU HBM 和部分持久化
存储，而应用仍然使用“地址 + 长度”描述数据。

![Transfer Engine](https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/image/transfer-engine.png)

### 1.2 为什么内存必须注册

普通 `torch.Tensor.data_ptr()` 只在本进程虚拟地址空间有意义。要进行
RDMA、HCCS 或设备直接传输，还需要：

- 验证地址属于合法设备；
- 锁定或建立设备内存映射；
- 创建远端访问所需的 Key/Handle；
- 发布地址范围和协议元数据；
- 建立地址与首选 NIC/设备链路的关联。

内存注册通常比一次 memcpy 昂贵，所以 vLLM 在 KV Cache Pool 建立后注册
整片 KV Tensor，而不是每个请求临时注册若干 Block。

### 1.3 vLLM 如何注册 Paged KV Cache

vLLM Worker 得到的结构近似为：

```python
kv_caches: dict[layer_name, torch.Tensor]
```

`MooncakeConnectorWorker.register_kv_caches()` 对每个唯一底层 Tensor：

```python
base_addr = cache.data_ptr()
num_blocks = cache.shape[0]
block_len = cache.stride(0) * cache.element_size()
registered_len = num_blocks * block_len
```

然后执行：

```python
engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
```

这里使用 `stride(0)` 而非 `cache[0].numel() * element_size()` 非常关键。
某些 Backend 为了对齐 Kernel、DMA 或 MLA 数据，会在相邻 Block 之间加入
Padding。`stride(0)` 表示两个 Block 起始地址的真实字节距离，可以覆盖最后
一个 Block 之前的全部 Padding。

若错误地使用逻辑 shape 计算 Block 长度，会出现两类危险：

- 目标 Block 地址逐渐偏移，后续 Block 写入错误位置；
- 注册范围小于真实访问范围，传输层拒绝或越界。

### 1.4 Block 地址如何计算

假设某层 KV Tensor 注册地址为 \(B\)，Block 字节跨度为 \(S\)，请求使用
Block ID \(i\)，则该 Block 起始地址为：

\[
A_i = B + i \times S
\]

对一个请求，P 和 D 的 Block ID 通常不同：

```text
logical token range       P block          D block
0..block_size-1              17                3
block_size..2*block_size-1   18                9
2*block_size..               42               10
```

Connector 最终构造多组描述符：

```text
src_ptrs = [P_layer0_block17, P_layer0_block18, ...]
dst_ptrs = [D_layer0_block3,  D_layer0_block9,  ...]
lengths  = [bytes, bytes, ...]
```

如果源、目标 Block 都连续且布局一致，vLLM 0.22.1 会把相邻 Block 合并成
一个较大的描述符，降低 Python、ZMQ、Transfer Engine 和底层队列的固定开销。

### 1.5 BatchTransfer

BatchTransfer 不是 Collective。它更接近一组异步 Scatter/Gather Copy：

```text
WRITE remote_segment:
  local[src_0 : src_0+n_0] → remote[dst_0 : dst_0+n_0]
  local[src_1 : src_1+n_1] → remote[dst_1 : dst_1+n_1]
  ...
```

这适合 Paged KV，因为一个请求的 Block 在 Pool 中不保证连续；不同 Layer
也对应不同注册 Tensor。

在当前 vLLM Connector 中，P Worker 调用：

```python
batch_transfer_sync_write(
    remote_session,
    src_ptrs,
    dst_ptrs,
    lengths,
)
```

`sync` 表示该调用在线程中等待本批传输完成，不代表整个 vLLM Engine
同步阻塞。Connector 使用独立 Async Loop、Sender Task 和 ThreadPool，
把传输从主模型 Forward 路径移开。

### 1.6 拓扑感知与多链路

通用 Transfer Engine 会建立 Memory Device 与 NIC 的拓扑矩阵。例如：

```text
cuda/npu:0 → preferred NIC A → secondary NIC B
cpu:0      → preferred NIC A → secondary NIC C
cpu:1      → preferred NIC C → secondary NIC A
```

大传输可以切成 Slice，经多个 NIC 并行发送。链路失败时，Endpoint 从 Pool
移除，后续传输可以重建连接或选择备用路径。

![Topology Matrix](https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/image/topology-matrix.png)

需要注意，当前实验是同一 A3 Pod 内的不同 NPU。是否实际使用 HCCS、RoCE
还是其他 Ascend Transport 路径，必须通过 Mooncake/Ascend Transport 日志
和传输 Benchmark 证明，不能仅根据“同机”推断。

## 2. vLLM `MooncakeConnectorV1` 的内部结构

### 2.1 为什么 Connector 分成 Scheduler 和 Worker 两半

vLLM V1 Engine 中：

- Scheduler 掌握 Request 状态、Token 数、Block Allocation；
- Worker 掌握真实 KV Tensor、NPU Device 和内存地址。

因此 Connector 也分为两半：

```text
MooncakeConnectorScheduler
  - 判断请求是否需要远端 KV
  - 参与 Block Allocation
  - 生成本轮 Worker Metadata
  - 决定是否延迟释放 Block

MooncakeConnectorWorker
  - 初始化 Transfer Engine
  - 注册 KV Tensor
  - 建立 Bootstrap、ZMQ 和发送线程
  - 生成地址描述符并执行传输
  - 向 Scheduler 返回完成集合
```

`MooncakeConnector` 外层类只负责把 vLLM 标准 KV Connector Hook 转发给
对应一半。

### 2.2 Scheduler 如何把 Prompt 标记为远端已计算

D 请求携带：

```json
{
  "do_remote_prefill": true,
  "remote_engine_id": "...",
  "remote_bootstrap_addr": "http://...",
  "transfer_id": "..."
}
```

`get_num_new_matched_tokens()` 返回：

```text
len(prompt_token_ids) - num_computed_tokens
async_load = true
```

这并不意味着 KV 已经到达，只是告诉 Scheduler：

> 这些 Prompt Token 不需要在 D 本地重新 Prefill，但执行 Decode 前必须
> 等待外部 KV 加载完成。

随后 Scheduler 为这些 Token 分配本地 Block，并把请求置于类似
`WAITING_FOR_REMOTE_KVS` 的状态。只有 Worker 报告接收完成后，请求才可进入
正常执行队列。

### 2.3 Connector Metadata 是 Scheduler 到 Worker 的桥

Scheduler 不能直接传 Tensor 地址，只构造：

```text
request_id
transfer_id
local_block_ids
remote_engine_id
remote_bootstrap_addr
```

Worker 已经持有注册 Tensor 的 `base_addr` 和 `block_len`，它把二者结合，
才得到可交给 Transfer Engine 的绝对地址。

这种分层还保证 Scheduler 不必依赖 CUDA/NPU Context，Worker 也不必复制
完整 Request 对象。

### 2.4 Bootstrap Server 解决什么问题

Prefill Engine 的 Global Rank 0 Worker 启动一个轻量 HTTP Bootstrap：

```text
POST /register
GET  /query
```

每个 Prefill Worker 注册：

```text
engine_id
dp_rank
tp_rank
pp_rank
worker_addr
```

形成：

```text
engine_id
└── dp_rank
    └── tp_rank
        └── pp_rank → ZMQ worker address
```

D 从 Proxy 获得 `remote_bootstrap_addr` 后查询这个表，定位需要联系的 P TP
Rank。Bootstrap 只传 Worker 地址和拓扑，不传 KV 数据。

### 2.5 ZMQ Side Channel 传什么

D 为请求分配 Block 后，向对应 P Worker 的 ZMQ Listener 发送
`MooncakeXferMetadata`：

```text
D hostname / Transfer Engine RPC port
D TP size / TP rank
request → transfer_id
D local block IDs
D KV Tensor base addresses
D per-layer block lengths
```

换言之，D 主动把“请把这些逻辑 KV 写到我的这些地址”告诉 P。

P 收到元数据后，要等待两件事同时成立：

1. Prefill 请求已经完成，源 Block ID 已知；
2. D 已经分配目标 Block，目标地址元数据已知。

`SendBlockMeta.ready` 等同步结构用于汇合这两条异步路径。

### 2.6 数据面为什么由 P 执行 Write

元数据汇合后，P：

1. 打开 D 的 Remote Segment/Session；
2. 根据 P/D TP Rank 计算源和目标 Region；
3. 展开每层、K/V 和每个 Block 的描述符；
4. 调用 `batch_transfer_sync_write()`；
5. 通过 ZMQ 返回成功或失败 Request 集合。

因此端到端路径是：

```text
D Scheduler：我需要远端 KV
  ↓
D Worker：这是我的目标地址
  ↓ ZMQ
P Worker：源 KV 已就绪
  ↓ Mooncake WRITE
D registered KV memory
  ↓ completion
D Scheduler：允许 Decode
```

### 2.7 完成通知与 Block 生命周期

P Scheduler 在 Prefill 请求结束时调用 `request_finished()`。若请求需要
远端 Decode，它返回 `delay_free_blocks=true`。

P Worker 维护：

```text
reqs_need_send
finished_sending_reqs
```

D Worker 维护：

```text
reqs_to_recv
finished_recving_reqs
```

Engine 每轮通过 `get_finished()` 取走完成集合：

- P 收到“发送完成”后才能释放源 Block；
- D 收到“接收完成”后才能解除 Remote KV Barrier。

若 D 从未出现，P 不能永久占着 Block。vLLM 使用
`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` 为未完成传输设置过期时间。

### 2.8 TP Rank 映射

相同 TP 最简单：

```text
P TP0 → D TP0
P TP1 → D TP1
```

当 TP 不同，Connector 必须知道 KV 在源端是切分还是复制。

P TP 大于 D TP 时，一个 D Rank 可能需要接收多个 P Rank 的 Slice：

```text
P TP0 ┐
      ├→ D TP0 的不同目标偏移
P TP1 ┘
```

D TP 大于 P TP 时，一个 P Rank 可能需要把同一份或不同 Slice 写入多个 D
Rank。只有 TP 比例为整数、KV Layout 和复制策略可证明兼容时，地址计划才是
确定的。

vLLM 0.22.1 还会为非 MLA 场景要求 HND KV Layout，以降低异构 TP 下布局
不一致的风险。不能只让 `prefill.tp_size` 和 `decode.tp_size` 在 JSON 中
看起来合理；真实 Engine 的 TP/DP 配置必须与该声明一致。

## 3. Ascend 数据路径

### 3.1 分层关系

在 Ascend 上，各组件所处层次如下：

```text
vLLM Scheduler / Paged KV Cache
            │
MooncakeConnectorV1
            │ addresses + descriptors
Mooncake Transfer Engine
            │ transport selection
Ascend Transport / Ascend Direct Transport
            │
HIXL / ADXL / HCCL runtime
            │
HCCS or RoCE / device memory operation
            │
Ascend 910 HBM
```

HCCL 主要负责 TP 等模型并行通信；Mooncake 负责 P/D KV 地址传输。两者都
可能使用相同底层网络能力，但语义不同：

- TP HCCL 通信是模型一次 Forward 内的 Collective/P2P；
- Mooncake 传输发生在两个独立 Engine 的 KV Cache Pool 之间。

### 3.2 Ascend Transport 与 Direct Transport

Mooncake 上游提供两类相关实现：

- Ascend Transport：基于 HCCL 能力，自动判断 HCCS/RoCE；
- Ascend Direct Transport：基于 CANN ADXL/HIXL，支持 H2D、D2H、D2D。

Direct Transport 的关键运行参数包括：

```text
ASCEND_AUTO_CONNECT
ASCEND_USE_ASYNC_TRANSFER
ASCEND_CONNECT_TIMEOUT
ASCEND_TRANSFER_TIMEOUT
ASCEND_THREAD_POOL_SIZE
ASCEND_BASE_PORT
HCCL_INTRA_ROCE_ENABLE
```

这些是上游能力列表，不代表当前镜像已经逐项开启。实验时应先记录
Transfer Engine 初始化日志中的实际 Protocol，再做单变量调参。

### 3.3 HCCS 与 RoCE

同一 A3 超节点内通常具有 HCCS 设备互联。Mooncake Ascend Transport 可以
根据拓扑选择 HCCS；设置 `HCCL_INTRA_ROCE_ENABLE=1` 则可能要求节点内走
RoCE。

研究时不能只看带宽峰值：

- HCCS 可能有更低延迟和更少网络配置；
- RoCE 便于跨节点统一，但受 TC/SL、重传、交换网络影响；
- 小 Block 的固定开销可能主导；
- 大 Block 才能充分利用链路；
- TP2 的两个 Rank 会形成多个并行数据流。

### 3.4 内存对齐与 Buffer Pool

Ascend HCCS 注册设备内存可能要求 2 MiB Page Alignment。若实际 Tensor
地址或范围不满足直接传输条件，可使用中间 Buffer Pool，但会引入：

```text
HBM source → aligned staging buffer → transport → destination
```

这失去一部分 Zero-copy 收益。验证时应观察：

- 是否启用了中间 Buffer；
- HBM 到 Buffer 的拷贝时间；
- Transport 时间；
- 接收端是否还有二次拷贝。

“Mooncake 支持 Zero-copy”是能力上限，不是所有配置下的自动事实。

### 3.5 设备编号

宿主机 `npu-smi` 的 Phy-ID、设备插件资源名和容器内 Logical-ID 是三套命名：

```text
Kubernetes annotation: Ascend910-2
Host Phy-ID:           2
Container Logical-ID:  可能是 0，也可能仍是 2
/dev/davinci*:         取决于设备插件挂载方式
```

因此启动脚本先读取 `npu-smi info` 和 `/dev/davinci*`，构造
Phy-ID → Logical-ID 映射，再设置：

```text
ASCEND_VISIBLE_DEVICES
ASCEND_RT_VISIBLE_DEVICES
```

Mooncake、vLLM 和 HCCL 最终必须看到同一 Logical Device。错误映射可能表现
为模型加载到错误卡、Transfer Engine 注册错误 HBM，甚至与宿主机任务抢卡。

### 3.6 控制面端口与数据面端口

当前系统同时存在：

| 通道 | 用途 |
|---|---|
| Proxy `8080` | 客户端 OpenAI API |
| P/D `13700-13702` | Proxy 到 vLLM HTTP |
| `kv_port` | Connector/传输拓扑配置 |
| Bootstrap HTTP | 查询 P Worker Rank 地址 |
| ZMQ Side Channel | D 向 P 发送目标 KV 元数据 |
| Transfer Engine RPC | Segment/连接握手 |
| HCCL/HIXL 端口 | 实际设备通信 |
| Ray GCS `6379` | Ray Worker 注册 |

HTTP Proxy 环境变量不得代理本地和集群内部地址。否则可能发生一种迷惑状态：
HCCL 初始化正常，但 Bootstrap HTTP 或 Proxy→vLLM 请求走代理后失败。

## 4. 当前 1P2D 实现

### 4.1 物理布局

计划拓扑：

```text
Prefill
  physical NPU 2,3
  TP=2
  HTTP=13700
  kv_role=kv_producer
  kv_port=36000

Decode A
  physical NPU 4,5
  TP=2
  HTTP=13701
  kv_role=kv_consumer
  kv_port=36100

Decode B
  physical NPU 6,7
  TP=2
  HTTP=13702
  kv_role=kv_consumer
  kv_port=36200

Proxy
  HTTP=8080
```

三个 vLLM 进程各自加载完整模型；每个进程内部使用 TP2。它不是 TP6：

```text
3 model replicas × TP2
```

P 和 D 之间不共享权重，也不共享 Scheduler。它们只通过控制元数据和 KV
Transfer 形成一次逻辑请求。

### 4.2 当前 KV 配置

启动脚本为每个服务构造：

```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_role": "kv_producer or kv_consumer",
  "kv_port": "service-specific",
  "kv_connector_extra_config": {
    "prefill": {
      "dp_size": 1,
      "tp_size": 2
    },
    "decode": {
      "dp_size": 2,
      "tp_size": 2
    }
  }
}
```

这里存在一个必须在 smoke 中验证的边界：

> Decode A 和 Decode B 是两个独立 `vllm serve`，启动命令没有显式配置
> 一个统一的 `--data-parallel-size=2`、DP Rank 和 DP RPC 组。

因此 JSON 中声明 `decode.dp_size=2` 不足以证明两个 Decode 已经组成 vLLM
Data Parallel Engine。它们可以是 Proxy 管理的两个独立 Decode Replica，
但 Mooncake Connector 如何解释 `dp_size=2` 必须通过 Bootstrap 注册表、
Engine ID、DP Rank 和实际请求验证。正式验证前，不应把它写成“已验证 DP2”。

### 4.3 服务参数为什么不同

当前脚本：

```text
Prefill: max_num_batched_tokens=8192, max_num_seqs=16
Decode:  max_num_batched_tokens=4096, max_num_seqs=64
```

意图是：

- P 用更大的 Token Budget 形成高效 Prefill；
- D 用更多 Sequence 保持 continuous batch。

但 `8192` 也扩大了 Ascend 首次 Compile/Capture 的 Shape 范围。首次启动日志
已经观察到权重加载约 22 秒，但随后图/算子编译持续数十分钟。因此
“权重已加载”不能作为服务 Ready 的证据。

### 4.4 分阶段启动

Entrypoint 顺序为：

```text
发现设备映射
→ 注册 Ray Worker
→ 启动 Prefill 并等待 /health
→ 启动 Decode A 并等待 /health
→ 启动 Decode B 并等待 /health
→ 启动 Proxy
→ 写 READY
```

优点：

- 不让三个实例同时冷编译；
- 能定位具体服务失败；
- 避免模型加载和编译的瞬时内存峰值叠加。

缺点：

- 总冷启动时间可能接近三个实例编译时间之和；
- Prefill 成功后 Decode 失败，已占用的 P HBM 仍需保留；
- 单 Pod 任何主进程退出都可能触发整体重启。

首次编译等待默认已从 1200 秒提高到可配置的 3600 秒：

```text
VLLM_STARTUP_TIMEOUT=3600
```

更好的后续方案是把 CANN/vLLM Compile Cache 挂到持久 Volume，并验证缓存
Key 是否包含模型、vLLM、CANN、Shape 和设备架构。

### 4.5 Ray 的边界

Worker 向既有 Ray Head 注册：

```text
CPU=64
NPU=6
PD_PREFILL=1
PD_DECODE=2
QWEN36_PD_WORKER=1
```

这些是上层调度标签，不建立 TP，也不传输 KV。Ray 的职责是让研究 Job
找到具备该服务能力的节点；vLLM TP 由 HCCL 完成，P/D KV 由 Mooncake 完成。

### 4.6 当前运行状态

本项目镜像、ConfigMap、Service 和 Deployment 已构建。Deployment 当前为：

```text
replicas=0
```

原因不是镜像构建失败，而是 A3 资源冲突：

- 物理 NPU 0-7 被宿主机 `/llamafactory` 8 卡训练占用；
- 物理 NPU 8-15 被既有 vLLM 服务占用；
- 宿主机 Docker 不进入 Kubernetes Device Plugin 资源账本。

一次启动曾完成：

- 镜像拉取；
- 物理 2-7 到容器逻辑设备映射；
- Ray 6 NPU 资源注册；
- Prefill TP2 HCCL 初始化；
- 9 个权重分片加载。

随后检测到训练进程与物理 2、3 冲突，项目立即缩容，没有继续启动 D/Proxy。
因此当前不能声称已经完成端到端 Mooncake PD smoke。

## 5. 如何证明 PD 数据路径真的成立

### 5.1 正确性门禁

按顺序验证：

1. 只启动 1P1D；
2. P/D `/health` 均通过；
3. Bootstrap 中出现 P 的全部 TP Rank；
4. D 请求进入 `WAITING_FOR_REMOTE_KVS`；
5. P 日志出现非零 Descriptor、Bytes 和 Transfer Duration；
6. D 收到全部 TP Rank 完成通知；
7. P Block 在完成前保持占用，完成后释放；
8. D 不执行完整 Prompt Prefill；
9. 输出与普通混部基线在确定性设置下等价；
10. 客户端取消不会留下永久 Block。

### 5.2 数据量交叉校验

对 Prompt 长度 \(L\)，先用模型配置估算理论 KV Bytes。再比较：

```text
P connector reported bytes
Mooncake transfer bytes
D allocated block count × block bytes
```

三者应在考虑 Padding、TP 切分和最后一个未满 Block后相符。

若传输 Bytes 远小于理论值，可能是：

- 只有一个 TP Rank 完成；
- K 或 V 只传了一半；
- Layer 数量不完整；
- D 实际重新 Prefill；
- Prefix Cache 命中改变了应传 Token 数。

### 5.3 时间分解

端到端 TTFT 应拆为：

\[
TTFT =
T_{proxy\_p}
+ T_{queue\_p}
+ T_{prefill}
+ T_{metadata}
+ T_{kv\_transfer}
+ T_{queue\_d}
+ T_{first\_decode}
\]

只有测出每项，才能判断 Mooncake 是否瓶颈。NPU 利用率下降不必然意味着
传输慢，也可能是 Proxy 等待、Bootstrap/ZMQ 延迟、D Block Allocation 或
Scheduler Barrier。

### 5.4 指标归属

建议记录：

| 层次 | 指标 |
|---|---|
| Proxy | P/D 选择、HTTP latency、active request |
| P Scheduler | waiting/running、Prompt Token/s |
| P Connector | descriptor、bytes、transfer latency、failure |
| D Connector | recv complete/error、remote KV wait |
| D Scheduler | waiting/running、Output Token/s、KV usage |
| Mooncake | protocol、RPC/endpoint、transfer status |
| Ascend | 每张卡 AICore、HBM、HCCS/RoCE 指标 |
| System | CPU、RSS、网络、线程池、超时 |

由于当前 Connector 数据面是 P-side write，成功传输指标重点看 P；只看 D
可能误判为“Mooncake 没有传输”。

## 6. 研究实验设计

### 6.1 基线

保持总卡数、模型、量化、Prompt/Output 分布一致：

```text
B0: 普通混部 TP2
B1: 1P1D, TP2 + TP2
B2: 1P2D, TP2 + 2×TP2
```

分别测试：

- 短 Prompt、长 Output；
- 长 Prompt、短 Output；
- 长 Prompt、长 Output；
- 混合长度和突发到达。

### 6.2 主要指标

不要只比较平均 Token/s，应报告：

```text
TTFT p50/p90/p99
TPOT p50/p90/p99
request goodput under SLO
prompt tokens/s
output tokens/s
KV transfer GB/s
KV transfer / TTFT 比例
P/D queue wait
P/D AICore、HBM
失败与取消后的 Block 回收
```

### 6.3 单变量顺序

推荐顺序：

1. 证明 1P1D 正确；
2. 比较 TCP/HCCS 或实际可用 Transport；
3. 调 Prefill `max_num_batched_tokens`；
4. 调 Decode `max_num_seqs`；
5. 加第二个独立 Decode Replica；
6. 验证 Decode Replica 调度；
7. 再考虑 Layer-wise Connector；
8. 最后才评估 Mooncake Store 和跨请求 KV 复用。

不要同时修改 TP、P:D 比例、Batch 和 Transport，否则无法归因。

## 7. 关键风险清单

| 风险 | 后果 | 门禁 |
|---|---|---|
| P/D 模型或 KV dtype 不同 | KV 语义错误 | 启动时比较模型与 Cache Config |
| TP/DP 声明与真实 Engine 不同 | Rank 缺失或地址错误 | 检查 Bootstrap Rank 表 |
| 容器设备编号假设错误 | 抢占其他 NPU | Phy-ID→Logical-ID 动态映射 |
| P Block 提前释放 | 静默错误输出 | 延迟释放和完成集合 |
| 部分 Rank 成功 | 不完整 KV | All-rank completion barrier |
| 代理污染内部 HTTP | 控制面超时 | `NO_PROXY` 和连接测试 |
| Compile 超时重启 | 无限冷启动 | 长启动窗和持久编译缓存 |
| 宿主机 Docker 绕过 K8s | 物理卡冲突 | 启动前后双门禁 |
| 只看 AICore | 错判瓶颈 | 完整 TTFT 分解和队列指标 |

## 参考资料

- [Mooncake Transfer Engine Design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md)
- [Mooncake Ascend Transport](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/ascend_transport.md)
- [Mooncake Ascend Direct Transport](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/ascend_direct_transport.md)
- [vLLM 0.22.1 Mooncake Connector](https://github.com/vllm-project/vllm/tree/v0.22.1/vllm/distributed/kv_transfer/kv_connector/v1/mooncake)
- [vLLM-Ascend PD Design](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md)
- [vLLM-Ascend Mooncake Deployment Guide](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md)
- [本项目 Entrypoint](../scripts/pd-worker-entrypoint.sh)
- [本项目当前状态](./CURRENT_STATUS.md)
- [本项目实施日志](./IMPLEMENTATION_LOG.md)
