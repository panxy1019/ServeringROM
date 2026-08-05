# PD 分离与 Mooncake 系统架构：从计算特征到 KVCache 状态机

> 面向推理系统研究者。本文关注为什么需要 PD 分离、一个请求如何跨越
> Prefill/Decode 边界，以及完整 Mooncake 系统到底由哪些组件构成。
>
> 与本文配套的工程实现篇：
> [Mooncake Transfer Engine、vLLM 与 Ascend 实现深析](./MOONCAKE_VLLM_ASCEND_IMPLEMENTATION_CN.md)

## 0. 阅读边界

本文刻意区分三个经常被混用的概念：

1. **PD disaggregation**：把 Prefill 和 Decode 放到不同推理实例。
2. **Mooncake 完整系统**：包含全局调度、分布式 KVCache 池、存储和传输。
3. **本实验中的 Mooncake**：只使用 vLLM `MooncakeConnectorV1` 和
   Mooncake Transfer Engine 完成本次请求的 P/D KV 传输。

当前实验没有启用 Mooncake Conductor、Mooncake Store、跨请求的全局
KVCache 复用，也没有启用论文中的 Layer-wise Prefill。理解这个边界很重要：
“使用了 Mooncake”不等于“部署了论文中的完整 Mooncake serving system”。

本文以以下上游材料为事实基线：

- [Mooncake FAST'25/arXiv 论文](https://arxiv.org/abs/2407.00079)
- [Mooncake 官方代码库](https://github.com/kvcache-ai/Mooncake)
- [vLLM Disaggregated Serving](https://docs.vllm.ai/en/latest/examples/online_serving/disaggregated_serving/)
- [vLLM-Ascend PD 设计文档](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md)

## 1. PD 分离的计算本质

### 1.1 一次生成为什么天然分成两个阶段

设输入 Prompt 长度为 \(L\)，模型有 \(N\) 层。Prefill 把整个 Prompt
送入模型，为每一层计算 Query、Key、Value，并将 Key/Value 保存下来。
Decode 每轮只输入一个新 Token，但它必须让这个 Token 的 Query 与此前
所有 Token 的 Key/Value 做注意力计算。

因此两阶段的核心区别不是“一个快、一个慢”，而是工作集与计算方式不同：

| 属性 | Prefill | Decode |
|---|---|---|
| 每次新增 Token | 通常是整个 Prompt 或一个大 Chunk | 1 个/序列 |
| 主要输出 | 首个 Token、各层 Prompt KV | 下一个 Token、新增 KV |
| GEMM 形状 | M 维较大，易形成高效大矩阵 | M 很小，需要聚合很多序列 |
| 主要压力 | 算力、长上下文注意力、TTFT | HBM 带宽、调度、TPOT |
| 理想批处理 | 按 Prompt Token 形成大 Token Batch | continuous batching |
| SLO | Time To First Token | Time Per Output Token |

对普通 Transformer，Prefill 注意力部分随输入长度近似呈二次增长，MLP
部分随 Token 数线性增长。Decode 每步只增加一个 Query，但要读取持续增长的
KV 历史；单序列 Decode 往往不能把矩阵计算单元喂满，只能通过并发序列形成
Batch。

### 1.2 KVCache 的容量模型

对标准 MHA/GQA，每个 Token、每层需要保存 K 和 V。忽略对齐和分页开销时：

\[
M_{KV} =
2 \times N_{layer} \times L \times N_{kv\_head}
\times D_{head} \times bytes(dtype)
\]

其中：

- 系数 2 分别代表 K 与 V；
- \(L\) 是已经进入上下文的 Token 数；
- \(N_{kv\_head}\) 是 KV Head 数，不一定等于 Query Head 数；
- \(D_{head}\) 是单个 Head 维度。

例如，模型有 64 层、8 个 KV Head、Head Dimension 为 128、KV 使用 BF16，
则每个 Token 的理论 KV 大小为：

\[
2 \times 64 \times 8 \times 128 \times 2
= 262144\ bytes = 256\ KiB
\]

32K 上下文仅 KV 就约为 8 GiB。真实占用还会受到 Block 对齐、KV Cache
布局、混合注意力、并发序列和预留容量影响。

在 TP 下，KV 是否按 Rank 切分取决于 Attention Backend、KV Head 数和
复制策略。不能简单地用总容量除以 TP：如果 KV Head 少于 TP Rank，某些实现
会复制 KV；这也直接影响 P/D 传输时一个源 Rank 应写入哪些目标 Rank。

### 1.3 混部实例为什么会产生干扰

普通 vLLM 实例同时执行 Prefill 和 Decode。设 Decode Batch 正在为数十个
请求逐 Token 生成，这时一个长 Prompt 到达：

```text
时间 →

Decode:  d d d d | 等待 Prefill | d d d d d
Prefill:         | PPPPPPPPPPPP |
```

Chunked Prefill 可以把 `PPPP` 切成小块插入 Decode Batch，减小单次阻塞，
但它没有消除以下矛盾：

- Chunk 太大，Decode TPOT 抖动；
- Chunk 太小，Prefill GEMM 效率和 TTFT 下降；
- Prefill 与 Decode 争用同一份 HBM、KV Block 和 Scheduler Budget；
- 两类请求需要不同的并行策略、Batch 上限和实例数量；
- 长 Prompt 的突发到达会改变 Decode 尾延迟。

PD 分离的核心价值是**资源和调度目标解耦**，不是 KV 传输本身：

```text
Prefill Pool：最大化 Prompt Token/s，同时约束 TTFT
Decode Pool ：最大化 Output Token/s，同时约束 TPOT
```

代价是必须把 Prefill 已经生成的 KV 从 P 实例交给 D 实例。只有当节省的
干扰成本大于 KV 传输和额外模型副本成本时，PD 分离才有净收益。

### 1.4 应该优化 throughput 还是 goodput

吞吐只统计系统完成了多少 Token；goodput 只统计满足 SLO 的完整请求。
一个系统即便 Token/s 很高，如果长 Prompt 让大量 Decode 请求超过 TPOT
目标，goodput 仍然可能很低。

Mooncake 论文进一步采用严格语义：只有完整完成且满足服务目标的请求才贡献
有效吞吐；若请求最终被拒绝，此前消耗的 Prefill 和 Decode 计算都属于浪费。
这也是全局调度器必须在请求开始前考虑未来 Decode 负载的原因。

## 2. 一条 PD 请求的状态机

### 2.1 不只是两次 HTTP 调用

最简单的描述是“先调用 Prefill，再调用 Decode”，但真正实现必须解决：

- D 在哪里为远端 KV 预留空间；
- P 如何知道 D 的目标地址和 Block 编号；
- TP Rank 如何一一对应；
- 传输期间 P 的 Block 何时可以释放；
- 请求取消或超时时谁负责清理；
- D 如何区分“KV 尚未到达”和“KV 传输失败”。

一个非 Layer-wise 请求可抽象成：

```text
RECEIVED
  ↓
PREFILL_ASSIGNED
  ↓
PREFILL_RUNNING
  ↓
PREFILL_KV_HELD
  ↓
DECODE_BLOCKS_ALLOCATED
  ↓
KV_TRANSFERRING
  ↓
WAITING_FOR_REMOTE_KV
  ↓
DECODE_RUNNING
  ↓
FINISHED
```

关键的不变量是：

> D 只有在目标 KV Block 已分配且全部传输成功后，才能把 Prompt Token
> 视为已经计算；P 只有在所有目标 Rank 完成接收或请求明确失败后，才能释放
> 源 KV Block。

### 2.2 三种 ID 的职责

一个实现通常同时存在以下标识：

| 标识 | 生命周期 | 用途 |
|---|---|---|
| `request_id` | 单个 vLLM Engine 内 | Scheduler 跟踪本地请求 |
| `transfer_id` | P/D 两阶段共享 | 把 P 请求和 D 请求关联起来 |
| `engine_id` | vLLM Engine 生命周期 | 标识 KV 来自哪个远端 Engine |

P 和 D 分别接收到独立 HTTP 请求，本地 `request_id` 不一定相同。
`transfer_id` 是跨阶段关联键；`engine_id` 则用于从 Bootstrap 信息中定位
远端 Engine 下的 DP/TP/PP Worker。

如果只传 `request_id`，多个 Engine 可能产生相同本地 ID；如果只传地址，
Engine 重启后旧地址可能指向错误进程。因此数据传输元数据必须同时表达
“哪次传输、哪个 Engine、哪些 Rank、哪些 Block”。

### 2.3 Proxy 的两阶段路由

非 Layer-wise Proxy 的典型动作如下：

1. 选择一个 P 实例。
2. 向 P 发送原始 Prompt，并设置 `do_remote_decode=true`。
3. 强制 P 只生成最少 Token，使请求在 Prefill 完成后迅速返回。
4. 接收 P 返回的 `kv_transfer_params`。
5. 选择一个 D 实例。
6. 把原始请求和 `kv_transfer_params` 发送给 D。
7. 将 D 的流式输出转发给客户端。

官方 vLLM-Ascend Pull 流程图：

![vLLM-Ascend PD Pull](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/assets/disaggregated_prefill_pull.png)

图中 HTTP 路径只传 Prompt、调度元数据和生成结果。真正的 KV Tensor 不会
编码进 HTTP Response；它通过 Mooncake 数据面直接在注册内存间移动。

### 2.4 为什么 D 必须先分配 Block

vLLM 使用 Paged KV Cache。P 的 Block ID 只是 P 本地 Block Pool 的索引，
对 D 没有意义。D 必须先根据 Prompt Token 数从自己的 Block Pool 分配目标：

```text
P local blocks: [17, 18, 42]
D local blocks: [ 3,  9, 10]
```

随后 D 把目标 Tensor 的 Base Address、每个 Block 的 Byte Stride 和本地
Block ID 发给 P。P 才能计算：

\[
src = P_{base} + P_{block\_id} \times P_{block\_stride}
\]

\[
dst = D_{base} + D_{block\_id} \times D_{block\_stride}
\]

因此“远端 KV 命中”不是直接复用 P 的 Block ID，而是把相同逻辑 Token
对应的 KV 内容写入 D 自己管理的物理 Block。

### 2.5 “D Pull”与“P Push”为什么都正确

vLLM-Ascend 文档把普通 `MooncakeConnector` 描述为 D Pull，这是调度语义：

1. D 声明自己需要远端 KV；
2. D 分配目标 Block；
3. D 向 P 发起传输请求。

但 vLLM 0.22.1 的 Mooncake Worker 数据路径是：

```python
producer.engine.batch_transfer_sync_write(
    remote_session,
    src_ptrs,
    dst_ptrs,
    lengths,
)
```

即真正执行 DMA Write 的是 P。准确表述应为：

```text
控制面：D-triggered pull
数据面：P-side write/push
```

这是理解日志和指标归属的关键。成功传输的 latency、bytes、descriptor
统计通常记录在 P Worker；D 侧主要知道请求何时完成以及是否发生接收错误。

### 2.6 Block 延迟释放与故障语义

Prefill 结束时，如果直接把 Block 归还 Pool，下一请求可能覆盖其中内容，
而 Mooncake 仍在传输旧地址。vLLM Connector 因此让
`request_finished()` 返回“延迟释放”：

```text
Prefill finished
→ 保留源 Block
→ 等待 D 提供目标地址
→ 执行 Write
→ 收到完成/错误
→ Scheduler 才释放源 Block
```

必须考虑四类终止：

- **正常完成**：全部 TP Rank 传输成功；
- **客户端取消**：D 不再需要 KV，P 应停止跟踪并释放；
- **D 失败**：P 等待超过 abort timeout 后释放；
- **P 失败**：D 的请求不能进入 Decode，必须失败或重算，不能使用部分 KV。

“部分 Layer 或部分 Rank 已到达”不能视为成功。错误地让 Decode 使用部分
KV 通常不会产生明确崩溃，而会生成不可解释的错误 Token，因此应 fail-closed。

## 3. 完整 Mooncake 系统

### 3.1 官方架构

![Mooncake Architecture](https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/image/architecture.png)

完整 Mooncake 是 KVCache-centric serving system。它把 GPU/NPU、CPU
DRAM、SSD 和网络视为一个分层 KVCache 资源池，而不是只把 KV 当作某个
Engine 的临时副产物。

主要组件如下。

### 3.2 Conductor：缓存感知的全局调度器

Conductor 维护两类信息：

- 各 Prefill/Decode 实例的负载和可用资源；
- KV Block 在集群中的位置、热度和副本。

选择 Prefill 实例时，它不是单纯挑队列最短者，而是在三个目标之间权衡：

1. 最大化可复用前缀；
2. 平衡 Prefill 负载；
3. 满足 TTFT SLO。

选择 Decode 实例时，更关注预期输出长度、当前 continuous batch 和 TBT
目标。由于 Prefill 需要时间，D 的负载在请求真正抵达时可能已经变化，所以
论文设计还要求 D 本地 Scheduler 再次验证 SLO。

这带来一个重要研究结论：

> PD 路由不是两个独立的最短队列问题，而是带有时间延迟和 KV 数据位置约束
> 的联合调度问题。

### 3.3 Distributed KVCache Pool

完整 Mooncake 把 KV 分成固定 Token Block。一个 Block 的 Key 不只取决于
当前 Token 块，还取决于其全部前缀：

\[
h_i = H(h_{i-1}, tokens_i, model\_identity, cache\_format)
\]

这样只有“当前块相同且前缀也完全相同”时，KV 才能安全复用。仅对当前
Token 文本做 Hash 会错误地复用处于不同上下文的隐藏状态。

KV Pool 可以使用：

- NPU/GPU HBM：容量小、访问最快；
- CPU DRAM：容量大，可通过高速互联靠近加速器；
- SSD：更大，但需要预取和分层策略。

缓存策略可以是 LRU、LFU 或长度/重算成本感知策略。热点 Block 还可生成
多个副本，避免所有请求集中读取单一节点或单一 NIC。

### 3.4 Mooncake Store 与 Transfer Engine 的区别

这两个组件不能互换：

| 组件 | 回答的问题 |
|---|---|
| Mooncake Store | 一个 KV 对象叫什么、在哪里、有几个副本、何时淘汰 |
| Transfer Engine | 已知源地址和目标地址后，如何高效搬运字节 |

Store 提供对象和生命周期语义；Transfer Engine 提供 Segment、内存注册、
地址传输、协议选择和完成状态。只部署 Transfer Engine 时，没有自动获得
跨请求前缀缓存、对象副本或全局淘汰策略。

### 3.5 Messenger/Transfer Engine

论文中的 Messenger 负责 CPU/GPU 间和跨节点的 KV 传输。开源实现中，
Mooncake Transfer Engine 把这类能力抽象成统一的 Segment 与
BatchTransfer，并支持 TCP、RDMA、NVMe-oF、NVLink 以及 Ascend Transport。

![Mooncake Components](https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/image/components.png)

Transfer Engine 不理解“题目”“Token”或“请求语义”；它理解的是注册内存、
远端 Segment、地址、长度、读写方向和完成状态。推理语义由 vLLM Connector
负责翻译。

### 3.6 完整 Mooncake 的请求工作流

论文给出的典型过程为：

1. Tokenize 后，Conductor 找到可复用前缀 Block；
2. 选择 P 和 D；
3. P 从分布式 Cache Pool 加载已有前缀；
4. P 只计算未缓存的增量 Prompt；
5. 新 KV 写回 Cache Pool，并传给 D；
6. D 收齐 KV 后加入 continuous batch；
7. 热点 Block 根据负载被复制或迁移。

论文中的 Layer-wise Prefill 还可以让每层 KV 在该层计算完成后立即传输，
把 KV Store/Transfer 与后续层计算重叠。当前实验采用 Non-layerwise
Connector，必须等 Prefill 请求完成并形成可发送 Block 后再进入传输。

### 3.7 本实验实际启用的 Mooncake 子集

当前实现的组件边界是：

```text
Client
  │ OpenAI-compatible HTTP
  ▼
vLLM-Ascend example Proxy
  ├── HTTP → Prefill vLLM
  └── HTTP → Decode vLLM
                  ▲
                  │ KV memory transfer
         MooncakeConnectorV1
                  │
         Mooncake Transfer Engine
```

已启用：

- `MooncakeConnectorV1`；
- 每个 vLLM Worker 的 KV Cache 内存注册；
- Bootstrap/Side Channel；
- P/D Block 地址映射；
- Mooncake 批量写传输；
- P 侧延迟释放。

未启用：

- Conductor；
- Mooncake Store；
- etcd/Redis 支撑的全局 KV 元数据；
- DRAM/SSD KV 分层；
- 跨请求前缀缓存复用；
- 热点副本与迁移；
- Layer-wise KV 流水传输。

因此当前实验的研究问题应表述为：

> 在 Ascend 单节点 1P2D 中，vLLM 能否通过 Mooncake Transfer Engine
> 正确且高效地把一次 Prefill 产生的 Paged KV Cache 交给独立 Decode
> 实例，并降低混部产生的 TTFT/TPOT 干扰？

而不应表述为“已经验证完整 Mooncake 分布式缓存系统”。

## 4. 第一阶段实验应回答的问题

正式性能比较前，依次证明：

1. P 和 D 使用同一模型、Tokenizer、KV dtype 和 Cache Layout；
2. D 的 Prompt Token 被标记为 remote-computed，而不是重新 Prefill；
3. P/D 每个 TP Rank 都参与了传输；
4. Mooncake 成功传输 bytes 与理论 KV 大小处于同一数量级；
5. P 的 Block 在传输完成前没有被释放；
6. 关闭 Connector 后相同 PD 请求不能“偶然成功”；
7. 1P1D 正确后再加入第二个 Decode；
8. PD 的 TTFT/TPOT goodput 优于相同卡数的混部基线。

第二篇将从源码和地址计算层面对这些验证点展开。

## 参考资料

- Ruoyu Qin et al.,
  [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)
- [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- [Mooncake Transfer Engine Design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md)
- [vLLM Mooncake Connector](https://github.com/vllm-project/vllm/tree/main/vllm/distributed/kv_transfer/kv_connector/v1/mooncake)
- [vLLM-Ascend Disaggregated Prefill Design](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md)
