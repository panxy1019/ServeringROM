# 任务二：1P2D 吞吐、时延与 Goodput 实验手册

> 适用环境：`Qwen3.6-27B-w8a8`、vLLM/vLLM-Ascend `0.22.1/0.22.1rc1`、
> Ascend 910C、MooncakeConnectorV1、单 Pod `1P2D`。
>
> 本文不是“跑一个 requests/s 数字”的压测说明。目标是建立可复现的负载模型，
> 区分 Prefill、KV 迁移、Decode 和排队造成的瓶颈，并判断第二个 Decode 是否真正
> 增加了满足 SLO 的有效吞吐。

## 1. 当前实验对象

```text
Client
  -> Proxy :8080
       -> Prefill :13700 -> Phy-ID 2,3 -> TP2
       -> Decode A :13701 -> Phy-ID 4,5 -> TP2
       -> Decode B :13702 -> Phy-ID 6,7 -> TP2
```

三个 vLLM 服务是三个独立 Engine，各自加载一份完整权重：

```text
1 × Prefill replica, TP=2
2 × Decode replica,  TP=2 each
```

当前关键参数：

| 服务 | `max_num_batched_tokens` | `max_num_seqs` | KV blocks |
|---|---:|---:|---:|
| Prefill | 8192 | 16 | 718/rank |
| Decode A | 4096 | 64 | 717/rank |
| Decode B | 4096 | 64 | 717/rank |

三者均为 `gpu_memory_utilization=0.88`、`max_model_len=32768`、
`safetensors-load-strategy=eager`、`prefix_caching=false`。

## 2. 为什么不能只看 output tokens/s

一次请求的端到端时间可以写成：

\[
T_{e2e}=T_{client}+T_{proxy,p}+T_{queue,p}+T_{prefill}
+T_{metadata}+T_{kv}+T_{queue,d}+T_{decode}+T_{proxy,d}
\]

其中：

- `TTFT` 同时受到 Prefill、KV 迁移和 Decode 首步影响；
- `TPOT/ITL` 主要反映 Decode 连续批处理质量；
- request throughput 会被输入、输出长度同时影响；
- total tokens/s 把计算特征完全不同的 input/output token 混在一起；
- 平均值会隐藏长尾和某个 Decode replica 的不均衡。

因此每个实验至少报告：

```text
completed requests/s
input tokens/s
output tokens/s
TTFT p50/p90/p95/p99
TPOT p50/p90/p95/p99
ITL p50/p90/p95/p99
E2E p50/p90/p95/p99
error/retry/cancel count
goodput under SLO
```

Goodput 的定义是：只有完整成功且同时满足指定 TTFT、TPOT、E2E SLO 的请求
才计入吞吐。高 token/s 但大量请求超过 SLO，不应被解释为更高的服务能力。

## 3. 测试纪律

每组实验固定以下条件：

```text
模型、量化和镜像版本
P/D 数量及 TP 大小
max_model_len
max_num_batched_tokens
max_num_seqs
prompt/output 长度分布
采样时长和随机种子
客户端所在节点
Proxy 与后端日志级别
```

执行规则：

1. 模型启动和图编译完成后再开始计时。
2. 每个 shape 先进行至少 16 次不计入统计的 warmup。
3. 每组至少运行 3 次；短实验每次至少 128 个请求。
4. 各组之间等待 running/waiting 回到 0，防止前一组污染下一组。
5. 同时保存 Client 结果、三个 `/metrics` 快照和 NPU 采样。
6. 不在测试中修改 vLLM 参数；参数 A/B 必须重新命名结果目录。
7. 首轮不要启用 Prefix Cache，避免命中率改变实际 Prefill 工作量。

## 4. 准备命令

在 `server-00`：

```bash
export KUBECONFIG=/home/admin/k3s.yaml
export NS=infra-learning
export APP=ray-vllm-pd-worker-qwen36-27b
export POD=$(kubectl -n "$NS" get pod -l app="$APP" \
  -o jsonpath='{.items[0].metadata.name}')

kubectl -n "$NS" get pod "$POD" -o wide
kubectl -n "$NS" exec "$POD" -- curl -fsS \
  http://127.0.0.1:8080/healthcheck
```

期望：

```json
{"status":"ok","prefill_instances":1,"decode_instances":2,"request_num":0}
```

建立结果目录：

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR=/home/admin/testpanxy/infralearning/qwen36_pd_1p2d/results/$RUN_ID
mkdir -p "$RESULT_DIR"
echo "$RESULT_DIR"
```

## 5. 基准命令模板

当前 Worker 镜像内已经包含 `vllm bench serve`。以下命令在 Pod 内运行，适合
实验室初测；正式饱和测试建议将同版本 benchmark client 放在 `server-00`，避免
客户端与三个 Engine 争用 Pod CPU。

```bash
kubectl -n "$NS" exec "$POD" -- vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8080 \
  --endpoint /v1/completions \
  --model /models/Qwen3.6-27B-w8a8 \
  --served-model-name qwen36-27b-w8a8 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 256 \
  --num-warmups 16 \
  --request-rate inf \
  --max-concurrency 8 \
  --seed 1024 \
  --temperature 0 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --goodput ttft:2000 tpot:80 e2el:30000 \
  --save-result \
  --result-dir /tmp \
  --result-filename pd-balanced-c8.json
```

`--ignore-eos` 用于固定输出长度，使不同实验真正可比；它测的是容量上限，不代表
真实用户提前遇到 EOS 时的业务吞吐。

该命令已用 `input=32/output=8/concurrency=2` 做过命令级 smoke：2/2 成功，
`TTFT p50≈554 ms`、`TPOT p50≈25.6 ms`。样本量过小，这些数值只证明工具和
PD Proxy 协议兼容，不代表系统容量。

结果从 Pod 复制出来：

```bash
kubectl -n "$NS" cp "$POD":/tmp/pd-balanced-c8.json \
  "$RESULT_DIR/pd-balanced-c8.json"
```

## 6. 实验矩阵

### 6.1 E0：正确性和冷暖边界

目的：确认所有后续数字来自热服务，而不是图编译或首次连接。

```text
input=128, output=32, concurrency=1, requests=16
```

记录第一次和后续请求的 TTFT。第一次显著慢、后续稳定是 warmup；持续抖动则应
先检查图 shape、KV 连接或请求重派，不能直接进入吞吐实验。

### 6.2 E1：并发饱和曲线

固定：

```text
input=1024
output=256
requests=max(256, 16 × concurrency)
request_rate=inf
concurrency=1,2,4,8,16,32,64
```

每次只改变 `--max-concurrency`。画出：

```text
x = concurrency
y1 = output tokens/s
y2 = TTFT p95
y3 = TPOT p95
y4 = goodput requests/s
```

饱和点不是 output tokens/s 的绝对最高点，而是 goodput 开始下降前的最后一个点。

### 6.3 E2：Prefill-heavy

```text
input=4096, output=16
input=8192, output=16
concurrency=1,2,4,8,16
```

重点看：

- Prefill `prompt_tokens_total` 增速；
- Prefill running/waiting 和 AICore；
- TTFT；
- KV transfer latency；
- Decode A/B 是否大部分时间空闲。

该实验回答“P 是否已经成为唯一瓶颈”。增加 Decode 不会改善 P 饱和造成的 TTFT。

### 6.4 E3：Decode-heavy

```text
input=128, output=512
input=128, output=1024
concurrency=1,2,4,8,16,32,64
```

重点看 Decode A/B 的：

```text
generation_tokens_total delta
num_requests_running/waiting
kv_cache_usage_perc
TPOT/ITL
AICore
```

这个实验最能体现第二个 Decode replica 的价值。

### 6.5 E4：长输入长输出

```text
input=4096, output=512
concurrency=2,4,8,16
```

它同时施压 P、Mooncake 和 D，用于发现流水线背压。不要用它单独定位根因；必须
结合 E2/E3 才能判断瓶颈在哪一段。

### 6.6 E5：开放环到达率

闭环 `request-rate=inf` 回答容量上限，开放环才接近在线流量：

```text
request_rate=0.5,1,2,4,8,... requests/s
burstiness=1.0
max_concurrency=64
```

逐步提高 RPS，直到：

```text
waiting 持续非零
TTFT p95 超过 SLO
goodput 不再增长
```

`burstiness=1` 对应 Poisson 到达。再用更突发的 Gamma 到达验证弹性，但不要把
突发实验与稳态容量混为同一个结论。

### 6.7 E6：调度器反例

当前 Proxy 使用 `len(req_body)`，即 HTTP body 字节数估算 P/D 负载：

```python
prefill_score = (body_bytes / 4) * 0.0345 + 120.0745
decode_score = body_bytes
```

它没有使用真实 tokenizer tokens，也没有把 `max_tokens` 纳入 Decode 负载。
构造两组输入字节数接近但输出长度不同的并发请求：

```text
A: input≈256 tokens, output=32
B: input≈256 tokens, output=2048
```

验证两个 Decode 的 `active_tokens` 账面分配是否接近，而实际运行时间、
generation token 增量和 TPOT 是否严重不对称。这是研究当前示例 Proxy 调度质量
最有价值的实验之一。

### 6.8 E7：30 分钟稳态

选择 E1 得到的 goodput 最优并发，运行至少 30 分钟。检查：

```text
output tokens/s 是否漂移
TTFT/TPOT p99 是否随时间增长
KV cache 是否回落
request_num 是否归零
Prefill delayed-free block 是否最终释放
Mooncake transfer error 是否累计
Pod RSS/HBM 是否单调增长
```

## 7. 1P1D 与 1P2D 的正确比较

`1P1D` 使用 4 个逻辑 NPU，`1P2D` 使用 6 个，因此不能只比较总吞吐。

至少报告：

\[
资源归一化吞吐 = \frac{output\ tokens/s}{NPU\ count}
\]

以及 Decode 扩展效率：

\[
E_D = \frac{T_{1P2D}/T_{1P1D}}{4/2}
\]

其中分母 `4/2` 是 Decode NPU 从 2 增至 4。若 `E_D≈1`，Decode-heavy 场景接近
线性扩展；若 `E_D` 很低，瓶颈可能在单 Prefill、KV 迁移、Proxy 或调度不均衡。

预期关系：

- 1P2D 的 Prefill 能力与 1P1D 基本相同；
- Decode-heavy 吞吐应该增长；
- Prefill-heavy 的吞吐不应因多一个 Decode 显著增长；
- 低并发串行请求可能全部落到 Decode A，无法证明 B 的扩展价值。

## 8. 服务端指标采样

实验前后各保存一次完整指标：

```bash
for port in 13700 13701 13702; do
  kubectl -n "$NS" exec "$POD" -- curl -fsS \
    "http://127.0.0.1:$port/metrics" \
    > "$RESULT_DIR/metrics-$port-$(date -u +%H%M%S).prom"
done
```

在线观察：

```bash
watch -n 1 "kubectl -n $NS exec $POD -- sh -c '
for p in 13700 13701 13702; do
  echo ===\$p===
  curl -s http://127.0.0.1:\$p/metrics |
    grep -E \"^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|prompt_tokens_total|generation_tokens_total)\"
done'"
```

计数器必须用时间差分：

\[
rate(counter)=\frac{counter(t_2)-counter(t_1)}{t_2-t_1}
\]

不要把 `*_total` 当前累计值直接当吞吐。

## 9. 判断规则

| 现象 | 主要解释 | 下一步 |
|---|---|---|
| P waiting 高，D 基本空闲 | Prefill 饱和 | 提升 P 能力或减少 D 比例 |
| P 空闲，D waiting 高 | Decode 饱和 | 增加 D 或调 Decode batch |
| P/D 都不忙，TTFT 高 | Proxy、KV、CPU 或同步等待 | 做任务三请求级时间线 |
| D-A 忙、D-B 空闲 | 并发不足、tie-break 或负载估计失真 | 用并发和路由指标验证 |
| KV usage 高且 waiting=capacity | KV 容量压力 | 降并发/上下文或增加 D |
| output tok/s 增长但 goodput 降低 | 用尾延迟换吞吐 | 采用前一档并发 |
| HBM 稳定但 AICore 低 | 请求供应、通信或 CPU gap | 检查 running/waiting 和传输时间 |

## 10. 结果表模板

| Run | 拓扑 | 输入/输出 | 并发/RPS | req/s | input tok/s | output tok/s | TTFT p95 | TPOT p95 | E2E p95 | goodput | error |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1-C1 | 1P2D | 1024/256 | C=1 | | | | | | | | |

每个结论必须同时关联：

```text
benchmark JSON
P/D metrics before/after
NPU sample
Proxy/P/D logs
镜像和启动参数快照
```

## 11. 最小推荐执行顺序

```text
E0 warmup
-> E1 并发扫描
-> E2 Prefill-heavy
-> E3 Decode-heavy
-> E6 调度反例
-> E5 开放环 SLO
-> E7 稳态
-> 必要时再做 1P1D/1P2D A/B
```

这个顺序先测容量形状，再解释根因，最后验证长期稳定性。不要在第一轮同时修改
`max_num_seqs`、batch tokens、图模式和实例数，否则得到的只是一个不可归因的数字。

## 参考

- [vLLM `bench serve` 官方参数](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [vLLM-Ascend Mooncake PD 官方教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html)
- [vLLM 指标说明](https://docs.vllm.ai/en/latest/design/metrics/)
