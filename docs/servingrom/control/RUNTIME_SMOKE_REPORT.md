# ServingROM Control-v1 Runtime Smoke 报告

## 1. 结论

Step 13B runtime smoke 通过。A1 `decode_routing_ratio` 在不重启 Pod/Engine、不重载模型、不重建 Mooncake、不改变 D2 graph/async 配置的条件下真实改变了新请求的 Decode 分配。

```text
actuator_ready=true
control_excitation_ready=true
control_rom_ready=false
mpc_ready=false
Control-v1 U dimension=1
u=rho_A
safe range=[0.2, 0.8]
minimum dwell=5s
max delta=0.2
```

## 2. 冻结运行身份

- Deployment：`ray-vllm-pd-control-v1-qwen36-27b`
- Pod：`ray-vllm-pd-control-v1-qwen36-27b-5c58f55476-tptn8`
- Pod UID：`0649a4ff-c87a-4a6c-ba07-9ea42c4d53de`
- container restart：0
- container startedAt：`2026-08-10T02:33:27Z`
- Prefill：PID 614，start ticks 259889383，物理 NPU 10/11，TP2
- Decode A：PID 2863，start ticks 259907434，物理 NPU 12/13，TP2
- Decode B：PID 5687，start ticks 259929497，物理 NPU 14/15，TP2
- Proxy：PID 8228，start ticks 259952060

前后 Pod UID、startedAt 与 restartCount 一致。每个 Engine 只有一个 EngineCore initialization 和一个 API server start；日志中的两条 model load start 分别来自 TP0/TP1，不是模型重载。

Decode A/B 的 effective config 均为：

```text
--async-scheduling
--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
--max-num-batched-tokens 4096
--max-num-seqs 64
--tensor-parallel-size 2
```

## 3. 实验序列

实际执行：

```text
baseline
-> 0.5
-> 0.3
-> 请求 0.7，因 delta=0.4 被拒绝
-> 0.5
-> 0.7
-> 0.5
-> controlled unhealthy mock，自动 SAFE_BASELINE
-> 0.5
-> rollback baseline
```

每个合法档位发送 20 个真实、非流式请求，保持不少于 10 秒。总共完成 180 个请求，实验墙钟 221.330 秒。首个 baseline 请求包含模型首次业务 warmup，最大延迟 36.710 秒；后续各阶段 P50 约 1.017–1.031 秒，最大值 1.189 秒。

## 4. 分流结果

测试请求使用相同 `max_tokens=32`，因此本轮 request ratio 与 token ratio 相同。每条 `p_to_d_route` 同时保存选择前后的 expected remaining tokens 和 active requests；各档完成时两路均排空为 0。

| 阶段 | target rho_A | Decode A/B 请求 | actual request ratio | actual token ratio |
|---|---:|---:|---:|---:|
| rho_0.5 | 0.5 | 10 / 10 | 0.500 | 0.500 |
| rho_0.3 | 0.3 | 6 / 14 | 0.300 | 0.300 |
| rho_0.5 bridge | 0.5 | 10 / 10 | 0.500 | 0.500 |
| rho_0.7 | 0.7 | 14 / 6 | 0.700 | 0.700 |
| rho_0.5 final | 0.5 | 10 / 10 | 0.500 | 0.500 |
| rho_0.5 before rollback | 0.5 | 10 / 10 | 0.500 | 0.500 |

这证明控制量改变的是实际路由而非仅 API 状态。生产中的 token ratio 不应期待严格等于 request ratio，因为不同请求的实际输出长度不同；二者都必须保留。

## 5. Apply 与实际生效延迟

六个唯一合法 COMMIT 的 requested-to-applied 延迟为 1.052–1.149 ms。COMMIT 后首个携带相同 command ID 的 `p_to_d_route` 出现在 195.419–202.737 ms 后。所有首个路由事件均晚于 `applied_wall_ns`，并明确记录 `effective_from=next_decode_route`。

已有请求不迁移：质量检查按 `(trace_id, attempt_id)` 聚合，180/180 attempt 都只关联一个 Decode，跨 Decode ownership 数为 0。

## 6. 输出一致性

固定配置：

```text
temperature=0
seed=1024
max_tokens=32
```

全部 baseline、controlled、SAFE_BASELINE 和 rollback 请求只得到一个输出 SHA256：

```text
5d37861c325e18d766eaee48a018a9eb6a220a017682293e5bc84d496a8080b2
```

## 7. 遥测质量

| 指标 | 结果 |
|---|---:|
| JSONL events | 1476 |
| events enqueued / written | 1476 / 1476 |
| queue/full drop | 0 |
| writer-failed drop | 0 |
| serialization/build/write/flush errors | 0 / 0 / 0 / 0 |
| queue high watermark | 3 |
| JSONL 损坏行 | 0 |
| event_seq gap / duplicate | 0 / 0 |
| emit p50 / p95 / p99 | 21.881 / 26.021 / 27.021 us |

幂等 COMMIT 重放会返回并遥测同一个原始 applied acknowledgment，因此 raw 中每个受测命令有两条内容相同的 `actuator_applied` acknowledgment；二者拥有相同 command ID、generation、applied time 和 effective value，状态没有再次应用。后续构造 `u_k` 必须按 `control_command_id` 去重；本轮验证得到 6 个唯一 applied command。

## 8. 稳定性

- Pod restart：0
- Engine/API server restart：0
- 模型 reload：0
- 新 graph recapture：0
- Mooncake reinitialization：0
- OOM：0
- engine death：0
- Mooncake fatal：0
- 最终状态：`BASELINE`，两路 Decode healthy，active requests=0

冷启动期间的 Triton compile warning 含有 `Original traceback` 字样，但日志级别为 WARNING；fatal-only 扫描为空，且之后 Decode B 正常监听 13702。这不是 smoke 期间的运行错误。

## 9. 证据

- `results/control-runtime-smoke/runtime-smoke-artifacts/control-v1-runtime-smoke/reports/control_smoke.json`
- `results/control-runtime-smoke/runtime-smoke-artifacts/control-v1-runtime-smoke/raw/proxy/*.jsonl`
- `results/control-runtime-smoke/control_quality.json`
- `results/control-runtime-smoke/metadata-post/runtime-identities.txt`
- `results/control-runtime-smoke/d2-config-log-evidence.txt`
- `results/control-runtime-smoke/engine-lifecycle-counts.txt`
- `results/control-runtime-smoke/runtime-fatal-scan.txt`

