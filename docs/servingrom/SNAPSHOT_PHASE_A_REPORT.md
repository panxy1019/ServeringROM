# ServingROM Step 6/7 Phase A 实验报告

## 1. 结论

Step 6 固定周期 Full-order Snapshot Builder 与 Step 7 fail-closed 质量门均已通过。正式 run：

```text
experiment_id: servingrom-snapshot-phase-a
run_id: snapshot-phase-a-fixed-20260806T032500Z
config_id: qwen36-1p2d-d2-full-decode-only-async-v1
结果目录: /home/admin/servingrom-results/servingrom-snapshot-phase-a/
          snapshot-phase-a-fixed-20260806T032500Z
status: SEALED
eligible_for_training: true
```

实验 Deployment 已缩容到 0；冻结生产 D2 在实验结束后恢复。没有进入 POD、DMDc、MPC、actuator 或容量扫描。

## 2. 镜像与运行事实

```text
image tag:
110.120.0.3:8889/infra/qwen36-pd-worker:
v0.22.1rc1-a3-ray248-servingrom-snapshot-v7

registry digest:
sha256:0c9a4668e0c15f862fee733ab5c5b721e8f88985dd9cfa6f33404b976b15eadb

vLLM commit:        0decac0d96c42b49572498019f0a0e3600f50398
vLLM-Ascend commit: 5f6faa0cb8830f667266f3b8121cd1383606f2a1
Pod UID: b00e9ebe-6b26-49ad-ab79-d07b6580b12c
restartCount: 0
```

拓扑保持冻结：Prefill TP2 eager，Decode A/B TP2 `FULL_DECODE_ONLY` + async scheduling，Mooncake Decode pull。实验没有更改 Proxy admission、路由或 KV 顺序。

## 3. v7 修正

### 3.1 Proxy tokenizer 计数

`apply_chat_template(tokenize=True)` 在当前 transformers 返回 BatchEncoding。旧代码对对象本身执行 `len()`，得到 key 数量 2，而不是真实 token 数。v7 显式取得 `input_ids` 并处理 batch 维，Phase A 的确定性超限请求现在由 Proxy 返回 429，且没有 Prefill submit 或 engine request。

### 3.2 Prefill accounting

23/23 成功 PD attempt 的 iteration 均满足：

```text
computed_after = computed_before + scheduled
final - initial = sum(scheduled)
prompt = final + handoff_token_count
handoff_token_count = 1
```

第一次离线验证错误地按 request ID 合并 Prefill 和 Decode probe。由于物理 ID 按设计跨两个 engine 保持一致，Decode 的 computed counter 被误当作 Prefill terminal。将守恒键改为 `(component, request_id)` 后，23 项违规归零；运行时数据和执行语义没有修改。

### 3.3 窗口覆盖

早期 Builder 要求每个 200 ms 窗口同时出现 Prefill、Decode A 和 Decode B scheduler event，只得到 2 个有效窗口。这一判断混淆了“没有工作”和“组件失联”。修正后使用 writer checkpoint、event sequence、设备采样间隔及活动请求状态边界判断存活；空闲窗口保留为合法零工作样本。

## 4. Phase A 负载

两次受控 workload 合并构成同一封存 run，覆盖短请求、长 Prefill、C8、主动取消、长 Mooncake 和真实 429。首轮在修复 tokenizer 后证明两个原长输入超过 8192-token admission budget；它们被正确拒绝。第二轮将长输入缩至预算内并保留独立超限请求。

最终原始生命周期：

| 指标 | 数值 |
|---|---:|
| trace | 27 |
| accepted PD attempt | 23 |
| complete | 21 |
| cancel | 2 |
| Proxy 429 reject | 4 |
| Decode A / B | 12 / 11 |
| prompt token | 103,430 |
| completed output token | 608 |
| request-level KV transfer | 23 |
| KV actual bytes | 6,258,622,464 |

第二轮 C8：8/8 成功，墙钟 1.476 s，5.420 requests/s，TTFT P50/P95 为 0.602/0.620 s，E2E P50/P95 为 1.455/1.471 s。该值用于本轮采集链 smoke，不是正式容量结果。

固定 `temperature=0, seed=1024, output=64`：

```text
telemetry OFF SHA256:
e28e1f6446e8d1102173c7e19c214ed0e65aa46141f26d7baf0b19ec7578bb3c

v7 telemetry ON SHA256:
e28e1f6446e8d1102173c7e19c214ed0e65aa46141f26d7baf0b19ec7578bb3c
```

两侧 HTTP 200、output token 64，输出一致。

## 5. Snapshot 结果

```text
snapshot period:        200 ms
window semantics:       [t_k,t_{k+1})
window count:           942
valid window:           942/942 (100%)
full_state shape:       [942, 1804]
disturbance shape:      [942, 31]
output shape:           [942, 19]
next_state shape:       [942, 1804]
```

942 个窗口覆盖首次 arrival 到最后 terminal。状态向量保留 Prefill length/age/progress、TTFT slack、Mooncake D1/D2 age/bytes、Decode context/progress/TPOT slack 等分布，而不是只保留少量均值。

## 6. 自动质量门

| 检查 | 结果 |
|---|---:|
| snapshot window coverage | 100% |
| valid snapshot ratio | 100% |
| request inventory conservation | 100% |
| stage inventory conservation | 100% |
| KV lifecycle violation | 0 |
| Proxy route/Decode mismatch | 0 |
| scheduler membership violation | 0 |
| Decode token accounting violation | 0 |
| Prefill exact accounting violation | 0 |
| NaN / Inf / negative | 0 / 0 / 0 |
| writer checkpoints | 17 |
| writer mismatch/drop/error | 0 |
| JSONL damaged line | 0 |
| event_seq gap process | 0 |
| snapshot manifest mismatch | 0 |
| Pod restart | 0 |

Seal 生成 76 个文件的全 run SHA256 manifest。`metadata/run_status.json` 明确记录 `SEALED` 和 `eligible_for_training=true`。

## 7. 产物入口

```bash
./scripts/run_snapshot_phase_a.sh \
  /home/admin/servingrom-results/servingrom-snapshot-phase-a/\
snapshot-phase-a-fixed-20260806T032500Z

python scripts/inspect_snapshot.py \
  /home/admin/servingrom-results/servingrom-snapshot-phase-a/\
snapshot-phase-a-fixed-20260806T032500Z --window 0
```

核心报告：

```text
reports/proxy_lifecycle_quality.json
reports/internal_data_quality.json
reports/snapshot_data_quality.json
metadata/run_status.json
derived/snapshots/snapshot_manifest.json
```

## 8. 已知边界

- AICore、物理 HBM、功耗和 DMA 硬件时间仍为 null，capability 文件已声明；不作为本阶段阻断项。
- KV inflight 只有 start/complete，没有 descriptor 进度，因此 bytes 在完成前按实际总 bytes 保持，不做线性插值。
- 当前没有运行时 actuator，所有 engine 参数都是 `mu`，不能用于识别时变控制响应。
- 本 run 包含功能 smoke 与正式 Phase A 场景，中间空闲窗口被保留。Step 8A 应使用明确 warmup/measurement/drain marker 缩短非测量区间。

## 9. Step 8A 建议（未执行）

仅采 Balanced workload 的约 30%、60%、85% capacity 三档，每档固定 config、随机种子和至少三次重复。先检查 200 ms 下的状态稀疏率、POD 奇异值衰减、跨重复稳定性和高负载窗口覆盖；若 30% 档过度稀疏，可只在下一轮单变量比较 100/200/500 ms。通过后再设计完整容量扫描，本轮未启动这些实验。
