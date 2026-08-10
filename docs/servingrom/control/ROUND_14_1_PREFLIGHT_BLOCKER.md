# ServingROM Round 14.1 Preflight 阻断报告

## 1. 结论

Round 14.1 尚未启动任何 workload 或 actuator excitation。当前常驻 Control-v1 Pod 无法在不重启的约束下生成合格的 Pilot 数据，继续执行会把 12 个 run 混入 Step 13 的 Proxy 文件，并缺失原 Full-order X 所需的 Engine/Mooncake/device 遥测。因此本轮在 preflight 阶段 fail-closed。

```text
pilot_collection_started=false
existing_dataset_modified=false
control_pilot_dataset_created=false
persistent_excitation_pass=not_evaluated
control_authority_pass=not_evaluated
```

## 2. 当前 Pod 的事实状态

- Pod：`ray-vllm-pd-control-v1-qwen36-27b-5c58f55476-tptn8`
- Pod UID：`0649a4ff-c87a-4a6c-ba07-9ea42c4d53de`
- Ready：true
- restartCount：0
- 当前镜像：`qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730`
- 当前 digest：`sha256:15c3a3db3772807cc09d9ad37756cd973e3b078956b6a239375ec4ae23317133`
- 历史 Snapshot v7 镜像：`qwen36-pd-worker:v0.22.1rc1-a3-ray248-servingrom-snapshot-v7`
- Snapshot v7 digest：`sha256:0c9a4668e0c15f862fee733ab5c5b721e8f88985dd9cfa6f33404b976b15eadb`

当前 Pod 保持 D2、`FULL_DECODE_ONLY`、async scheduling、Prefill TP2 和两路 Decode TP2，Control-v1 A1 正常。这些事实没有问题；阻断只在遥测能力与 run 隔离。

## 3. 阻断证据

容器内不存在热轮转控制设施：

```text
/opt/qwen36-pd/servingrom_run_control.py                 missing
/var/run/qwen36-pd/servingrom-run-control.json          missing
/var/run/qwen36-pd/servingrom-run-control-acks/         missing
```

当前 run 的 raw 目录只有：

```text
raw/proxy/5818b0cee50a028553a78fb54ab3d1da.00000.jsonl
```

缺失：

```text
raw/prefill
raw/decode-0
raw/decode-1
raw/mooncake
raw/device
```

Prefill、Decode A、Decode B 和 Proxy 虽继承相同的 `SERVINGROM_*` 环境变量，但只有 Proxy 代码实际创建 emitter。三个 vLLM Engine 没有 Snapshot v7 runtime hook，也没有每个进程的 run-control ACK。

## 4. 为什么不能降级执行

只靠 Proxy 可以观测：

- `actuator_applied` 与 `p_to_d_route`；
- 新请求的目标/实际 request ratio；
- expected remaining tokens 与 active request 的 Proxy 账本；
- 请求 arrival、completion、TTFT 和输出数量。

但无法无扰恢复原 1804 维 X 中的关键内容：

- Prefill/Decode scheduler iteration 与 membership；
- running/waiting、scheduled token 和 scheduler workload；
- Engine token emission；
- KV block 使用、KV handoff 和 Mooncake rank/request transfer；
- 100 ms device/HBM/AICore 采样；
- writer component completeness 与原 Snapshot v2 守恒门。

将这些字段填 0、从 actual ratio 反推 U，或用 Proxy expected tokens 冒充 Engine scheduler 状态都会改变数据语义。Snapshot v2 的 `_writer_health()` 也会明确报 `writer_checkpoint_missing`，run 无法 fail-closed seal。

此外，当前 Proxy writer 的 `experiment_id/run_id/output_dir` 在进程启动时固定为 Step 13 smoke。没有热轮转意味着 12 个 Pilot run 会写进同一 JSONL，无法满足 run-level 隔离，也会污染已完成的 Step 13 原始证据。

## 5. 已确定的 Pilot 工作点

复用封存 Dataset v1 的 capacity calibration，只读得到：

| workload | lambda_stable | 55% rate | 85% rate |
|---|---:|---:|---:|
| balanced | 0.500 req/s | 0.275 req/s | 0.425 req/s |
| mixed-bimodal | 0.650 req/s | 0.3575 req/s | 0.5525 req/s |

计划规模仍为 `2 workloads x 2 levels x 3 excitation types = 12 runs`，每次 120 s warmup、600 s excitation、drain。未发送任何 Pilot 请求。

## 6. 最小正确修复

需要构建并启动一次“Snapshot v7 + Control-v1”的合并 Worker：

1. 基础运行时使用已通过 Step 6/7 的 Snapshot v7 hooks；
2. Proxy 使用已通过 Step 13 的 Control-v1；
3. entrypoint 启用 `servingrom_run_control.py`、control file 和各 writer ACK；
4. 保持完全相同的 D2、模型、Mooncake、NPU 映射和 Engine 参数；
5. 新 Pod Ready 后先做一次 capability smoke，确认六类 writer 都能 activate/deactivate；
6. 此后冻结该合并 Pod，在同一 Pod 内热轮转完成 12 runs，不再重启。

这需要一次 Pod/Engine/model 冷启动，与本轮“保持当前同一个 Pod且不得重载”的约束冲突。不能通过 ConfigMap 热更新已运行的 vLLM Engine hooks；强行向进程注入代码会破坏旁路、安全和可审计性。

## 7. 需要解除的唯一约束

允许一次受控切换：从当前 Proxy-only Control-v1 Pod 切换到合并的 Snapshot-v7+Control-v1 Pod。完成这一次启动后，12-run Pilot 全程保持该 Pod 常驻。若不允许这次切换，Round 14.1 无法在既定数据合同下正确执行。

