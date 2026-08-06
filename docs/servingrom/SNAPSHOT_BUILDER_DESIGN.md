# Full-order Snapshot Builder 设计

## 1. 数据路径

```text
raw JSONL
  -> Proxy lifecycle / Engine / Scheduler / Mooncake 标准 Parquet
  -> request_id + trace_id + attempt_id 精确关联
  -> 固定墙钟边界上的请求状态回放
  -> x_k / d_k / y_k / x_{k+1}
  -> 守恒验证
  -> SHA256 seal
```

Builder 不读取 prompt 或生成正文，不调用模型，不修改 Proxy、Scheduler、Mooncake 和 raw 文件。endpoint 只用于将已记录的路由转换成 `decode-0/1`，关联键始终是 request/attempt ID。

## 2. 为什么不能按 JSONL 行分桶

`scheduler_iteration` 是动作，`device_metric` 是采样，二者都不是库存。某个 200 ms 内没有 scheduler event 可能表示引擎空闲，不能被解释为引擎离线。库存必须从 arrival、submit、first schedule、KV rank complete、first Decode schedule 和 terminal 边界持续回放。

因此覆盖判定分为两层：

1. run 级：Proxy、Prefill、Decode A/B、Mooncake、device writer checkpoint 均存在且平衡；
2. window 级：设备采样与窗口中心距离不超过一个周期，活动请求所需的状态边界完整。

空闲窗口是有效的零工作窗口。旧版“每窗口必须同时看到三个 scheduler component”的判定会把正常 idle 错判为缺失，已经移除。

## 3. 时间和重启

窗口由首次 request arrival 向下对齐 200 ms，至最后 terminal 向上对齐 200 ms。组件 duration 不跨进程相减。每个 JSONL 文件独占 process instance，event sequence 从 1 严格连续；重复、缺口、损坏行、writer drop 或 process checkpoint 缺失都会 fail-closed。

## 4. 守恒

逐窗口检查：

```text
active[k+1] = active[k] + accepted_arrivals[k]
              - completed[k] - cancelled[k] - errors[k]
```

拒绝不进入 active。每个边界的 11 个互斥 active 阶段之和必须等于 active inventory。`request_state_inventory.parquet` 还逐 request 检查状态只能沿有向状态机前进。

Scheduler membership token/count 与 iteration 总量在上游 validator 对账；Decode emitted token 与 Proxy output token 对账；Prefill 使用 runtime probe 证明的精确等式；Mooncake 检查 enqueue/start/ready 顺序、TP rank 完整性、实际 bytes 和 Proxy route 一致性。

## 5. 使用

```bash
./scripts/run_snapshot_phase_a.sh results/<experiment_id>/<run_id>
python scripts/inspect_snapshot.py results/<experiment_id>/<run_id> --window 0
```

只有 `validate_full_order_snapshots.py` 0 违规后，`seal_servingrom_run.py` 才写入：

```json
{"status":"SEALED","eligible_for_training":true}
```

否则写 `INVALID` 并保留 reasons。Seal 生成全 run SHA256 manifest，不改变原始事件。
