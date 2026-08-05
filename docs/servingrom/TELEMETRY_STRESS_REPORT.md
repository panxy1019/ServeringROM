# ServingROM Telemetry Step 1 压力测试报告

执行时间：2026-08-05 UTC

## 测试配置

```text
producer threads=4
queue_capacity=131072
batch_size=2048
flush_interval_ms=100
max_file_bytes=256 MiB
payload=3 个整数/短字符串字段 + 完整统一事件头
```

producer 只在队列达到 60% 容量时在 `emit()` 外部短暂让出 CPU。`emit()` 本身始终使用 `put_nowait()`，延迟统计不包含外部 pacing。

## 最终结果

| 指标 | 100,000 events | 1,000,000 events |
|---|---:|---:|
| emit latency P50 | 6.894 us | 6.856 us |
| emit latency P95 | 12.417 us | 11.484 us |
| emit latency P99 | 20.925 us | 21.384 us |
| emit latency max | 56.503 ms | 89.324 ms |
| emit wall time | 1.592 s | 21.991 s |
| end-to-end wall time | 2.205 s | 23.446 s |
| producer events/s | 62,796 | 45,472 |
| writer events/s | 45,344 | 42,651 |
| writer throughput | 27.215 MiB/s | 25.720 MiB/s |
| maximum queue depth | 42,942 | 79,418 |
| JSONL files | 1 | 3 |
| output size | 60.019 MiB | 603.048 MiB |
| events enqueued | 100,000 | 1,000,000 |
| events written | 100,000 | 1,000,000 |
| queue drops | 0 | 0 |
| writer-failure drops | 0 | 0 |
| serialization errors | 0 | 0 |
| write/flush errors | 0 | 0 |

两档均满足：

```text
events_written == events_enqueued == requested_events
dropped_events == 0
```

完整流式检查结果：

```text
100,000：1 file，100,000 valid lines，event_seq 连续
1,000,000：3 files，1,000,000 valid lines，event_seq 连续
```

原始结果目录（受 `.gitignore` 保护，不提交仓库）：

```text
results/telemetry-step1/stress-100000-final-20260805T052500Z/
results/telemetry-step1/stress-1000000-final-20260805T052500Z/
```

第一份目录中的 JSONL 是本阶段生成的可检查示例；仓库中另保存了两行精简 schema 示例 `docs/servingrom/examples/sample_events.jsonl`。

## 压测发现并修复的问题

第一轮百万事件检查发现：多 producer 线程可能在线程 A 取得序号后切换到线程 B，导致 B 先入队，文件出现相邻序号交换。序号没有重复，但原始文件顺序不再严格递增。

修复方式是使用一个短临界区包住：

```text
双时钟采样 → event_seq 分配 → 事件封装 → put_nowait
```

同一把锁也与 `close()` 协调，保证 close 的入队目标快照不会漏掉正在提交的事件。修复后重新执行 10 万和 100 万完整压力测试，两档原始文件均严格连续。

检查工具同时从 O(events) 内存改为 O(processes) 流式校验，百万行检查不再保存全部事件键或序号。

## NullEmitter 证明

单元测试在创建前后比较 `threading.enumerate()` 的线程 ID 集合，并检查输出目录：

```text
线程集合不变
输出目录为空
emit() 返回 False
flush()/close() 返回 True
```

`NullEmitter.__slots__=()`，没有 queue、lock、clock、sequence、writer 或 sink 成员。关闭配置经工厂判断后不会实例化 `AsyncTelemetryEmitter`。

## 已知限制

- 当前数字是合成事件库吞吐，不代表接入 D2 后的 TTFT/TPOT 开销；
- Python GIL/OS 调度会产生远高于 P99 的极少数 max latency；
- payload 只做浅拷贝，调用方在 `emit()` 返回后不得修改嵌套可变对象；
- emitter 必须在 worker fork/spawn 后创建，不能把已有 writer 线程跨 fork 继承；
- 进程硬崩溃时，尚在内存队列中的事件无法恢复；
- 没有跨进程聚合、分布式全局序号或跨节点 monotonic clock 对齐；
- 本阶段未接入任何真实业务 hook，因此没有改变 D2 输出或调度行为。
