# ServingROM 遥测开销报告

## 测试口径

- 基线：1P2D D2，Decode 使用 `FULL_DECODE_ONLY`，开启 async scheduling。
- 负载：8 个请求，闭环并发 8，输入 128 words，输出 32 tokens，seed `20260805`。
- OFF/ON 各 3 轮；比较三轮中位数。
- ON 期间只增加旁路内存队列和异步 JSONL writer，不改变准入、路由或模型参数。

## 结果

| 指标 | Telemetry OFF | Telemetry ON | 变化 |
|---|---:|---:|---:|
| requests/s | 1.2347 | 1.2452 | +0.86% |
| TTFT P95 | 5533.16 ms | 5474.49 ms | -1.06% |
| E2E P95 | 6381.82 ms | 6321.00 ms | -0.95% |
| Proxy CPU 平均核数 | 0.0157 | 0.0202 | +28.53% |

固定请求的 8 个输出 SHA256 在 OFF 和 ON 间逐项一致。Proxy writer 共写入 1425 个事件，
`events_written == events_enqueued`，queue high watermark 为 8，dropped、serialization error 和
write error 均为 0；emit P50/P95/P99 分别约为 19.9/25.8/35.8 微秒。

## 结论

端到端延迟和吞吐未出现可测退化，均在 5% 验收范围内。Proxy CPU 相对增幅较大，但绝对值只增加
约 0.0045 个 CPU core，属于低基数效应。当前开销结论仅适用于本次 C8 小负载，不替代后续容量扫描。

