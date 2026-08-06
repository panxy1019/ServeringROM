# Snapshot Builder 设计

Builder 的输入是已经校验并派生的 Proxy lifecycle、engine request、scheduler iteration/membership、token emission、Mooncake transfer 及 device metric Parquet。它不回写 raw JSONL，也不会调用 Proxy、vLLM、Mooncake 或 NPU。

运行命令：

```bash
PYTHONPATH=. python scripts/build_full_order_snapshots.py results/<experiment>/<run>
PYTHONPATH=. python scripts/validate_full_order_snapshots.py results/<experiment>/<run>
PYTHONPATH=. python scripts/seal_servingrom_run.py results/<experiment>/<run>
```

Seal 是 fail-closed：派生文件、数组长度、窗口连续性、无效窗口和 writer 平衡任一失败都会写 `metadata/run_status.json` 为 `INVALID`。成功时才生成最终 SHA256 manifest，状态为 `SEALED`。

### 已知能力边界

CPU 运行队列、内核 block device counter、精确 KV free-block、per-layer attention 统计和网络 RDMA 硬件计数目前没有无扰 hook，因此分别为 `null` 或不出现；不会将它们伪造成零。设备采样的覆盖由质量表显式记录。
