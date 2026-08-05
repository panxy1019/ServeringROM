# Decode 图执行与异步调度 A/B

该目录冻结生产 1P2D 的 Prefill、Mooncake、Proxy、模型和路由逻辑，仅改变 Decode 启动参数。

| 模式 | Decode 图模式 | Async scheduling |
|---|---|---|
| D0 | 默认 `FULL_AND_PIECEWISE` | 显式关闭 |
| D1 | `FULL_DECODE_ONLY` | 显式关闭 |
| D2 | `FULL_DECODE_ONLY` | 显式开启 |

生产事实基线的命令未显式传入 async 参数，但当前 Ascend 平台会自动启用 async。实验必须显式关闭 D0/D1，才能把 D0→D1 解释为图模式单变量、D1→D2 解释为调度单变量。

实验 Deployment 为 `ray-vllm-pd-decode-ab-qwen36-27b`，默认 `replicas=0`。由于实验和生产使用同一组物理设备 10..15，执行器会先保存生产快照并缩容生产 Deployment，逐组测试完成后恢复生产 Deployment。

采集器不执行 `ps`。进程 CPU/RSS 仅通过 `/var/run/qwen36-pd/*.pid` 和 `/proc/<pid>/task/<pid>/children` 读取。

```bash
cd /home/admin/testpanxy/infralearning/qwen36_pd_1p2d
./decode_graph_ab/scripts/run_decode_ab_suite.sh
```
