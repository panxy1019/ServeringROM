# ServingROM ROM Dataset v1 采集运行手册

## 固定基线

- Dataset：`servingrom-qwen36-1p2d-d2-rom-v1`
- 推理配置：`qwen36-1p2d-d2-full-decode-only-async-v1`
- 镜像：`snapshot-v7`
- 快照周期：`200 ms`
- 状态、扰动、输出维度：`1804 / 31 / 19`

采集期间模型 Pod 常驻。每个正式 run 仍拥有独立的 `RUN_ID`、writer、事件序号、raw 目录、派生表、质量报告和 seal。run 切换采用 drain、writer deactivate ACK、writer activate ACK 三步协议，不重启 vLLM 或重载模型。

## 启动

在 server-00 执行：

```bash
cd /home/admin/testpanxy/servingrom_rom_v1
sudo -i
nohup python3 scripts/run_rom_data_collection.py \
  > /home/admin/servingrom-results/rom-v1-controller.log 2>&1 &
```

控制器自动完成容量标定、72 个核心 run、12 个瞬态 run、逐 run seal、数据集合并、质量审计和生产 D2 恢复。

## 查看进度

```bash
cd /home/admin/testpanxy/servingrom_rom_v1
./scripts/show_rom_collection_progress.sh | python3 -m json.tool
```

只查看控制器日志，不启动高频子进程：

```bash
tail -n 120 /home/admin/servingrom-results/rom-v1-controller.log
```

查看 warm Pod 的 writer 切换状态：

```bash
POD=$(kubectl -n infra-learning get pod \
  -l app=ray-vllm-pd-servingrom-qwen36-27b \
  -o jsonpath='{.items[0].metadata.name}')
kubectl -n infra-learning exec "$POD" -- \
  python3 /opt/qwen36-pd/servingrom_run_control.py status
```

## 中断与恢复

控制器只从 `PENDING` run 继续，已经 `SEALED` 的目录不会覆盖。失败尝试保留原始证据，非系统性失败使用新 `RUN_ID` 最多重试一次；schema、守恒、事件完整性、OOM、Mooncake fatal 或 engine death 会 fail-closed 停止。

```bash
python3 scripts/run_rom_data_collection.py
```

意外终止控制器时，warm Pod 默认保留，避免再次加载模型。确认无需继续后，可显式恢复生产：

```bash
kubectl -n infra-learning scale deploy/ray-vllm-pd-servingrom-qwen36-27b --replicas=0
kubectl -n infra-learning scale deploy/ray-vllm-pd-decode-ab-qwen36-27b --replicas=1
kubectl -n infra-learning rollout status deploy/ray-vllm-pd-decode-ab-qwen36-27b --timeout=90m
```

禁止使用高频 `ps` 监控。进度只读取单个 JSON 文件，设备指标由 Pod 内已有 200 ms 采集器旁路记录。
