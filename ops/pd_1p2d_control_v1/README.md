# Qwen3.6 1P2D + Mooncake + ServingROM Control-v1 启动说明

本目录是 `server-00` 上当前已验证 1P2D 实例的生命周期入口。它管理的是：

```text
namespace:  infra-learning
Deployment: ray-vllm-pd-control-pilot-qwen36-27b
Service:    qwen36-pd-control-pilot
config_id:  qwen36-1p2d-d2-full-decode-only-async-control-v1
```

脚本只管理 Kubernetes Deployment。模型、Prefill、Decode、Mooncake 和 Proxy 都在同一个 Pod 内由 entrypoint 顺序启动，不应在宿主机手工执行多个 `vllm serve`。

## 1. 冻结配置

| 项目 | 当前值 |
|---|---|
| 模型 | Qwen3.6-27B-w8a8 |
| 镜像 | `110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-servingrom-snapshot-v7` |
| 节点 | `a3-server-00` / arm64 |
| 物理 NPU | Ascend910 10--15，共 6 张 |
| CPU / 内存 | 64 CPU / 256 GiB |
| Prefill | TP2，物理 10/11，API 13700，Mooncake producer 36000 |
| Decode A | TP2，物理 12/13，API 13701，Mooncake consumer 36100 |
| Decode B | TP2，物理 14/15，API 13702，Mooncake consumer 36200 |
| Proxy | OpenAI API 8080 |
| Ray | 注册 CPU=64、NPU=6、PD_PREFILL=1、PD_DECODE=2 |

Pod 申请注解固定的是物理卡。容器启动后必须运行 `discover_npu_mapping.py`，再把物理卡映射成容器内逻辑 ID；不能假设容器逻辑 ID 仍是 10--15。

## 2. 实际启动链

```text
kubectl scale Deployment to 1
  -> Kubernetes 在 a3-server-00 创建 6-NPU Pod
  -> 挂载模型、Ascend driver、ConfigMap、/dev/shm 和结果目录
  -> 执行 pd-worker-entrypoint-instrumented.sh
     -> 初始化 run-control 与 telemetry 目录
     -> source Ascend/CANN 环境
     -> 发现物理/逻辑 NPU 映射
     -> 加入 Ray Head，注册自定义资源
     -> 启动 Prefill TP2，并等待 /health
     -> 启动 Decode A TP2，并等待 /health
     -> 启动 Decode B TP2，并等待 /health
     -> 启动 PD Proxy，并等待 /openapi.json
     -> 启动 200 ms device collector
     -> 写入 /var/run/qwen36-pd/READY
     -> Kubernetes readinessProbe 通过
```

启动采用有意的串行顺序。这样某个 Engine 加载失败时，entrypoint 会在对应日志中保留明确现场，不会出现三个模型同时加载导致难以归因的 HBM、CANN 或 shard 错误。

## 3. 三个 vLLM Engine

公共参数包括：

```text
--tensor-parallel-size 2
--quantization ascend
--max-model-len 32768
--gpu-memory-utilization 0.88
--no-enable-prefix-caching
--seed 1024
--safetensors-load-strategy eager
--kv-transfer-config MooncakeConnectorV1
```

Prefill 使用：

```text
--port 13700
--max-num-batched-tokens 8192
--max-num-seqs 16
--enforce-eager
kv_role=kv_producer
```

Decode A/B 使用：

```text
--port 13701 / 13702
--max-num-batched-tokens 4096
--max-num-seqs 64
--async-scheduling
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
kv_role=kv_consumer
```

这里不是 `tensor_parallel_size=6`。它是三个彼此独立的 TP2 Engine：一个 Prefill 完整模型副本和两个 Decode 完整模型副本。Mooncake 负责把 Prefill 产生的 KV 数据交给选中的 Decode Engine，Proxy 负责选择 Decode A/B。

## 4. Proxy 与控制面

Proxy 监听 8080，将请求先送到 13700 完成 Prefill，再根据 `rho_A` 与安全策略选择 13701 或 13702。Control-v1 只允许热更新 Decode 路由比例：

```text
rho_A range:       [0.2, 0.8]
maximum step:      0.2
minimum dwell:     5 s
load skew ceiling: 2048 expected tokens
```

Prefill token budget、Decode max-num-seqs、图模式和 TP 配置不是本版本的热控制量。修改它们需要新配置 ID 和受控重启。

## 5. 常用命令

首次使用：

```bash
cd /home/admin/testpanxy/infralearning/pd_1p2d_control_v1
sudo -i
```

启动保留在集群中的现有 Deployment：

```bash
./start.sh
```

重新 apply 本目录 YAML 后启动。该命令会先确认四个所需 ConfigMap 已存在：

```bash
./apply-and-start.sh
```

停止并释放 6 张 NPU：

```bash
./stop.sh
```

状态、日志和容器 Shell：

```bash
./status.sh
./logs.sh all 200
./logs.sh prefill 300
./logs.sh decode-a 300
./logs.sh decode-b 300
./logs.sh proxy 300
./shell.sh
```

## 6. 启动成功标准

必须同时满足：

1. Deployment `1/1` available；
2. Pod `Ready=true` 且 `restart=0`；
3. `/var/run/qwen36-pd/service-device-map.txt` 显示三组不同逻辑设备；
4. 13700、13701、13702 的 `/health` 成功；
5. 8080 的 `/openapi.json` 成功；
6. `effective-config.txt` 确认 Decode 为 `FULL_DECODE_ONLY + async scheduling`；
7. 日志没有 OOM、engine death、Mooncake fatal 或 graph recapture 异常。

## 7. 停止语义和数据保留

`stop.sh` 只把 Deployment 缩容到 0。Kubernetes 会向 entrypoint 发送终止信号，entrypoint trap 会终止 Prefill、Decode A/B、Proxy、device collector，并执行 `ray stop --force`。

以下 `emptyDir` 内容随 Pod 删除：

```text
/var/run/qwen36-pd
/var/log/qwen36-pd
/dev/shm
```

`/servingrom-results` 映射到宿主机 `/home/admin/servingrom-results`，已落盘的 raw/derived/report 不会因为缩容而删除。Deployment、Service、ConfigMap 和模型权重也不会被删除。

## 8. 配置来源

本目录的 `qwen36-control-pilot-v1.yaml` 是可重新 apply 的声明式 Deployment/Service。`reference/pd-worker-entrypoint-instrumented.sh` 是当前 Pod 内启动逻辑的只读参考副本。真正运行的 entrypoint 和 Python 控制代码来自以下 ConfigMap：

```text
servingrom-entrypoint-control-pilot-v1
servingrom-telemetry-control-pilot-v1
qwen36-pd-control-pilot-scripts
servingrom-control-pilot-v1-code
```

只执行 `start.sh` 不会重建或修改这些 ConfigMap，因此适合普通启停。代码升级应在源码仓库中完成，生成新 ConfigMap/hash/config ID 后再进行受控部署。
