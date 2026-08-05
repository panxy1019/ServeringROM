# Qwen3.6-27B W8A8 单节点 1P2D 实验

本目录记录 A3 节点上的 Prefill/Decode 分离实验。模型使用
`/home/admin/models/Qwen3.6-27B-w8a8`，Worker 固定申请物理 `Phy-ID 2-7`，
即三块 910C 板卡。

## 深度技术文档

- [PD 分离与 Mooncake 系统架构](./PD_DISAGGREGATION_AND_MOONCAKE_ARCHITECTURE_CN.md)
- [Mooncake Transfer Engine、vLLM 与 Ascend 实现深析](./MOONCAKE_VLLM_ASCEND_IMPLEMENTATION_CN.md)
- [任务二：1P2D 吞吐、时延与 Goodput 实验手册](./TASK2_PD_THROUGHPUT_EXPERIMENT_CN.md)
- [任务三：请求路由、KVCache 与物理 NPU 的深度观测手册](./TASK3_PD_REQUEST_KV_OBSERVABILITY_CN.md)
- [任务二与任务三实测报告（2026-08-03）](./PD_TASK2_TASK3_EXPERIMENT_REPORT_20260803_CN.md)
- [Proxy 调度优化与容量评审附录（2026-08-04）](./PD_PROXY_SCHEDULING_AND_CAPACITY_APPENDIX_CN.md)

任务二和任务三以当前已经跑通的 `decode.dp_size=1`、两个独立 Decode replica
和 1P2D 实测状态为准。较早文档中的未完成状态与 `decode.dp_size=2` 讨论属于
历史诊断记录。

## 资源

```text
Kubernetes: cpu=64, memory=256Gi, huawei.com/Ascend910=6
Prefill:  Phy-ID 2,3, TP=2
Decode A: Phy-ID 4,5, TP=2
Decode B: Phy-ID 6,7, TP=2
```

## 构建和推送

在 A3 节点执行：

```bash
cd /home/admin/qwen36_pd_1p2d
chmod +x build-and-push.sh scripts/*.sh
./build-and-push.sh
```

目标镜像：

```text
110.120.0.3:8889/infra/qwen36-pd-worker:v0.22.1rc1-a3-ray248-20260730
```

## 部署

在 server-00 执行：

```bash
cd /home/admin/testpanxy/infralearning/qwen36_pd_1p2d
chmod +x deploy.sh start.sh rollback.sh
./deploy.sh
```

`deploy.sh` 只以 `replicas=0` 注册资源。启动前先在 A3 执行：

```bash
cd /home/admin/qwen36_pd_1p2d
sudo python3 scripts/check_target_npus_idle.py
```

检查通过后在 server-00 启动：

```bash
cd /home/admin/testpanxy/infralearning/qwen36_pd_1p2d
NPU_IDLE_CONFIRMED=YES ./start.sh
```

查看启动过程：

```bash
sudo /usr/local/bin/k3s kubectl -n infra-learning get pod -w
sudo /usr/local/bin/k3s kubectl -n infra-learning logs \
  deploy/ray-vllm-pd-worker-qwen36-27b -f
```

进入 Pod 检查：

```bash
POD=$(sudo /usr/local/bin/k3s kubectl -n infra-learning get pod \
  -l app=ray-vllm-pd-worker-qwen36-27b -o jsonpath='{.items[0].metadata.name}')
sudo /usr/local/bin/k3s kubectl -n infra-learning exec "$POD" -- \
  /opt/qwen36-pd/status.sh
sudo /usr/local/bin/k3s kubectl -n infra-learning exec "$POD" -- \
  /opt/qwen36-pd/smoke.sh
```

## 回退

```bash
cd /home/admin/testpanxy/infralearning/qwen36_pd_1p2d
./rollback.sh
```

回退只缩容本实验 Worker，不修改现有 Ray Head 和其他工作负载。

## 官方参考

- vLLM-Ascend 单节点 Mooncake PD：
  https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_single_node.html
- vLLM-Ascend 多节点/多实例 Mooncake PD：
  https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/pd_disaggregation_mooncake_multi_node.html
- 支持模型矩阵：
  https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_models.html
