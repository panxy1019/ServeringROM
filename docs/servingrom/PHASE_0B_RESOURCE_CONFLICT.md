# Phase 0B D2 恢复资源冲突报告

检查时间：2026-08-05 04:32 UTC

## 结论

初次检查结论为 **BLOCKED**：不允许在生产 Pod 运行时启动占用同一物理卡的 D2 Pod。2026-08-05 用户随后明确授权生产缩容并要求启动既有冻结 D2，该资源冲突已通过受控切换解除，运行结果见 `D2_DEPLOYMENT_SMOKE.md`。

生产和冻结 D2 都固定绑定物理 NPU `Ascend910-10` 至 `Ascend910-15`。节点虽然报告 16 张 NPU capacity/allocatable，但相同设备不能同时分配给两个 Pod；改用其他物理卡会改变冻结基线，不应沿用 D2 config ID。

## 只读证据

| 对象 | replicas/状态 | NPU request | 固定设备 |
|---|---|---:|---|
| `ray-vllm-pd-worker-qwen36-27b` | 1/Ready，restart=0 | 6 | 10,11,12,13,14,15 |
| `ray-vllm-pd-decode-ab-qwen36-27b` | 0 | 6 | 10,11,12,13,14,15 |
| `a3-server-00` | capacity=16，allocatable=16 | - | - |

当前 A3 上唯一申请 NPU 的 Pod 是生产 Pod，申请 6 张并固定到 10–15。ServingROM 专用 Deployment 已定义为 `replicas=0`，不会占用设备。

已安全创建的零副本资源：

| 资源 | UID/哈希 |
|---|---|
| Deployment `ray-vllm-pd-servingrom-qwen36-27b` | `1c48b244-65f9-401b-98fd-ef6545e209d8` |
| ConfigMap `qwen36-pd-servingrom-d2-scripts` | UID `e038d62b-1405-46b6-a18b-f7332e79d9e1` |
| ConfigMap data SHA256 | `8d062fa5a400946bbab94aa51673ebaca30792e10bbfc5e4d22d0f641bcb7cdd` |

创建前后生产 Deployment UID 均为 `b13fb087-caff-454a-916a-3220526221dc`，generation 均为 19，副本和 Ready 均为 1；生产 Pod restart 仍为 0。生产 Pod template 没有被修改。生产 config ID 只在仓库基线注册表中建立映射，没有为增加标签而触发生产 rollout。

冻结 D2 和 ServingROM 专用 ConfigMap 的数据 SHA256 均为：

```text
8d062fa5a400946bbab94aa51673ebaca30792e10bbfc5e4d22d0f641bcb7cdd
```

## 受控恢复顺序（本轮未执行）

```bash
./scripts/servingrom/prepare_d2_experiment.sh

CONFIRM_PRODUCTION_INTERRUPTION=scale-down-ray-vllm-pd-worker-qwen36-27b \
  ./scripts/servingrom/scale_down_production_for_d2.sh

./scripts/servingrom/start_d2_experiment.sh
```

实验完成或任意失败后：

```bash
./scripts/servingrom/rollback_to_production.sh
```

生产缩容需要明确维护窗口批准。本阶段没有执行上述命令，没有修改或重启生产 Deployment。

## 未产生的运行证据

因为实验 Pod 未启动，以下值当前不存在，不能伪造或沿用旧 run：

- ServingROM Pod UID；
- 本次启动日志和 effective engine config；
- 本次 logical NPU mapping；
- D2 回归结果及 output SHA256。

这些字段只适用于未启动的 ServingROM 专用零副本 Deployment。最终采用的是既有冻结 Deployment `ray-vllm-pd-decode-ab-qwen36-27b`；其恢复证据单独记录，避免把两个 Deployment 的 UID 和 metadata 混合。
