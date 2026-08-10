# ServingROM Control-v1 运行时控制面设计

## 1. 目标与非目标

Control-v1 只控制新请求的 Decode A/B 目标分配比例。它不是 Kubernetes 配置控制器，不写 ConfigMap，不重启 Pod，不更新 vLLM Engine，也不迁移 KV。

```text
Client request
    -> Proxy admission
    -> Prefill + Mooncake metadata
    -> SharedProxyScheduler.pick_decoder
         baseline: frozen least-load + fair tie
         controlled: weighted deficit + load guard
    -> Decode A or B
```

## 2. 状态机

```text
BASELINE
   | PREPARE(valid CAS/range/health/dwell)
   v
PREPARED --COMMIT--> CONTROLLED
   |                    | decoder unhealthy / explicit failure injection
   | expiry             v
   +--> BASELINE     SAFE_BASELINE
                        |
CONTROLLED --ROLLBACK---+
```

`PREPARE` 只创建有 TTL 的候选状态；它不改变路由。`COMMIT` 在 scheduler manager 的串行调用边界原子替换状态。每次有效状态变化递增 generation。

## 3. API

```text
GET  /servingrom/control/state
POST /servingrom/control/prepare
POST /servingrom/control/commit
POST /servingrom/control/rollback
```

请求字段：

```json
{
  "control_command_id": "unique-id",
  "control_generation": 1,
  "actuator_name": "decode_routing_ratio",
  "requested_value": 0.5,
  "expected_current_value": "baseline",
  "requested_wall_ns": 0
}
```

返回包含 accepted/rejected、old/requested/effective value、validated/applied wall time、effective boundary、generation 和 reason。重复 `COMMIT` 返回同一个已应用结果并标记 `idempotent_replay=true`；重复 prepare 和迟到 generation 被拒绝。

## 4. 确定性 weighted fair

第 `n+1` 次 assignment 前，分别计算：

```text
deficit_A = rho_A * (n + 1) - assigned_A
deficit_B = (1-rho_A) * (n + 1) - assigned_B
```

选择 deficit 较大者。相等时固定选择 A，因此没有无 seed 随机性。累计误差被约束在有限范围，窗口变长后 request ratio 收敛到目标。Token ratio 是结果观测而非约束，因为请求输出预算并不等长。

## 5. 负载与健康 Guard

比例控制是软目标，服务安全是硬约束：

- 目标侧 expected remaining tokens 比另一侧高 2048 以上时，执行 load bypass；
- 任一 decoder unavailable/tainted 时退出 controlled，进入冻结 baseline；
- range、step、dwell、CAS 或 health 校验失败时 PREPARE 不改变状态；
- telemetry 故障被隔离，不能阻断业务路径。

## 6. 可观测生效

控制事件记录控制面的 requested/validated/applied 时间；`p_to_d_route` 同时记录命令 ID、generation、目标比例、近期 request/token 比例、两路 expected remaining tokens 和 active requests。二者关联后可得到：

```text
apply latency = applied_wall_ns - requested_wall_ns
first-effect latency = first controlled p_to_d_route.ts_wall_ns - applied_wall_ns
```

未来 `u_k` 只从 `actuator_applied` 构造，并按 `applied_wall_ns` 在 snapshot 时间轴上做前向保持。

## 7. 部署和回退

实验 Deployment `ray-vllm-pd-control-v1-qwen36-27b` 复用冻结 D2 的镜像与全部 Engine 参数，Control 代码通过独立只读 ConfigMap 挂载。`deploy_control_plane.sh` 从冻结 Deployment 克隆资源/NPU/镜像，默认创建为 0 副本；`rollback_control_plane.sh` 将实验实例缩容并恢复冻结 D2。

Control 关闭或处于 baseline 时，`SharedProxyScheduler` 执行原始函数，不调用 weighted fair actuator。故障时不需要重载模型即可回到 baseline；Deployment 级回退脚本是第二层恢复手段。

