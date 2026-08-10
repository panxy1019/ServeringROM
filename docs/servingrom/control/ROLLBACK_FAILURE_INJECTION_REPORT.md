# ServingROM Control-v1 回滚与故障注入报告

## 1. 负向命令验证

| 场景 | 输入 | 结果 |
|---|---|---|
| duplicate prepare | 已使用的 command ID | `duplicate_command_id`，HTTP 409 |
| idempotent commit replay | 已完成的相同 COMMIT | 返回原 applied 结果，不增加 generation，不再次改变状态 |
| stale generation | generation 未严格递增 | `stale_or_nonmonotonic_generation`，HTTP 409 |
| out of range | `rho_A=0.9` | `requested_value_out_of_range`，HTTP 409 |
| dwell violation | COMMIT 后不足 5 秒申请下一值 | `minimum_dwell_time_not_met`，HTTP 409 |
| rate limit | `0.3 -> 0.7` | `maximum_step_exceeded`，HTTP 409 |
| compare-and-swap | manager 单元测试的 expected value 不匹配 | `compare_and_swap_mismatch`，状态不变 |

所有 PREPARE 拒绝均未改变 generation、mode 或 effective value。

## 2. Safety fallback

在 `rho_A=0.5` controlled 状态下，通过仅在实验 Deployment 开启的受控测试端点模拟 Decode unhealthy。结果：

```text
old_value=0.5
effective_value=baseline
control_status=SAFE_BASELINE
control_generation=6
effective_from=next_decode_route
reason=controlled_mock:decode_unhealthy_smoke
```

随后真实请求成功，输出 SHA256 与 baseline 相同。两路 Decode 服务本身没有被停止，因此该测试验证的是控制状态机的 fail-closed 分支，不制造真实 Engine 故障。

## 3. 显式 rollback

Safety 验证后重新进入 `rho_A=0.5`，再提交 generation=8 的 rollback：

```text
old_value=0.5
requested_value=baseline
effective_value=baseline
reason=rollback_applied
effective_from=next_decode_route
```

最终控制状态为 `BASELINE`，恢复冻结的最小 expected remaining tokens + fair tie policy。rollback 后 20 个真实请求成功，固定输出哈希不变。

## 4. KV 与已有请求

Routing actuator 只在 Prefill 完成后为新请求执行一次。已生成的 `InstanceInfo.decoder_key` 不读取后续 control state；finish/release 仍使用原 key。raw telemetry 中不存在同一 attempt 出现在两个 Decode 的情况，因此没有 KV ownership 迁移。

## 5. 限制

本轮 unhealthy 是受控 mock，不是通过杀死 Decode 进程制造故障；这样可以验证 fail-closed 分支而不违反“不重启 Engine”的冻结约束。真实 NodeListener taint 路径与该分支调用同一个 `force_safety_fallback()`。

自主 safety fallback 没有外部 PREPARE 请求，其 `requested_wall_ns` 是内部事件构造时间；状态实际生效和后续数据对齐应以 `applied_wall_ns` 与首个后续 `p_to_d_route` 为准。

