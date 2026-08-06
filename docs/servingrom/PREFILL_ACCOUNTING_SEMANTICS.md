# Prefill Token Accounting 精确语义

## 1. 运行时事实

v7 在真实 Prefill scheduler `_update_after_schedule()` 旁路记录：

```text
computed_tokens_before
scheduled_tokens
computed_tokens_after
```

Engine terminal 记录 `final_computed_tokens` 和 `handoff_token_count`。23 个成功 PD attempt 的每轮关系全部满足：

```text
computed_after = computed_before + scheduled_tokens
```

每请求满足：

```text
final_computed_tokens - initial_computed_tokens
  = sum(prefill scheduled_tokens)

proxy_prompt_tokens
  = final_computed_tokens + handoff_token_count
```

本轮所有请求的 `handoff_token_count=1`，因此观察到的 `scheduled=input-1` 不是容差，而是由真实 handoff 字段证明的 PD 边界语义。

## 2. 曾出现的假失败

Prefill 和 Decode 故意复用同一物理 `request_id`。首版 validator 只按 request ID 汇总 probe，把 Decode 每轮增加的 generated token 误作 Prefill computed token，产生 23 个统一失败。修正后的守恒键为：

```text
(component, request_id)
```

修正只发生在离线重建层，没有改变 hook、Scheduler 或执行顺序。

## 3. Fail-closed 规则

以下任意一项缺失都不能 seal：

- 至少一个 Prefill iteration probe；
- 任一 iteration 加法不成立；
- terminal probe 缺失；
- final 与最后 observed-after 不同；
- request 级 scheduled 求和不成立；
- prompt、final 与 handoff 等式不成立。

`connector_computed_tokens` 和 `connector_external_tokens` 在当前路径无法无扰取得时保持 null；精确等式已由 scheduler 与 terminal 字段闭环，不填伪造值。
