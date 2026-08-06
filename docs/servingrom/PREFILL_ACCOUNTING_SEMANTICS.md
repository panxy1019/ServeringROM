# Prefill Accounting 语义

`prefill_accounting_probe` 仅用于 Phase A，目标是证明真实 Scheduler token 语义，而非根据名字猜测。每个 Prefill request 必须记录：Proxy tokenizer input token、Engine request 初始 computed token、每一 iteration 的 computed-before/scheduled/computed-after、terminal final computed token、输出前后 token、connector computed/external token、handoff token 与 finish reason。

通过条件：每 iteration `computed_after = computed_before + scheduled_tokens`；请求级 `final - initial = sum(scheduled_tokens)`；并且 `input_tokens = final_computed_tokens + handoff_token_count`。因此 `input_tokens - scheduled_tokens = 1` 不能被默认为协议事实，必须由 probe 的 handoff token 值证明。

无法在当前运行时无扰取得的 connector 字段使用 `null`，且写入 capability metadata；关键三条等式任何一条未证明，该 run 不可 seal。
