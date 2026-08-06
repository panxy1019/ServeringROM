# Mooncake patches

Ascend 环境实际使用 vLLM-Ascend 注册的 `MooncakeConnectorV1`，因此有效的
KV transfer 遥测补丁位于
`patches/vllm_ascend/0002-servingrom-mooncake-transfer-telemetry.patch`。
本目录保留用于明确记录：不再向未执行的通用 vLLM Mooncake 路径打补丁。
