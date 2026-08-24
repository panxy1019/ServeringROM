# ServingROM Step 15C-2B.1 三阶段库存与转移流重构

## 状态

- `observed_flow_conservation_pass=true`
- `transition_flow_model_trained=true`
- `control_rom_ready=false`
- 未启动 1P2D、未重新采集、未读取 held-out、未实现 MPC。

## Validation Observed-Flow Replay

| 阶段 | request NRMSE | request exact | token NRMSE | token exact |
|---|---:|---:|---:|---:|
| handoff | 0.000000 | 100.00% | 0.000000 | 100.00% |
| waiting | 0.000000 | 100.00% | 0.000000 | 100.00% |
| running | 0.000000 | 100.00% | 0.000000 | 100.00% |

## Transition/Service Flow Model

### validation
- `handoff` requests/tokens：`0.992396` / `0.992914`
- `waiting` requests/tokens：`0.989236` / `0.990235`
- `running` requests/tokens：`0.491193` / `0.275600`
### test
- `handoff` requests/tokens：`0.992552` / `1.002710`
- `waiting` requests/tokens：`0.995188` / `0.991368`
- `running` requests/tokens：`0.483225` / `0.263241`

## 缺失字段与下一步

- `existing sealed telemetry is sufficient for three-stage conservation replay`
- `evaluate stage-flow model and redesign only failed transition heads`
