# ServingROM Step 15C-2B.2 Age-Structured Semi-Markov Transition ROM

## 状态

- `stage_inventory_ready=true`
- `age_transition_ready=false`
- `transition_pipeline_ready=false`
- `control_rom_ready=false`
- 未启动 1P2D、未采集新 run、未读取 Round 14.3、未实现 MPC。

## Validation

| Stage | Request differential NRMSE | Token differential NRMSE |
|---|---:|---:|
| handoff | 0.999531 | 0.999996 |
| waiting | 0.999519 | 0.999996 |

## Running attribution

- `oracle_admission` request/token：`0.477393` / `0.214088`
- `predicted_admission` request/token：`0.492884` / `0.277845`

## 结论

H1 fails the preregistered stage-inventory gate and does not materially improve H0. Both transit dwell distributions are predominantly below 200ms, so boundary snapshots erase the within-window cohort phase. H2 is not executed because symmetric common-load correction cannot restore that missing sub-window phase. Stop before actuator realization; next audit should test route phase/request-size/KV-byte stratified sub-window hazards using existing timestamps.
