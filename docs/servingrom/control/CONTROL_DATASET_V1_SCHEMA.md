# ServingROM Control Dataset v1 Schema

## Scope

Round 14.2 contains 36 complete runs. A run belongs wholly to one split:

- seed 101: train
- seed 202: validation
- seed 303: test

The matrix is `2 workloads × 3 loads × 2 arrival processes × 3 split seeds`.
The frozen topology is Prefill TP2 + Decode A TP2 + Decode B TP2.

## Fast coordinates

Every 200 ms row stores `(X_k, D_k, U_k, X_next_k)`:

- `X[1804]`: frozen ServingROM-v2 full-order state;
- `D[31]`: exogenous workload disturbance;
- `U[1]`: `rho_A`, sourced only from
  `actuator_applied.payload.effective_value`;
- `X_next[1804]`: the next fast state.

Audit-only control fields are stored separately as `U_aux`: previous U, delta U,
time since change, command ID/generation and applied wall time. Realized request
ratio, token ratio, queue imbalance and Decode imbalance are responses, not U.

## Slow coordinates

Every 25 fast rows produce one 5 s KPI row. It contains conservation-based
throughput/goodput/latency/token totals, queue/running/remaining-token integrals,
KV metrics, Decode imbalance, `U_start/U_mean/U_end/delta_U`, and realized route
diagnostics. It is an output table only; no 5 s State ROM is defined.

## Expected cardinality

- all: 36 runs, 108000 fast rows, 4320 slow rows;
- each split: 12 runs, 36000 fast rows, 1440 slow rows.

Raw telemetry and each run remain independently sealed. Dataset construction is
read-only and refuses schema drift, invalid run quality, partial runs, or an
existing output directory.
