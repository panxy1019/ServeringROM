# ServingROM Formal Composite Control Program v1

## Frozen topology

The topology is **Prefill TP2 + Decode A TP2 + Decode B TP2**. Any reference to
"Decode A TP3" is a documentation typo and is not an executable configuration.
The deployment remains `FULL_DECODE_ONLY` with async scheduling enabled.

## Program

The 600-second measurement program is fixed:

| Interval | Program | Levels / dwell |
|---|---|---|
| 0-60 s | initial | `rho_A=0.5` |
| 60-240 s | rate-limited PRBS | `0.3/0.5/0.7`, 15 s |
| 240-420 s | random dwell | `0.3/0.5/0.7`, 10/15/20 s |
| 420-540 s | step response | `0.5→0.7→0.5→0.3→0.5→0.7` |
| 540-600 s | recovery | `rho_A=0.5` |

Every change satisfies `abs(delta_U)<=0.2`. The random program uses a control
RNG derived independently from the arrival RNG. A rejected command, CAS error,
generation error, or safety fallback invalidates the entire run.

`U` is sourced exclusively from the effective value of `actuator_applied`.
Realized request and token ratios are diagnostics, never control inputs.
