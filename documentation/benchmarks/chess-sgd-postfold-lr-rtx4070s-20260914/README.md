# QAT post-fold learning-rate sweep

## Conclusion

A post-fold target learning rate of `0.08` was the strongest of the four
controlled arms. Training and held-out total loss improved monotonically as the
target increased from `0.02` to `0.08`. All arms remained numerically stable.

This was a frozen-replay diagnostic rather than an online self-play Elo test.
It selects `0.08` for the next progressive production run, where actual playing
strength will be measured on the 20-minute evaluation cadence.

## Controlled setup

- Source revision: `a47838526104e7329896f4788ec262eb66d5b856`
- Replay SHA-256: `d668a18e07be099538a79a70b2c65c275a724b7dba56eddeb5629d0a7df52a83`
- Architecture: 12 blocks by 128 channels, QAT enabled
- Pre-fold schedule: zero to `0.1` over 1,000 steps, then hold `0.1`
- Fold: step 3,000
- Post-fold schedule: `0.02` to the arm target over 500 steps
- Targets: `0.02`, `0.04`, `0.06`, and `0.08`
- Optimizer: SGD with Nesterov momentum, weight decay `0.0001`
- Batch size: 2,048; maximum gradient norm: 1.0
- Steps: 10,000 per arm; two RTX 4070 S GPUs per arm
- Seed: `20260913`
- Shared sample-index digest:
  `dd68e66932178c3c8c7df14fde897a86bb763ae592599b0a1b0e412a538d5440`

All 227 initial tensors were compared elementwise and were bit-identical across
the four arms.

## Final results

| Post-fold target | Total loss | Policy loss | WDL loss | Top-action agreement |
| ---: | ---: | ---: | ---: | ---: |
| **0.08** | **2.72231** | **1.94303** | 0.77928 | **48.877%** |
| 0.06 | 2.73265 | 1.95950 | 0.77315 | 48.853% |
| 0.04 | 2.74597 | 1.97580 | **0.77017** | 48.145% |
| 0.02 | 2.76752 | 1.99281 | 0.77470 | 48.047% |

Final training total loss followed the same monotonic order: `2.72577`,
`2.73959`, `2.75471`, and `2.77635` for targets `0.08`, `0.06`, `0.04`, and
`0.02`, respectively.

## Gradients and quantizer ranges

Final interval mean gradient norm and clipped-step fraction were:

| Target | Mean gradient norm | Clipped steps |
| ---: | ---: | ---: |
| 0.02 | 1.608 | 100.0% |
| 0.04 | 1.292 | 100.0% |
| 0.06 | 1.056 | 68.6% |
| 0.08 | 0.898 | 14.8% |

The higher rates moved more quickly into a lower-gradient region rather than
causing divergence. Activation min/median/max remained exactly
`6 / 6 / 46.905`. The final weight p95 increased with target learning rate
from `0.2182` at `0.02` to `0.2411` at `0.08`; the `0.08` maximum was `0.8781`.
All ranges remained finite and stable through step 10,000.

All four jobs completed normally in 856--878 seconds with `diverged=false`, no
NaNs or tracebacks, and balanced GPU utilization.
