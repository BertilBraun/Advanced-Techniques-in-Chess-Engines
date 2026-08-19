# Chess fixed-batch overfit benchmark (RTX 3090)

This evidence was collected on 2026-08-19 from the stopped
`vast-chess-8gpu-optimal` run at generation 94. The source checkout was
`6b00d893b677d8d586fcf5efcedeb7250b2b1024`; the benchmark tool and tests in
this commit were copied into an isolated worktree before execution. GPU 0 was
used, with bfloat16 precision, AdamW, learning rate 0.005, a 256-row immutable
training batch, and a disjoint 256-row holdout batch.

## Architecture comparison

All architectures used the complete production objective. Time-to-floor is
the first 25-step observation whose training loss was at most 1% above the
theoretical entropy floor of the soft targets.

| Model | Parameters | Steps to floor | Time to floor | Steps/s | Final holdout total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention 6x96 | 482,615 | 225 | 7.63 s | 29.48 | 5.2613 |
| Attention 8x128 | 1,102,775 | 250 | 9.54 s | 26.20 | 4.9799 |
| CNN 8x88 | 1,110,711 | 200 | 6.03 s | 33.16 | 5.5952 |

The test establishes that both attention models can fit the exact production
targets and that the action/auxiliary target wiring has usable gradients. It
does not establish online playing strength. The matched-size attention model
was about 21% slower per optimizer step than the CNN in this microbenchmark,
but had the lowest holdout loss.

## Auxiliary objective comparison

The 1.1-million-parameter attention model was also run with production,
half-strength auxiliary, primary-only, and legal-moves-disabled objectives.
All heads remain present and are computed in every profile, so timing differences
between profiles are mostly measurement noise. At convergence, the primary
holdout loss (`policy + WDL`) was:

| Objective profile | Primary holdout loss |
| --- | ---: |
| Production | 4.5717 |
| Half auxiliary weights | 4.4915 |
| Primary only | 4.6259 |
| Without legal-moves loss | 4.6915 |

This small fixed-batch test supports trying half-strength auxiliary weights.
It does not support removing the legal-moves objective as an optimization.
The `equal-100-*.json` files contain the additional equal-length runs used to
check that conclusion without comparing different stopping steps.
