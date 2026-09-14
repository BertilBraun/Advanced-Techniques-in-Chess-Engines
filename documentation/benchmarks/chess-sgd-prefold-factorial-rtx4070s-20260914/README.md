# QAT pre-fold schedule factorial screen

## Conclusion

The historical pre-fold learning-rate schedule with folding delayed from 1,000
to 3,000 optimizer steps was the strongest of the four controlled arms. This
screen provides no evidence that quantization-aware training failed: every arm
learned after folding, and all recorded quantizer ranges remained finite and
stable.

This was a frozen-replay diagnostic, not an online self-play Elo comparison.
Its result supports testing the winning schedule in the next production run;
it does not establish that schedule's playing strength by itself.

## Controlled setup

- Source revision: `11f943d78d30c6e1f9aea239080db23911bed0b3`
- Replay SHA-256: `d668a18e07be099538a79a70b2c65c275a724b7dba56eddeb5629d0a7df52a83`
- Architecture: 12 blocks by 128 channels, QAT enabled
- Optimizer: SGD with Nesterov momentum, weight decay `0.0001`
- Batch size: 2,048
- Gradient clipping threshold: 1.0
- Steps: 10,000 per arm
- Quantizer recalibration: every 500 steps and at folding
- Resources: two RTX 4070 S GPUs per arm
- Seed: `20260913`
- Sample-index digest, identical in all arms:
  `dd68e66932178c3c8c7df14fde897a86bb763ae592599b0a1b0e412a538d5440`

The initial model tensors were compared key by key and were bit-identical in
all four arms. PyTorch's serialized-file hashes differed because its archive
serialization is not byte-canonical.

The historical schedule warmed from zero to `0.1` over 1,000 steps. The
continuous schedule warmed from `0.0001` to `0.02` over 1,000 steps. All arms
used `0.02` immediately after folding. Delayed-fold arms held their pre-fold
peak from step 1,000 through step 3,000, so fold timing was not confounded with
warmup duration.

## Results

| Pre-fold schedule | Fold step | Final total loss | Policy loss | WDL loss | Top-action agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical | 3,000 | **2.77468** | **1.99764** | **0.77704** | **46.851%** |
| Historical | 1,000 | 2.82023 | 2.04132 | 0.77891 | 45.679% |
| Continuous | 1,000 | 2.84940 | 2.06802 | 0.78139 | 44.067% |
| Continuous | 3,000 | 2.87312 | 2.08831 | 0.78481 | 44.189% |

All arms started at total loss `4.30639` and action agreement `6.299%`.
Historical/fold-3,000 beat historical/fold-1,000 by `0.04555` total loss and
1.172 percentage points of action agreement. Historical pre-fold learning
rates beat continuous pre-fold learning rates at either fold timing.

## Fold transition and gradients

Folding produced a one-step gradient spike in every arm:

| Pre-fold schedule | Fold step | Pre-fold total | Fold+1 total | Fold+1 gradient norm |
| --- | ---: | ---: | ---: | ---: |
| Historical | 1,000 | 3.286 at step 999 | 3.359 | 9.36 |
| Continuous | 1,000 | 3.378 at step 999 | 3.899 | 4.14 |
| Historical | 3,000 | 2.942 at step 2,999 | 2.992 | 4.50 |
| Continuous | 3,000 | 3.181 at step 2,999 | 3.627 | 3.97 |

Every fold+1 gradient was clipped. By steps 6,000, 8,000, and 10,000, all arms
reported 100% clipping. Historical/fold-3,000 nevertheless had the lowest final
mean gradient norm at `1.590`, compared with `1.857`, `1.951`, and `2.007` for
historical/fold-1,000, continuous/fold-1,000, and continuous/fold-3,000.

Persistent post-fold clipping deserves a separate controlled investigation,
but it did not prevent continued learning in this screen.

## Quantizer ranges

Ranges were populated before training and were identical at step zero:

- Activation min/median/p95/max: `6 / 6 / 20.200 / 46.905`
- Weight min/median/p95/max: `0.11105 / 0.14760 / 0.18021 / 0.22453`

Activation minima, medians, and maxima remained `6 / 6 / 46.905`; final p95
values were between `21.366` and `23.164`. Folding changed the upper tail of
weight ranges, after which the values remained nearly fixed. No range was
missing, NaN, infinite, or exploding.

Consequently, delaying folding appears to help the historical schedule because
the learned weights and BatchNorm state are better conditioned at the
transition, not because the quantizer observers lack valid ranges at step
1,000.

## Operational result

All four supervisor jobs exited normally with `diverged=false` after 893--903
seconds. Steady throughput was approximately 24.8--25.5k samples/s and GPU use
was balanced across each pair, so the ranking is not a throughput artifact.
The GPUs were idle after completion.
