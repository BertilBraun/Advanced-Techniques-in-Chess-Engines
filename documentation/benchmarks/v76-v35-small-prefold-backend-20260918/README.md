# V76 v35 small pre-fold backend throughput

Date: 2026-09-18

## Result

At the production actor topology, TensorRT INT8 improves v35 scaled-post 12x128 search throughput by 14.4% over
TensorRT FP16. Both arms used the same generation-30 pre-fold QAT checkpoint and matched search settings.

| Backend | Searches/s total | Searches/s/GPU | Model positions/s total | Mean batch | Mean active GPU utilization | Mean active power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorRT FP16 | 1,283,300 | 160,412 | 1,279,032 | 319.44 | 93.84% | 189.09 W |
| TensorRT INT8 | 1,467,691 | 183,461 | 1,462,429 | 319.41 | 92.70% | 188.20 W |
| INT8 relative | **1.144x** | **1.144x** | **1.143x** | 1.000x | 0.988x | 0.995x |

The earlier live-run credit comparison did not isolate inference: it included different architecture, QAT training
cost, evaluation stalls, game completion, and replay materialization. The matched backend benchmark shows that INT8
does accelerate the small model, although substantially less than the 39.1% measured for 14x160.

## Matched workload

- Source revision: `bc23989bff368600c611153506171c8719148cd8`.
- GPUs: eight RTX 4070 SUPER devices.
- Topology: four self-play actor processes per GPU, one inference thread per actor, 512 games per actor.
- Search: 400 baseline visits, automatically selected parallel-search count 2.
- Inference: batch cap 320, two outstanding batches per inference worker, channels-last layout.
- Measurement: two warm-up batches followed by a nominal 60-second synchronized arm.
- Checkpoint: v75 generation 30, v35 scaled-post 12x128, pre-fold QAT phase, 15,000 optimizer steps.
- FP16 configuration SHA-256: `fa981fd2641ab3e3899879004eb8978340eaa6f4ffd66fd213af3f47263c907b`.
- INT8 configuration SHA-256: `72225de0098f4465c1698e085367a27a1560143c474f1cca753a4d2b9f8e18a5`.

All 32 workers in both final arms completed without stderr. FP16 completed 81,510,400 searches in 63.516 seconds;
INT8 completed 94,208,000 in 64.188 seconds.

## Fidelity and stability context

On 320 deterministic real positions at generation 30:

- float versus fake quant: 96.5625% legal-policy top-1 agreement, mean KL 0.0005873, expected-value MAE 0.01779;
- fake quant versus QDQ ONNX: 97.5% top-1 agreement, mean KL 0.0004965, expected-value MAE 0.01608;
- float ONNX versus TensorRT FP16: 99.6875% top-1 agreement, mean KL 0.00000634, expected-value MAE 0.00206.

The v75 training run remained finite through generation 30 and reached search Elo 1,033 at 43:11 effective elapsed
time. It trailed the matched-time v70 FP16 small control by about 60 search Elo and 80 policy Elo, so the learning
evidence is acceptable but not equivalent. The search-relevant quantization deviations are much smaller than the
earlier folded-model discrepancy.

## Decision

INT8 is a modest win during the short 12x128 stage and a large win during the longer 14x160 stage. With folding
disabled, the combined throughput, fidelity, and finite-learning evidence supports using pre-fold INT8 self-play
for the final progressive run, while retaining evaluation gates and an early Elo abort policy.
