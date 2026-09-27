# Appendix C. Supporting search and systems comparisons

The following controls provide the numerical comparisons supporting the search and systems decisions in
Chapters 4 and 5. Each comparison retains its own workload and measurement scale.

## Architecture shape and inference throughput

The width sweep used 12-block global-pooling CNNs with a dense policy head, BF16, batch 512, and one RTX 4070
SUPER. Each width was benchmarked in a separate process against an interleaved width-128 reference.
Table \ref{tab:appendix-c-supporting-comparisons-1} contrasts measured throughput with the inverse-width-squared estimate implied by convolutional
arithmetic. Ratios below are normalized to the unrounded width-128 rate of 103,490 positions/s.

| Depth × width | Positions/s (k) | Measured ratio | Arithmetic ratio |
| --- | ---: | ---: | ---: |
| 12×96 | 133 | 1.284 | 1.778 |
| 12×112 | 95.9 | 0.926 | 1.306 |
| 12×120 | 93.4 | 0.903 | 1.138 |
| 12×128 | 103 | 1.000 | 1.000 |
| 12×136 | 59.7 | 0.577 | 0.886 |
| 12×144 | 63.8 | 0.617 | 0.790 |
| 12×152 | 53.8 | 0.520 | 0.709 |
| 12×160 | 61.8 | 0.597 | 0.640 |
| 12×176 | 52.4 | 0.507 | 0.529 |
| 12×192 | 41.5 | 0.401 | 0.444 |
| 12×224 | 35.5 | 0.343 | 0.327 |
| 12×256 | 31.1 | 0.301 | 0.250 |

The 112- and 120-channel models perform less arithmetic than the 128-channel model but serve fewer positions
per second. The sharp loss at 136 channels likewise exceeds the arithmetic prediction. Width alone is therefore
an unreliable proxy for inference cost.

Table \ref{tab:appendix-c-supporting-comparisons-2} extends the comparison to depth and serving batch size. Ratios are relative to the specified reference
at the same batch size. The width sweep is backed by per-process measurements; the depth/batch comparisons are
transcribed benchmark summaries. Dashes indicate settings not measured.

| Model | Reference | Batch 512 | Batch 320 | Batch 64 |
| --- | --- | ---: | ---: | ---: |
| 20×128 | 14×152 | 1.36 | 1.20 | 0.730 |
| 10×176 | 14×152 | 1.35 | 1.24 | 1.34 |
| 13×160 | 14×152 | 1.23 | 1.13 | 1.10 |
| 14×160 | 14×152 | -- | 1.05 | 1.01 |
| 15×160 | 14×152 | -- | 0.980 | 0.930 |
| 16×160 | 14×152 | -- | 0.918 | 0.960 |
| 14×176 | 14×152 | -- | 0.888 | 0.956 |
| 11×224 | 18×176 | 1.10 | 1.24 | 1.66 |
| 34×128 | 18×176 | 1.06 | 1.16 | 0.556 |
| 19×176 | 18×176 | -- | 0.948 | 0.970 |
| 20×176 | 18×176 | -- | 0.899 | 0.912 |
| 22×160 | 18×176 | -- | 0.995 | 0.838 |
| 4×224 | 12×128 | 0.978 | 1.09 | 2.36 |
| 6×176 | 12×128 | 0.977 | 1.01 | 1.74 |

The 20×128 network is faster than 14×152 at batch 512 but slower at batch 64. Such reversals motivate measuring
candidate models at both self-play and interactive batch sizes rather than extrapolating from parameter count.

The attention comparison controlled bootstrap policy shape, serving precision, and runtime. Its best Smolgen
cell reduced held-out loss by 0.0090 nats relative to the parameter-matched from-to CNN, but delivered only
36.1% of the dense CNN reference's batch-512 forward rate and 45.8% at batch 64, with 5.17× its peak training
memory. These rates use the dense CNN reference, not the parameter-matched from-to comparison.

## Replay-reuse controls

Approximately 90-minute controls at reuse ratios 4, 6.25, and 8 showed comparable strength at their shared
evaluation boundaries despite different update rates. The retained ratio of four favours fresh self-play data.
Its long-run effect is coupled to replay capacity, optimization, and inference changes in the completed campaign.
Reuse also changes the wall-clock pace of evaluation, publication, search-budget schedules, and replay growth,
which advance at quantum boundaries. The configured credit per admitted row therefore describes the training
schedule rather than the exact exposure of every distinct replay position.

## Negative search and reuse controls

The learned pre-search allocator captured approximately 23% of the available improvement in deep-policy
divergence at nearly matched search cost, but trailed non-adaptive online training by roughly 60–100 ladder Elo.
The in-search stopper skipped approximately 14% of nominal simulations while improving generation cadence by
only about 3% under actor-trainer overlap. Its
paired strength differences, calculated as baseline minus stopper (+1.7 ± 9.9 and −4.2 ± 10.1 Elo, standard
errors), did not resolve an effect.

Exact graph search avoided only 0.0249% and 0.1769% of neural evaluations at 1,000 and 10,000 searches while being
8.63% and 8.28% slower. A bounded inference cache shared inside each self-play process had a 0.970% hit rate;
disabling it made game updates about 0.88% faster and lowered peak worker memory about 7.12%. A wider unbounded
repeat tracker observed at most about 4% reuse in the tested production-like workloads and itself cost throughput.
In these workloads, the measured reuse was insufficient to offset the cost of detecting and exploiting it.

## Throughput benchmarks

**Host submission.** The controlled 32-process self-play benchmark increased aggregate search throughput from
512,679 to 617,782 searches/s. On the same node restricted to 24 CPU cores, it increased from 215,265 to
496,036 searches/s. These are rates across the actor workload, including search and inference.

**Neural inference.** The matched runtime benchmark measured 75,889 positions/s with TensorRT FP16 and
40,716 with TorchScript BF16. These count positions evaluated by the network, not aggregate self-play searches.

**Replay delivery.** Compacting 5,000 producer shards into 25 containers increased loader throughput on
2.5 million rows from 3,943 to 32,579 samples/s. In a separate live interval, materialization appended
8,790 positions/s against an arrival rate of 1,365 accepted positions/s.

**Training.** An eight-GPU benchmark with global batch 2,048, bfloat16 autocast, and concurrent self-play
processed 6,252 training samples/s. The actor-overlap sweep in Table \ref{tab:05-systems-optimization-1} is a separate workload; its trainer
and concurrent-search rates should be compared within that sweep.

Table \ref{tab:appendix-c-supporting-comparisons-3} retains the unrounded trainer measurements. Chapter \ref{sec:05-systems-optimization} presents the comparison alongside the scheduling
decision. Cycle times are estimated from measured training duration and search throughput.

| Active actors | Training samples/s | Estimated cycle time (s) |
| ---: | ---: | ---: |
| 0 | 25,275 | -- |
| 8 | 21,492 | 117.1 |
| 16 | 17,139 | 112.9 |
| 32 | 9,210 | 111.2 |

The 1.86x TensorRT INT8 versus TorchScript search comparison used a production-shaped 400-visit actor workload,
with different checkpoint weights, so the comparison includes both backend and model changes. The
23.3% replay-admission gain compared separate live 400- and 600-visit stages that also differed in checkpoint
and actor scheduling. Absolute systems rates vary with CPU quota, GPU power, PCIe and NUMA layout, runtime versions,
model shape, batching, and concurrent training.

## Late-training strength

A retrospective comparison evaluated checkpoints 900, 960, and 1020 at 64 and 400 searches per move
(Table \ref{tab:appendix-c-supporting-comparisons-4}). Each cell used 50 paired openings, giving 100 games.
The 64-search matches used Stockfish 13 at 10,000 nodes (2,470 benchmark Elo); the 400-search matches used
20,000 nodes (2,700 benchmark Elo). Ratings are calculated from each match score against its opponent anchor.

| Checkpoint | Searches | W/D/L | Score | Benchmark Elo |
| ---: | ---: | --- | ---: | ---: |
| 900 | 64 | 18/26/56 | 31.0% | 2,331.0 |
| 960 | 64 | 16/31/53 | 31.5% | 2,335.0 |
| 1020 | 64 | 16/34/50 | 33.0% | 2,347.0 |
| 900 | 400 | 31/41/28 | 51.5% | 2,710.4 |
| 960 | 400 | 33/38/29 | 52.0% | 2,713.9 |
| 1020 | 400 | 26/36/38 | 44.0% | 2,658.1 |

The 64-search estimate increased by 16.0 Elo between the first and last checkpoints, while the 400-search
matches showed no sustained gain. Separately, a late continuation produced 28 successive observations on the
three-rung 64-search ladder over 9.2 hours. The mean of the first 14 observations was 2,373.7 Elo, compared with
2,376.7 for the last 14: a change of 3.0 Elo. These within-protocol comparisons support diminishing returns
from continued training, rather than demonstrating that further improvement is impossible.
