# Appendix C. Supporting search and systems comparisons

These controls make the denominators behind the local speedups explicit. They describe measured operating points,
not additive contributions to the final model's Elo.

## Actor overlap and training supply

On the eight-GPU production-shaped workload, keeping more actors alive during a training quantum traded trainer
throughput for ongoing search. The complete-quantum estimate includes the training time and remaining self-play
wait; it is not the trainer-only duration.

| Active actors | Trainer samples/s | Concurrent searches/s | Complete quantum (s) |
| ---: | ---: | ---: | ---: |
| 8 | 21,492 | 506,000 | 117.1 |
| 16 | 17,139 | 606,000 | 112.9 |
| 32 | 9,210 | 742,000 | 111.2 |

With no actors active, trainer throughput was 25,275 samples/s, but no new self-play games arrived during training.
Half-active operation was retained near the measured tradeoff.
The optimal fraction can change with visit budget, model size, and hardware contention.

## Replay-reuse controls

Short controls at reuse ratios four, 6.25, and eight remained matched at their shared evaluation boundaries
despite different update rates. They lasted about 90 minutes, and their full raw curves are no longer preserved.
The completed campaign also changed replay capacity, optimizer, objective weighting, and inference. The selected
ratio of four is therefore a freshness choice, not an isolated final-strength estimate. Configured presentation
credit need not equal the observed number of presentations per distinct replay position. Because evaluation,
publication, visit schedules, and replay growth advance at quantum boundaries, changing reuse also changes their
wall-clock pace.

## Parallel-search operating point

At 1,000 searches per move against the same 20,000-node Stockfish opponent, parallel leaves substantially shortened
the recorded match wall time while reducing measured playing strength. The two timing sources give ranges, not
statistical intervals.

| Parallel leaves | Benchmark Elo | Match wall time (min) | Elo relative to serial |
| ---: | ---: | ---: | ---: |
| 1 | 2,823 | 18.1–18.5 | 0 |
| 4 | 2,804 | 3.4–3.8 | −19 |
| 16 | 2,778 | 1.3–1.8 | −45 |

The relative penalty was much larger at only 100 searches: sixteen parallel leaves cost about 235 Elo. These few
budget-by-parallelism points do not define a universal safe concurrency curve. The terminal strength curve varies
parallelism with budget and must be read as an operating curve.

## Negative search and reuse controls

The learned pre-search allocator captured about 23% of available deep-policy-divergence headroom at nearly matched
spend but trailed non-adaptive online training by roughly 60–100 ladder Elo. A later in-search stopper skipped about
14% of nominal simulations, yet improved generation cadence by only about 3% under actor/trainer overlap. Its
paired strength differences, calculated as baseline minus stopper (+1.7 ± 9.9 and −4.2 ± 10.1 Elo, standard
errors), did not resolve an effect.

Exact graph search avoided only 0.0249% and 0.1769% of neural evaluations at 1,000 and 10,000 searches while being
8.63% and 8.28% slower. A bounded inference cache shared inside each self-play process had a 0.970% hit rate;
disabling it made game updates about 0.88% faster and lowered peak worker memory about 7.12%. A wider unbounded
repeat tracker observed at most about 4% reuse in the tested production-like workloads and itself cost throughput.
These are workload-specific negative findings, not general limits on graph search or caching.

## Throughput comparison boundaries

The 1.86x TensorRT INT8 versus TorchScript search comparison used a production-shaped 400-visit actor workload,
but different checkpoint weights; it is a backend-and-checkpoint result, not an isolated precision effect. The
23.3% replay-admission gain compared separate live 400- and 600-visit stages that also differed in checkpoint
and actor scheduling. Absolute systems rates vary with CPU quota, GPU power, PCIe and NUMA layout, runtime versions,
model shape, batching, and concurrent training.
