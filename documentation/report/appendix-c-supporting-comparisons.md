# Appendix C. Supporting search and systems comparisons

The compact controls below preserve denominators that are easy to lose when comparing local speedups. They are
measured operating points, not an additive attribution of the final model's Elo.

## Actor overlap and training supply

On the eight-GPU production-shaped workload, keeping more actors alive during a training quantum traded trainer
throughput for ongoing search. The complete-quantum estimate includes the training time and remaining self-play
wait; it is not the trainer-only duration.

| Active actors | Trainer samples/s | Concurrent searches/s | Complete quantum (s) |
| ---: | ---: | ---: | ---: |
| 8 | 21,492 | 506,000 | 117.1 |
| 16 | 17,139 | 606,000 | 112.9 |
| 32 | 9,210 | 742,000 | 111.2 |

With no actors active, trainer throughput was 25,275 samples/s, but that excludes a concurrent self-play supply and
is therefore not a complete-quantum comparison. Half-active operation was retained near the measured tradeoff.
The optimal fraction can change with visit budget, model size, and hardware contention.

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
paired strength differences (+1.7 ± 9.9 and −4.2 ± 10.1 Elo, standard errors) did not resolve an effect.

Exact graph search avoided only 0.0249% and 0.1769% of neural evaluations at 1,000 and 10,000 searches while being
8.63% and 8.28% slower. A bounded inference cache shared inside each self-play process had a 0.970% hit rate;
disabling it made game updates about 0.88% faster and lowered peak worker memory about 7.12%. A wider unbounded
repeat tracker observed at most about 4% reuse in the tested production-like workloads and itself cost throughput.
These are workload-specific negative findings, not general limits on graph search or caching.
