# Appendix C. Supporting search and systems comparisons

The following controls provide the numerical comparisons supporting the search and systems decisions in
Chapters 4 and 5. Each comparison retains its own workload and measurement scale.

## Actor overlap and training supply

The eight-GPU overlap sweep measures optimizer throughput and concurrent search as the number of active actors
increases. Complete-quantum time includes both the training block and the remaining wait for self-play data.

| Active actors | Trainer samples/s | Concurrent searches/s | Complete quantum (s) |
| ---: | ---: | ---: | ---: |
| 8 | 21,492 | 506,000 | 117.1 |
| 16 | 17,139 | 606,000 | 112.9 |
| 32 | 9,210 | 742,000 | 111.2 |

Pausing all actors yielded 25,275 training samples/s with no concurrent game production. Half-active operation
retained most of the complete-cycle benefit while reducing contention during training. Its balance depends on
visit budget, model size, and available hardware capacity.

## Replay-reuse controls

Approximately 90-minute controls at reuse ratios 4, 6.25, and 8 showed comparable strength at their shared
evaluation boundaries despite different update rates. The retained ratio of four favours fresh self-play data.
Its long-run effect is coupled to replay capacity, optimization, and inference changes in the completed campaign.
Reuse also changes the wall-clock pace of evaluation, publication, search-budget schedules, and replay growth,
which advance at quantum boundaries. The configured credit per admitted row therefore describes the training
schedule rather than the exact exposure of every distinct replay position.

## Parallel-search operating point

At 1,000 searches per move against the same 20,000-node Stockfish opponent, parallel leaves substantially shortened
the recorded match wall time while reducing measured playing strength. The two timing sources give ranges, not
statistical intervals.

| Parallel leaves | Benchmark Elo | Match wall time (min) | Elo relative to serial |
| ---: | ---: | ---: | ---: |
| 1 | 2,823 | 18.1–18.5 | 0 |
| 4 | 2,804 | 3.4–3.8 | −19 |
| 16 | 2,778 | 1.3–1.8 | −45 |

At 100 searches, sixteen parallel leaves reduced the central strength estimate by approximately 235 Elo,
compared with 45 Elo at 1,000 searches. The tested points support budget-dependent parallelism but are insufficient
to determine a general schedule. The final strength curve uses the operating settings reported in Chapter 8.

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
processed 6,252 training samples/s. The actor-overlap sweep in Table C1 is a separate workload; its trainer
and concurrent-search rates should be compared within that sweep.

The 1.86x TensorRT INT8 versus TorchScript search comparison used a production-shaped 400-visit actor workload,
with different checkpoint weights, so the comparison includes both backend and model changes. The
23.3% replay-admission gain compared separate live 400- and 600-visit stages that also differed in checkpoint
and actor scheduling. Absolute systems rates vary with CPU quota, GPU power, PCIe and NUMA layout, runtime versions,
model shape, batching, and concurrent training.
