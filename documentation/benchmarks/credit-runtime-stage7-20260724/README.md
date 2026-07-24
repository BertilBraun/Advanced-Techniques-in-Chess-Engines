# Credit runtime Stage 7 canary

This directory records the July 24, 2026 production-path canaries for the
credit-driven trainer rework. The long training process remained stopped. All
measurements used the existing four-GPU Vast.ai node and the trained V5
checkpoint 319; no clean V6 training run was initialized.

## Mixed self-play and DDP canary

The mixed canary used source revision
`c4b400f4db70ed3bf38140ae59a2c61724370bdc`, four RTX 4070 SUPER GPUs, 16
self-play processes (four per GPU), two MCTS threads and two direct inference
workers per process, 600/150 full/fast searches, no inference cache, and a
simultaneous four-rank DDP quantum at global/local batches 1,024/256.

| Measurement | Result |
| --- | ---: |
| Aggregate MCTS searches/s | 172,221.80 |
| Mean GPU utilization, devices 0–3 | 93.53%–96.86% |
| Peak GPU utilization | 100% on every device |
| Peak GPU memory | 2,385 MiB |
| Minimum measured OOM margin | 9,897 MiB |
| Mean host CPU utilization | 55.79% |
| Peak host memory utilization | 33.6% |
| Peak aggregate self-play RSS | 44,462.88 MiB |
| DDP optimizer throughput under contention | 1,235.25 samples/s |
| DDP optimizer duration, 50 steps | 41.45 s |
| Complete DDP phase duration | 43.72 s |
| DDP peak host RSS | 10,417.04 MiB |

The 180-second run completed too few games to establish a steady-state
positions/hour or games/hour rate. The artifact reports extrapolated values of
1,820 unique positions/hour and 1,402 games/hour, but these are startup-biased
and must not be used as production capacity estimates. Search throughput and
resource measurements are representative of the active mixed workload.

The isolated sampled-diagnostics DDP benchmark from Stage 6 reached 9,651.90
samples/s. Simultaneous self-play therefore reduced trainer throughput by about
7.8 times, but the 50-step quantum still completed in under 44 seconds while
self-play stayed active.

Raw data: [credit-canary-c4b400f-results.json](credit-canary-c4b400f-results.json)

### MCTS thread A/B

Because the mixed canary included DDP contention, it could not establish whether
two MCTS threads helped. A subsequent isolated 60-second A/B used identical
checkpoint, seeds, 16-process topology, 1,024 games/process, and two direct
inference workers:

| MCTS threads/process | Searches/s | Mean GPU utilization | Host CPU capacity | Worker RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **176,828.94** | 87.89%–94.16% | 53.06% | 33,742.98 MiB |
| 2 | 175,415.83 | 85.37%–94.21% | 53.14% | 33,757.66 MiB |

Two threads were 0.80% slower and did not improve batching or resource use.
The locked clean-run configuration therefore uses one MCTS thread and two direct
inference queues per self-play process.

Raw data:

- [mcts-thread-1-summary.json](mcts-thread-1-summary.json)
- [mcts-thread-2-summary.json](mcts-thread-2-summary.json)

## In-place model refresh

Ten measured refreshes followed two warmups while retaining 1,024 populated
MCTS roots.

| Measurement | Result |
| --- | ---: |
| Mean refresh latency | 195.28 ms |
| Median refresh latency | 190.55 ms |
| P95 refresh latency | 212.14 ms |
| RSS growth across measured refreshes | 0.145 MiB |
| GPU-memory growth | 0 MiB |
| Peak process RSS | 1,317.95 MiB |
| Peak GPU memory | 379 MiB |
| Retained root visits after every refresh | 65,536 |
| Retained live arena nodes after every refresh | 66,560 |
| Retained child records after every refresh | 1,836,928 |

Raw data: [credit-canary-c4b400f-refresh.json](credit-canary-c4b400f-refresh.json)

## Complete evaluation-suite canary

The evaluation benchmark used evaluator source revision
`e7716378421a8a4309d4b80370e79fc9eeb01a86`, V5 checkpoint 319 from source
revision `1806a2df9de9683f5261c87b6c259f797fad0cce`, the V6 evaluation protocol,
100 games per match, 64 MCTS searches per move, and at most 16 concurrent tasks
cycled evenly over the four GPUs.

The suite completed successfully in **293.68 seconds (4m53.68s)**. Fixed-node
Stockfish completed in about 2.5 minutes, all Stockfish skill-level tasks
completed in about 3.3 minutes, and the historical model matches completed by
4.9 minutes. No task crashed, no CUDA process remained, and peak observed GPU
allocation during status sampling was about 1.3 GiB per device.

The benchmark intentionally used a historical V5 model directory while applying
the current V6 evaluation settings. Credit production converts evaluation
checkpoint ordinals to 50-step publication versions in
`credit_evaluation_arguments`; this benchmark measured the configured workload
duration and device placement, not the scheduler's early-run availability of
historical checkpoints.

The fixed holdout was regenerated in replay schema 3 from 50 Lichess Elite
October 2024 games. It contains 3,782 positions, all are natural-outcome
eligible, and all WDL classes are present. Its SHA-256 is
`3d4ac41f5baed318e014cffecebe7d00971425541c2a57be255a711ab980b4b0`.
Checkpoint 319 scored:

| Holdout metric | Result |
| --- | ---: |
| Policy accuracy @1 | 44.71% |
| Policy accuracy @5 | 83.58% |
| Policy accuracy @10 | 93.20% |
| WDL cross-entropy | 1.30799 |

This top-1 result is not the same series as the approximately 39.x% V5
TensorBoard result. A direct checkpoint-319 recheck produced 39.77% on the old
5,592-position holdout and 44.69% on the new 3,782-position holdout. The old
generator randomly retained 75% of positions and added left-right symmetric
variations; the schema-3 generator includes every canonical position exactly
once. The difference is a holdout-distribution change, not a retrospective model
improvement. New V6 metrics must start a new TensorBoard series and retain the
dataset hash in provenance.

Artifacts:

- [credit-canary-e771637-evaluation-results.json](credit-canary-e771637-evaluation-results.json)
- [credit-canary-e771637-evaluation.log](credit-canary-e771637-evaluation.log)
- [match-vs-stockfish-fixed-nodes-iteration-319.json](match-vs-stockfish-fixed-nodes-iteration-319.json)
- [chess-elite-2024-10-50-schema3-manifest.json](chess-elite-2024-10-50-schema3-manifest.json)

## Validation and limitations

- Fresh Release/O3 LibTorch build with timing disabled: successful.
- Native CTest: 11/11 passed.
- Fresh-extension focused Python suite: 83 passed.
- Deterministic credit integration covers shard ingestion, exact credit,
  four-rank partitioning, 50 optimizer steps, publication/hash acknowledgement,
  retained-tree refresh, eviction, restart before and after commit, and
  evaluation crash/retry/interruption persistence.
- The mixed canary was deliberately short and cannot establish steady-state game
  production.
- The evaluation canary used a trained V5 checkpoint because the clean V6 run
  has not been initialized.
- Training must remain stopped until the final source/configuration review and
  explicit clean-run approval.
