# Direct self-play inference benchmark

This benchmark evaluates whether the optimized interactive evaluation pipeline transfers to
self-play. The answer is yes: batching one search from each of many independent games through the
direct reusable inference pipeline is substantially faster than the general cached client, even
with fewer simultaneous searches in every tree.

## Result

The quality-oriented direct configuration uses 512 games, one outstanding search per game, two
model workers, batches of 64, and one outstanding batch per worker. Across three 30-second seeded
runs it reached a median **22,028 searches/s**, 66.9% above the **13,195 searches/s** median of the
current cached configuration. The baseline used 192 games, four simultaneous searches per tree,
four MCTS threads, a 256-position client batch limit, a 1,000 microsecond collection timeout, and a
3,000,000-position cache.

| Configuration | Searches/s runs | Median searches/s | Parallel searches/tree | Process CPU | Mean model batch |
|---|---:|---:|---:|---:|---:|
| Cached general client | 13,090; 13,506; 13,195 | 13,195 | 4 | 254% | 76.6 |
| Direct pipeline | 22,088; 21,661; 22,028 | **22,028** | **1** | 182% | 61.4 |

The direct configuration therefore improves throughput while removing within-tree virtual-loss
staleness. A faster but lower-quality-width pilot reached 22,934 searches/s with 192 games and four
parallel searches per tree, only 4.1% above the selected width-one result. That small margin does
not justify four-way tree parallelism during self-play.

The cache did not drive the baseline result: its hit rate was 4.2-5.0% in the three diversified
runs. Direct inference currently has no cache, so long synchronized-opening workloads can have a
different tradeoff. Searches/s must always be reported together with model positions/s and cache
hit rate.

## Where the speedup comes from

The previous self-play path creates a tensor and promise/future per cache miss, enqueues generic
requests, gathers them with a timeout, stacks CPU tensors, copies full Torch outputs back to the
CPU, resolves per-row futures, and then validates and converts policies through Torch operations.
Four C++ search slices block on synchronous `inferenceBatch` calls and rely on their shared request
queue to form model batches.

The direct path has one serial owner for all independent game trees. It selects at most one leaf
per tree, applies virtual loss, encodes directly into reusable pinned int8 batch slots, and submits
explicit batches to persistent model replicas on dedicated CUDA streams. Completion processing
reads contiguous float pointers, filters only legal moves, expands the owning arena tree, and
removes virtual loss. Inference workers never access trees. Fixed slots remove per-position
tensors, promises, futures, timeout collection, `torch::stack`, and repeated allocations.

Self-play retains its existing indexed fixed-capacity `SearchTree`, lazy child materialization,
subtree reuse, discounting, reroot reclamation, Dirichlet noise, randomized playout caps, and
Python game/dataset lifecycle. The new scheduler fills batches across games rather than issuing
many simultaneous searches into one tree. Model updates reconstruct the direct worker set at the
existing iteration boundary.

## Bottleneck

The selected direct runs used a median 79.6% of the two workers' aggregate active time. One
representative run spent 25.0 seconds waiting for inference, compared with 2.03 seconds selecting,
0.62 seconds encoding, 1.34 seconds processing results, and 0.46 seconds expanding/backing up.
The process used about 1.82 CPU cores, down from 2.54 for the baseline.

Quarter-second `nvidia-smi` samples include model loading and warmup. Direct runs had 66.7-67.8%
mean and 84-85% median GPU utilization, versus 44.4% mean and 48% median for the instrumented
baseline. Mean direct power was 96-98 W versus 71 W. On this eight-logical-core RTX 3060 node the
remaining limit is GPU/model inference, not CPU tree work. Faster GPU inference would help;
additional CPU cores are not currently needed for one direct process.

Two processes with one model worker and 256 games each reached only 18,671 aggregate searches/s.
One process with two replicas is the better topology on this node. The earlier four-GPU RTX 4070
SUPER topology should be retuned rather than inheriting its old four-process-per-GPU layout;
multiple direct workers in every old process would oversubscribe each GPU.

## Methodology and provenance

- Hardware: NVIDIA GeForce RTX 3060 12 GiB, eight logical CPUs, 32 GiB host RAM.
- Software: Linux 6.8.0-124, GCC 13.3, PyTorch 2.12.0+cu130, driver 595.71.05.
- Build: Release, `-O3 -march=native -flto`, timing instrumentation disabled.
- Model: deterministic untrained 12-layer, 112-channel TorchScript network,
  SHA-256 `d85f782970cbd1cdc07f0d2c71ce3b9480517720fe0a858f61eaa99341d36e7d`.
- Workload: 600 full and 150 fast searches/ply, one warmup step, three independent seeds.
- Primary metric: exact native root-visit deltas, including terminal backups and cache hits.
- Direct module SHA-256: `fe47cde3467681b67fc01e713ab212a2932c38b8ed590f02372f57d6dbdaa623`.

The benchmark duration is checked between complete lockstep game-update steps, so elapsed times are
31.5-33.3 seconds rather than exactly 30 seconds. Throughput uses each run's measured elapsed time.
Raw JSON and GPU telemetry CSV files are stored beside this document. The model is untrained, so
the runs validate throughput and structural invariants, not playing strength.

## Remaining work

The direct implementation is production-configurable but not enabled in an existing four-GPU run
configuration. Before a long training run, sweep one versus two processes per GPU and one versus
two model workers on the actual RTX 4070 SUPER node while keeping width one and enough games to
fill every direct batch. A small cache or opening de-synchronization can be evaluated separately if
the production hit rate is materially higher than the 4-5% measured here.
