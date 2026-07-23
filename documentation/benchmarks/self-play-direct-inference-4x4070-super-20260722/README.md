# Production self-play throughput on four RTX 4070 SUPER GPUs

This benchmark selects the self-play topology for `complete-training-run-v5` on the actual
four-GPU training node. The selected direct-inference topology sustains a median **233,141
searches/s**, compared with **142,843 searches/s** for the previous production path: a **63.2%
increase**. It also reduces process CPU demand from 58.9 to 32.1 logical cores and aggregate
worker RSS from 52.7 to 24.3 GiB.

## Selected topology

Each GPU runs two self-play processes. Each process owns 2,048 independent games, four persistent
model replicas/inference workers, a batch size of 64, and room for two outstanding batches per
worker. Every tree has one outstanding search. Eight processes therefore maintain 16,384 games
and up to 1,024 in-flight leaves across the node without using parallel searches within a tree.

During a four-GPU DDP training phase, Commander pauses one of the two self-play processes on each
GPU. The remaining process continues generating games while the four DDP ranks train on devices
`(3, 2, 1, 0)`. This is a deterministic half-pause rather than relying on GPU oversubscription to
throttle self-play.

## Implementation details

The old general inference client accepted individual tensor requests backed by promises and
futures. MCTS threads submitted misses to a shared queue, a batching worker waited for either its
batch limit or timeout, stacked CPU tensors, copied them to CUDA, copied complete policy and value
outputs back, and resolved each request. Every search thread blocked synchronously on its future.
That abstraction is flexible and supports caching, but self-play paid allocation, synchronization,
timeout, stacking, and result-dispatch costs for every evaluated leaf.

The direct scheduler instead has one tree owner and four inference workers per process:

1. The owner visits independent games and selects at most one leaf per tree.
2. It applies virtual loss and encodes the leaf directly into a reusable pinned `int8` slot.
3. A full or currently dispatchable 64-position slot is handed to a persistent model replica on
   its own CUDA stream. Each worker may own two outstanding batches.
4. The owner continues selecting from other games while CUDA inference is running.
5. Completed slots are consumed as soon as each result becomes available; the owner filters legal
   policy entries, expands the corresponding arena tree, backs up the value, removes virtual loss,
   and immediately reuses the slot. It never waits for all inference workers to finish together.

Inference workers do not mutate search trees. The tree owner remains the only writer, which avoids
locks around expansion and backup while completion order remains asynchronous. Each game retains
its indexed fixed-capacity arena, lazy child materialization, subtree reuse, discounting,
Dirichlet noise, randomized full/fast search targets, and existing Python game/dataset lifecycle.
Model updates rebuild the worker replicas only at the existing iteration boundary.

The production configuration also makes parallel-search width explicit. Width one means the
selected action in a tree is never distorted by another outstanding virtual loss from that same
tree. Parallelism comes from thousands of games, not speculative searches contending in one tree.

| Topology | Searches/s | CPU | Worker RSS | Mean batch |
|---|---:|---:|---:|---:|
| Previous: 10 processes/GPU, 3 MCTS threads, 96 games, width 4 | **142,843** median | 58.9 cores | 52.7 GiB | 46.0 |
| 1 process/GPU, 2 workers, 512 games, queue 1 | 72,468 | 7.2 cores | 6.7 GiB | - |
| 2 processes/GPU, 2 workers, 512 games, queue 1 | 129,776 | 14.8 cores | 13.2 GiB | - |
| 2 processes/GPU, 3 workers, 768 games, queue 1 | 160,809 | 20.4 cores | 15.2 GiB | - |
| 3 processes/GPU, 2 workers, 512 games, queue 1 | 140,591 | 22.7 cores | 19.7 GiB | - |
| 2 processes/GPU, 3 workers, 1,536 games, queue 2 | 220,665 | 25.9 cores | 20.6 GiB | - |
| **2 processes/GPU, 4 workers, 2,048 games, queue 2** | **233,141 median** | **32.1 cores** | **24.3 GiB** | **60.5-60.9** |
| 2 processes/GPU, 5 workers, 2,560 games, queue 2 | 236,351 single run | 38.8 cores | 27.8 GiB | - |
| 2 processes/GPU, 4 workers, 2,048 games, batch 128, queue 1 | 199,190 | - | - | - |
| 4 processes/GPU, 2 workers, 1,024 games, queue 2 | 184,025 | 33.9 cores | 33.6 GiB | 60.6 |

The five-worker pilot did not beat the four-worker configuration's first run and consumed more
CPU and memory. Batch 128 with queue depth one regressed because it removed useful overlap. Four
processes per GPU reached 100% GPU utilization in spot samples, but was 21% slower than the
selected median; utilization percentage is therefore not the optimization target.

## Repeated and sustained results

The selected topology produced 243,199, 224,640, and 233,141 searches/s in independent seeded
30-second runs. A separate 120-second run sustained 242,186 searches/s and completed 31,970,009
searches. Its processes used 32.7 logical cores and 25.1 GiB RSS, with a mean inference batch of
59.76 positions.

Quarter-second GPU telemetry for the sustained run averaged 81.7-89.2% utilization. Those means
include staggered process shutdown and lockstep game-update tails. Active snapshots repeatedly
reached 100% utilization and approximately 109-110 W on all four cards. Adding process pools to
smooth the tail increased overhead and reduced the metric that matters: completed searches/s.

## Comparison with interactive evaluation

The integrated interactive search on the separate RTX 3060 node reached about 8,647 searches/s
for one tree with two inference workers and batches of 50. That number is not directly comparable
to the four-GPU aggregate because the hardware and workload differ. The transferable optimization
is the scheduler design: persistent model replicas, reusable pinned slots, explicit full batches,
and completion-driven result processing. Self-play gains additional throughput by batching one
leaf from each of thousands of independent trees rather than issuing many virtual-loss searches
into one tree.

## Direct inference for training evaluation

The periodic training evaluator originally remained on the general MCTS inference client even
after the optimized interactive and self-play schedulers were merged. Search-based matches against
previous, historical, reference, random, and Stockfish opponents created a non-caching client with
a 16-position maximum batch and a 500-microsecond collector. Dataset metrics and policy-only games
do not use MCTS and were already on the appropriate paths.

Training evaluation now uses the direct multi-tree MCTS scheduler. This is deliberately not the
single-tree interactive wrapper: a 100-game paired match presents about 50 same-side roots at each
ply, so the multi-tree scheduler can batch one leaf from every independent game while preserving
one outstanding search per tree and an exact 64-visit move budget. Candidate and neural opponent
each receive their own snapshot-scoped pipeline inside the spawned evaluation task.

The conservative production evaluation topology is one inference worker, batch 64, one outstanding
batch, and parallel width one per model. Only four evaluation tasks may run concurrently, assigning
at most one task to each physical GPU. A second worker cannot normally be fed at width one because
the roughly 50 active roots are all waiting after the first batch; it would add a model replica
without increasing useful concurrency. Direct evaluation and evaluation caching are validated as
mutually exclusive.

A same-node proxy benchmark used checkpoint 196, 50 parallel games, width one, and 64 fast/65 full
searches per ply. The previous four-thread, batch-16 non-caching client completed **2,601
searches/s** with a mean model batch of 10.32. The direct one-worker, batch-64, queue-one scheduler
completed **7,270 searches/s** with a mean batch of 40.76: a **2.80x throughput increase**, while
process CPU fell from 163% to 99%. Each run covered ten measured seconds after warmup and no game
state or tree was shared between the legacy and direct processes.

Match-report provenance now records `direct_multi_tree`, inference workers, inference batch size,
outstanding depth, parallel searches per tree, one serial tree-owner thread, and the unchanged
`mcts_root_visits=64` search limit. Legacy configurations retain the general cached/non-cached path
when direct evaluation is not configured.

## Training configuration

The four-GPU trainer uses one NCCL rank per GPU, devices `(3, 2, 1, 0)`, global batch 2,048, and
local batch 512. The production DDP path previously measured 22,599 samples/s in isolation and
15,026 samples/s while the old half-self-play workload was active. The selected self-play layout
leaves substantially more CPU and host memory available during overlap than that old workload.

The initial four-GPU restart doubled the learning-rate schedule together with the global batch.
Observed optimization behavior showed that linear scaling was too aggressive for this run, so the
schedule is restored to `0.005` from iteration 0, `0.0035` from iteration 50, and `0.002` from
iteration 100 onward. The checkpoint-216 restart therefore resumes at **0.002**. Optimizer state is
restored normally; only the configured per-step rate changes.

## Overnight evaluation stability

Monitoring through checkpoint 216 exposed a native/Python terminal-rule mismatch. The native
board classified king and two knights versus king as insufficient material, while python-chess
correctly kept the position playable. Direct MCTS consequently returned a valid terminal result
with no child visits, and the Python evaluator attempted to normalize a zero-mass policy. The
native rule now treats only zero or one knight without other material as insufficient. Native
tests cover both K+N versus K (terminal) and K+NN versus K (non-terminal), and policy conversion
rejects zero visit mass instead of producing NaNs.

Paired evaluation represents a native terminal search as an explicit terminal decision and
adjudicates it as a draw, rather than encoding terminal state as a fake move policy. Move policies
are validated for finite positive mass before move selection.

The same evaluation window revealed a poisoned low-skill Stockfish UCI session returning an
illegal stale best move. Skill-level matches now force a `ucinewgame` boundary when switching
boards, disable pondering explicitly, close and recreate the engine after an `EngineError`, and
retry the affected position once. Fixed-node monitoring also uses explicit game boundaries.
Finally, every paired monitoring game is capped at 400 plies. This prevents one pathological
low-skill ending from occupying the single evaluation slot indefinitely; a capped game is recorded
as a draw.

## Restart and pause behavior

The run configuration validates that the 2,048 global batch equals four ranks times the 512 local
batch. It also includes the four direct-inference CPU workers when checking CPU oversubscription.
When training begins, Commander pauses process IDs 1, 3, 5, and 7: exactly one process on each GPU.
At the phase boundary it resumes those same processes. Paused processes keep their compact model
replicas resident, avoiding model reload churn; measured self-play and trainer allocations fit
comfortably within each 12 GiB card.

The restart is deliberately gated on three matching values: committed source revision, canonical
run-configuration SHA-256, and the explicit production approval record. Checkpoint 190 includes
the eager model, TorchScript model, optimizer, and manifests needed for continuation. Training is
kept stopped while this document is reviewed; the approval and service wrapper must be regenerated
for the final merged `master` revision before restart.

## Method and provenance

- Hardware: four NVIDIA GeForce RTX 4070 SUPER 12 GiB GPUs, 64 logical CPUs, 125 GiB RAM.
- Workload: production checkpoint `model_190.jit.pt`, 600 full and 150 fast searches per ply,
  25% full searches, inference cache disabled.
- Model SHA-256: `3c47dec3cb29e2d7f240675411d3b67d9af0c863cc82d2c79f88e3759e6ec851`.
- Benchmark source revision: `1164304f`.
- Native module SHA-256: `020636f8f22a75513a304e451a2e5ef8610902598ec01a410beaa0c465fd708a`.
- Build: Release, LTO enabled, native timing instrumentation disabled.
- Metric: exact completed native simulation/root-visit deltas divided by measured wall time.
- Baseline runs: 143,398; 142,843; and 142,706 searches/s.
- Direct winner runs: 243,199; 224,640; and 233,141 searches/s.

Raw JSON and resource telemetry remain on the benchmark node under
`/workspace/chess/direct-selfplay-sweep-20260722`. The production configuration records the
selected topology in source control so restart validation covers the same parameters measured
here.
