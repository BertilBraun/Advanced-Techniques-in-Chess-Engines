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

## Training configuration

The four-GPU trainer uses one NCCL rank per GPU, devices `(3, 2, 1, 0)`, global batch 2,048, and
local batch 512. The production DDP path previously measured 22,599 samples/s in isolation and
15,026 samples/s while the old half-self-play workload was active. The selected self-play layout
leaves substantially more CPU and host memory available during overlap than that old workload.

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
