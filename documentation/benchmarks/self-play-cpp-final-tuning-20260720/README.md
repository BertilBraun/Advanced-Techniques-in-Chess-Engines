# Final C++ self-play topology and timeout tuning

These are the final single-run topology and inference-collection-timeout
measurements after the C++ self-play optimizations. The primary metric is
completed searched plies per second across four GPUs.

## Result

Use four processes per GPU, four MCTS threads and 192 parallel games per
process, a 3,000,000-entry cache per process, NUMA-aware CPU pinning, and a
1,000 microsecond inference collection timeout.

The 1,000 microsecond result is only 0.63% faster than 500 microseconds. That
difference is small enough to be single-run noise, so the timeout choice is
provisional; topology and affinity are the stronger results.

| Run | Processes/GPU | MCTS threads/process | Games/process | Cache/process | Pinning | Timeout (µs) | Plies/s | Delta vs. 32-process reference | Mean batch |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 32-process reference | 8 | 3 | 96 | 1,500,000 | No | 500 | 371.624 | 0.00% | 68.891 |
| 4×4 unpinned | 4 | 4 | 192 | 3,000,000 | No | 500 | 443.351 | +19.30% | 124.956 |
| 4×4 pinned | 4 | 4 | 192 | 3,000,000 | NUMA-aware | 500 | 518.098 | +39.41% | 132.884 |
| 2×8 pinned | 2 | 8 | 384 | 6,000,000 | NUMA-aware | 500 | 441.795 | +18.88% | 159.938 |
| 4×4 pinned | 4 | 4 | 192 | 3,000,000 | NUMA-aware | 1,000 | **521.353** | **+40.29%** | 145.948 |
| 4×4 pinned | 4 | 4 | 192 | 3,000,000 | NUMA-aware | 2,000 | 512.566 | +37.93% | 163.086 |
| 4×4 pinned | 4 | 4 | 192 | 3,000,000 | NUMA-aware | 5,000 | 488.629 | +31.48% | 174.008 |

NUMA pinning improved the otherwise identical 500 microsecond 4×4 run by
16.86%. Reducing the topology further to two processes per GPU and eight MCTS
threads per process was 14.73% slower than pinned 4×4 despite producing larger
batches. Relative to pinned 4×4 at 500 microseconds, 2,000 microseconds was
1.07% slower and 5,000 microseconds was 5.69% slower.

All configurations kept 3,072 parallel games and 48,000,000 aggregate cache
entries. The 4×4 and 2×8 configurations both used 64 MCTS threads; the
32-process reference used 96. Each run used 600 searches per ply, four parallel
searches, a maximum batch size of 256, two warmup steps, and ten measured
steps.

## Reproducibility

The 32-process reference used source revision
`f4b56ed29ba89417ab3403d05bdb66b205cf9527`. The remaining runs used
`7e781cb646d18c66d06555f932cc8458d4345fb0`; the only change between those
revisions is `py/tools/run_self_play_cpp_baseline.sh`. Every manifest reports a
clean worktree and the same compiled module SHA-256,
`be215c6502b612dcba5d46595476ed0798c7e5fcf8ecff34079099b07431ad9b`.

The model was
`/workspace/chess-artifacts/topology-benchmark-model-zero-12x112.jit.pt`, with
SHA-256
`6d8fb642655715057e5dfe8ea33c09db4cd8228ac1c79fa84574989f3579a98e`.
The build was Release, `-O3`, with timing instrumentation off. The node had 64
logical CPUs and four NVIDIA GeForce RTX 4070 SUPER GPUs; PyTorch was
2.12.0+cu130 with CUDA 13.0.

The common remote invocation was:

```bash
BASELINE_OUTPUT_ROOT=/workspace/chess-artifacts \
GPU_COUNT=4 \
SEARCHES_PER_PLY=600 \
PARALLEL_SEARCHES=4 \
MAXIMUM_BATCH_SIZE=256 \
bash py/tools/run_self_play_cpp_baseline.sh \
    /workspace/chess-artifacts/topology-benchmark-model-zero-12x112.jit.pt \
    2 10 /workspace/chess-engine
```

The reference used the runner defaults. The matrix runs additionally set the
following variables:

| Run | `PROCESSES_PER_GPU` | `MCTS_THREADS_PER_PROCESS` | `PARALLEL_GAMES_PER_PROCESS` | `CACHE_CAPACITY` | `PIN_WORKERS_TO_CPUS` | `INFERENCE_TIMEOUT_MICROSECONDS` |
|---|---:|---:|---:|---:|---:|---:|
| 4×4 unpinned, 500 µs | 4 | 4 | 192 | 3000000 | 0 | 500 |
| 4×4 pinned, 500 µs | 4 | 4 | 192 | 3000000 | 1 | 500 |
| 2×8 pinned, 500 µs | 2 | 8 | 384 | 6000000 | 1 | 500 |
| 4×4 pinned, 1,000 µs | 4 | 4 | 192 | 3000000 | 1 | 1000 |
| 4×4 pinned, 2,000 µs | 4 | 4 | 192 | 3000000 | 1 | 2000 |
| 4×4 pinned, 5,000 µs | 4 | 4 | 192 | 3000000 | 1 | 5000 |

The source artifacts were:

| Local directory | Remote directory under `/workspace/chess-artifacts/` |
|---|---|
| `reference-32p-500us/` | `self-play-cpp-baseline-4x8x3x96-timeout500us-20260720T085749Z/` |
| `4x4-unpinned-500us/` | `self-play-cpp-baseline-g4-pg4-t4-games192-s600-ps4-b256-cache3000000-timeout500us-affinitydisabled-20260720T091020Z/` |
| `4x4-pinned-500us/` | `self-play-cpp-baseline-g4-pg4-t4-games192-s600-ps4-b256-cache3000000-timeout500us-affinitytaskset-20260720T091235Z/` |
| `2x8-pinned-500us/` | `self-play-cpp-baseline-g4-pg2-t8-games384-s600-ps4-b256-cache6000000-timeout500us-affinitytaskset-20260720T091424Z/` |
| `4x4-pinned-1000us/` | `self-play-cpp-baseline-g4-pg4-t4-games192-s600-ps4-b256-cache3000000-timeout1000us-affinitytaskset-20260720T091622Z/` |
| `4x4-pinned-2000us/` | `self-play-cpp-baseline-g4-pg4-t4-games192-s600-ps4-b256-cache3000000-timeout2000us-affinitytaskset-20260720T091821Z/` |
| `4x4-pinned-5000us/` | `self-play-cpp-baseline-g4-pg4-t4-games192-s600-ps4-b256-cache3000000-timeout5000us-affinitytaskset-20260720T092019Z/` |

Only `manifest.json`, `resource-summary.json`, and `summary.json` were retained
from each run.

## Validation

All 21 JSON files parse successfully. The manifests were checked for the
common model and module hashes, build flags, workload size, and hardware.
Throughput deltas in the table were recomputed from each `summary.json` using:

```text
100 × (run completed_game_plies_per_second / reference completed_game_plies_per_second − 1)
```
