# C++ self-play baseline: 4×8×3×96

This is the pre-optimization C++ self-play baseline measured from
2026-07-20 07:16:43 UTC to 07:18:36 UTC. The synchronized measurement
completed 30,720 game plies and 17,946,496 actual MCTS searches in a
109.359-second makespan.

## Result

| Metric | Result |
| --- | ---: |
| Completed game plies per second | 280.910 |
| Actual MCTS searches per second | 164,106.190 |
| Mean inference batch size | 62.378 |
| Batch-size-one calls | 41,454 / 133,370 (31.082%) |
| Cache hit rate | 9,615,040 / 17,934,368 (53.612%) |

The model processed 8,319,328 positions in 133,370 calls. Observed batch
sizes ranged from 1 to 200 against a configured maximum of 256. The caches
held 10,210,304 entries (6,016 MiB summed across workers) at the end, with
zero recorded fingerprint collisions.

## Conditions

- Source: clean commit `7db7a88cdb6c5402a2d87f0f872c58ce8dc2d71e`.
- Native module SHA-256:
  `208537c6e3ce05edd10b1babc9e6877ca4825181d7b5788b692ee3edb8a8a9c7`.
- Build: `Release`, `-O3`, timing instrumentation `OFF`.
- Runtime: PyTorch `2.12.0+cu130`, CUDA `13.0`.
- Hardware: four NVIDIA GeForce RTX 4070 SUPER GPUs, each reporting
  12,282 MiB, driver `595.71.05`; 64 logical CPUs.
- Topology: eight processes per GPU, 32 processes total, three MCTS threads
  per process (96 total), and 96 parallel games per process (3,072 total).
- Workload: two warm-up steps followed by ten measured steps; 600 target
  searches per ply, four parallel searches, maximum batch size 256,
  500-microsecond inference timeout, and 1,500,000 cache entries per process.

The model was
`/workspace/chess-artifacts/topology-benchmark-model-zero-12x112.jit.pt`,
SHA-256
`6d8fb642655715057e5dfe8ea33c09db4cd8228ac1c79fa84574989f3579a98e`.
It is the zero-parameter benchmark model rather than a trained checkpoint,
so this result is a systems-throughput baseline for that fixed model.

## Resources

The 32 workers used an aggregate 5,795.612% process CPU, estimated as
90.556% of the 64-logical-CPU host capacity. Peak host RAM usage was
83.619%. Summed per-worker peak resident set size was 110,165.355 MiB;
because these are per-process peaks, the sum is not a simultaneous-memory
measurement.

| GPU | Mean utilization | Peak memory used |
| ---: | ---: | ---: |
| 0 | 60.365% | 6,783 MiB |
| 1 | 64.000% | 2,288 MiB |
| 2 | 60.346% | 2,288 MiB |
| 3 | 59.750% | 2,288 MiB |

Each GPU contributed 52 resource samples. Mean utilization across the four
GPUs was 61.115%.

## Reproduction

From a clean checkout of the source commit with the recorded Release build:

```bash
bash py/tools/run_self_play_cpp_baseline.sh \
  /workspace/chess-artifacts/topology-benchmark-model-zero-12x112.jit.pt \
  2 \
  10
```

The machine must expose four GPUs and admit the fixed 96-MCTS-thread
topology under the runner's 1.5× CPU oversubscription limit.

Machine-readable evidence is in [manifest.json](manifest.json),
[resource-summary.json](resource-summary.json), and
[summary.json](summary.json).
