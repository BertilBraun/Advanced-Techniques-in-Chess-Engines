# Self-play MCTS node-arena and inference-cache benchmarks

All runs used the production stochastic C++ self-play path on four RTX 4070 SUPER GPUs. The
topology was 8 processes per GPU, 3 MCTS threads per process, and 96 games per process (3,072
concurrent games). Workload settings were 600 full searches, 4 parallel searches, a maximum
inference batch of 256, a 500 microsecond batching timeout, one warm-up step, and a 20 second
measurement request. Builds were Release/O3 with timing instrumentation disabled.

## Node arena versus baseline

The baseline is revision `439997d1a2e79fd5e59f2745e161655e1b37fce1`. The arena run is
revision `d2880b78ba470ee770e7b4307cbc44e7f773f6dc`.

| Metric | Baseline | Node arena | Change |
| --- | ---: | ---: | ---: |
| Game updates/s | 643.142 | 695.110 | +8.08% |
| Summed worker peak RSS | 68,624.3 MiB | 45,105.9 MiB | -34.27% |
| Peak host RAM | 48.629% | 30.860% | -36.54% |
| Mean GPU utilization | 83.771% | 82.938% | -0.83 percentage points |

At the end of the arena run, the 3,072 retained trees contained 36,311 materialized nodes and
1,008,507 child records. Arena capacity was exactly 605 nodes per game.

## Cached versus non-cached inference

Both runs used revision `f91345aa59ffa3a536fb0db607c6899dc6cb8fc4`, seed base 500, and
identical topology and workload settings. The cached client used a capacity of 1,500,000
positions per process. The non-cached client allocates no cache and performs no cache lookup,
insert, eviction, collision, or synchronization operation.

Time per ply is the topology-wide effective wall time:
`1,000 * maximum worker elapsed seconds / total attempted game updates`.

| Metric | Cached | Non-cached | Change |
| --- | ---: | ---: | ---: |
| Effective time/ply | 1.4322 ms | 1.4198 ms | -0.87% |
| Game updates/s | 698.212 | 704.336 | +0.88% |
| Summed worker peak RSS | 45,379.0 MiB | 42,148.4 MiB | -7.12% |
| Mean GPU utilization | 88.396% | 85.688% | -2.71 percentage points |
| Cache hit rate | 0.970% | 0% | — |

These are short stochastic runs, so the small timing difference should be treated as indicative,
not as a stable speedup. The memory difference and zero cache telemetry directly establish that
the non-cached implementation does not retain inference results.

The `*-manifest.json` files record the source, build, hardware, topology, workload, model hash,
and clean-worktree status. The `*-summary.json` files contain complete measurement, inference,
resource, and search-tree statistics. Worker logs and long-run training logs were not copied into
this artifact directory.
