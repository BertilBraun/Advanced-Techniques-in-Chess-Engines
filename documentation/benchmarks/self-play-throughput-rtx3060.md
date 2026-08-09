# Self-play throughput baseline: four RTX 3060s

This record freezes the isolated self-play/search baseline measured on 2026-08-09. It explains which historical
baseline the reworked runtime was compared with, how the current topology was selected, and which exact settings
should be used when this hardware is benchmarked again. It is a capacity baseline, not an integrated training-speed
or playing-strength result.

## Scope and method

The benchmark drives the production `SelfPlayWorker`, normalized native multi-game search, TorchScript inference,
move selection, tree updates, terminal handling, and completed-game publication. It excludes replay ingestion, DDP
training, evaluation jobs, and checkpoint I/O. The reported rate is completed MCTS searches divided by the slowest
worker's measured makespan; it is not model positions per second or games per second.

All authoritative current runs used source revision
`bdd6059e03e96232d6e3cfeffd4aaf52ddbc13a6`, one warm-up search batch, a requested 60-second measurement window,
and the real production model shapes with zeroed weights. Zero weights preserve the production compute graph while
making no playing-strength claim. The earlier parity comparison used a complete 30-second window because the
pre-rework chess runtime fails on a natural terminal in a longer run.

### Hardware and runtime

| Item | Frozen value |
| --- | --- |
| Vast instance / host | `47261298` / `95290` |
| GPUs | 4 x NVIDIA GeForce RTX 3060, 12,288 MiB each, 120 W limit, compute capability 8.6 |
| Driver | NVIDIA 580.65.06 |
| CPU | Host: 2 x AMD EPYC 7B12, 256 logical CPUs; rental allocation: 85.33 effective CPUs |
| RAM / disk | Host-visible RAM: 251 GiB; rental allocation: 86 GiB; disk: 150 GiB ephemeral storage |
| Runtime | Python 3.12.3, PyTorch 2.12.1+cu126, CUDA wheel 12.6, cuDNN 9.10.2 |
| Price at measurement | $0.303/hour including disk |

The benchmark model hashes are:

- chess 12-block, 112-channel model: `a31b5a5a31f3577e3e68d3e09a115de2e45c2d1473b9a4c651bd75fb9724a5bf`;
- Go 7x7 6-block, 64-channel model: `9bcef68c7cd7706967ac980f334c0607a52e974fb56fc27f4aa1fca2893c2757`.

Resource telemetry reports host-wide CPU and RAM denominators because the shared host inventory is visible inside
the container. Those values must not be interpreted as owned capacity. The benchmark worker process CPU and RSS
totals are the useful run-local measurements; every tested topology remained below the 85.33-CPU/86-GiB rental
allocation.

## Chess: historical baseline to current optimum

The useful pre-rework baseline is commit `bb334eb8c5c48eaa686c4d9a9a43c20b3cac10e4`: four processes per GPU,
192 games per process, inference batch 64, and 3,072 active games. On this same node it sustained 60,458.72
searches/s. The reworked runtime sustained 59,715.93 searches/s with the same topology, batch distribution, and
search schedule, a difference of -1.23%. This establishes practical runtime parity and is the correct starting point;
the much slower eight-process diagnostic was suboptimal in both implementations.

The current topology sweep then improved the reworked runtime from that matched point:

| Stage | Processes/GPU | Games/process | Batch limit | Searches/s | Mean batch | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pre-rework matched baseline | 4 | 192 | 64 | 60,458.72 | 26.87 | Historical reference |
| Current matched topology | 4 | 192 | 64 | 59,715.93 | 26.89 | Practical parity, -1.23% |
| Current process optimum before batching work | 2 | 384 | 64 | 69,439.93 | 36.74 | Fewer fragmented streams |
| Current frozen optimum | **2** | **1,024** | **64** | **76,719.77** | **63.53** | 92.45% full batches |
| Larger inference batch | 2 | 2,048 | 128 | 71,453.58 | 126.97 | Slower and coarser lifecycle |

The frozen result is 26.90% faster than the useful pre-rework baseline, 28.47% faster than the matched current
topology, and 10.48% faster than the initial two-process current result.

### Why it improved

Four facts explain the result:

1. Extra processes fragmented independent positions across separate inference queues. Moving from four to two
   processes per GPU retained enough independent work while producing denser inference batches.
2. Each process has two inference workers with two outstanding batches each, hence four independent inference slots.
3. At generation 20, only 25% of games use the full 600-search budget; the other 75% finish after 150 searches. The
   long-search tail therefore has only `games * 0.25 / 4` positions available to each slot.
4. About `64 * 4 / 0.25 = 1,024` independent games are needed per process to fill a batch of 64 during that tail.

The measurements confirm the model: 384 games averaged 36.74 positions, whereas 1,024 games averaged 63.53 and
made 92.45% of calls at the full limit. Increasing parallel games preserves sequential MCTS semantics. Parallel
leaves were intentionally kept at one because they can reduce search quality.

A larger model batch did not help. Batch 128 reduced throughput by 6.86% from the optimum and increased approximate
worker-batch latency from 27 to 57 seconds. Four processes with 1,024 games matched raw throughput but doubled CPU,
RAM, and worker-batch latency. The bottleneck is small-network inference density rather than RAM or host CPU: the
optimum used 8.63% host CPU, 16.37 GiB summed worker peak RSS, and roughly 54 W mean GPU power despite high reported
GPU utilization.

### Frozen chess baseline

Use this topology for an isolated repeat on four RTX 3060s:

```yaml
experiment:
  topology:
    self_play:
      device_ids:
        - 0
        - 0
        - 1
        - 1
        - 2
        - 2
        - 3
        - 3
      parallel_games_per_process: 1024
chess:
  self_play:
    search:
      parallel_searches: 1
    inference:
      inference_workers: 2
      inference_batch_size: 64
      outstanding_batches_per_worker: 2
```

Expected isolated capacity is approximately 76,720 searches/s. This is not yet the final integrated experiment
topology: replay, four-rank DDP training with local batch 512, and concurrent evaluation may require reserving GPU,
CPU, or latency headroom.

## Go 7x7: maximum-throughput sweep

The previously quoted 95,754.34 searches/s was an older smoke result without retained evidence for the exact model
artifact. The authoritative starting point was therefore reproduced with the current Go model and configuration:
one process per GPU, 64 games per process, batch 64, and two parallel leaves. It sustained 88,513.83 searches/s.

The first sweep removed parallel leaves and increased independent games. With batch 64, the supply requirement is
only `batch * inference slots = 64 * 2 = 128` games because every Go game uses the full 128-search budget.

| Processes/GPU | Games/process | Batch | Searches/s | Mean batch | Mean GPU utilization | Host CPU | Summed peak RSS |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1, two parallel leaves | 64 | 64 | 88,513.83 | 61.20 | 25.89-28.79% | 6.27% | 4.87 GiB |
| 1 | 128 | 64 | 81,901.02 | 61.81 | 24.82-26.68% | 12.48% | 4.87 GiB |
| 2 | 128 | 64 | 175,783.04 | 61.84 | 72.82-80.11% | 13.89% | 9.72 GiB |
| 4 | 128 | 64 | 234,008.03 | 62.71 | 91.59-95.59% | 11.18% | 19.43 GiB |
| 8 | 128 | 64 | 255,359.33 | 63.01 | 94.00-96.17% | 16.69% | 38.34 GiB |
| 12 | 128 | 64 | 255,271.18 | 63.11 | 94.50-97.87% | 27.69% | 56.90 GiB |

Eight processes reached the small-batch process plateau; twelve added CPU and RAM without throughput. Four
processes were retained while batch size and independent games were scaled proportionally:

| Processes/GPU | Games/process | Batch | Searches/s | Mean batch | Mean GPU utilization | Host CPU | Summed peak RSS |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 4 | 128 | 64 | 234,008.03 | 62.71 | 91.59-95.59% | 11.18% | 19.43 GiB |
| 4 | 256 | 128 | 337,353.34 | 125.66 | 91.95-94.55% | 10.80% | 20.79 GiB |
| 4 | 512 | 256 | 391,660.65 | 251.95 | 83.91-90.75% | 16.60% | 23.27 GiB |
| **4** | **1,024** | **512** | **446,725.00** | **504.04** | **82.16-87.48%** | **14.57%** | **27.02 GiB** |
| 4 | 2,048 | 1,024 | 423,854.66 | 1,006.54 | 68.30-76.73% | 10.34% | 31.88 GiB |

Batch 512 is the measured optimum. It is 404.70% faster than the reproduced starting point and 445.45% faster
than the quality-preserving one-process sequential-search point. Batch 1,024 is 5.12% slower, stretches the
makespan from 63.96 to 69.27 seconds, and lowers GPU utilization. The larger batch no longer compensates for its
latency.

The reason Go benefits much more than chess is structural. The 6x64 7x7 model is exceptionally small, every game
receives the full search budget, and its inference calls are cheap enough that per-call overhead dominates at batch
64. Four concurrent processes provide enough independent CUDA streams; increasing each stream to a dense batch of
512 amortizes dispatch and framework overhead. At the optimum, mean per-GPU power was 57.16 W and peak power was
54.79-72.06 W, still far below the 120 W training limit. Reported GPU utilization is therefore not evidence that
the arithmetic units are saturated.

### Frozen Go 7x7 baseline

Use this topology for an isolated repeat on four RTX 3060s:

```yaml
experiment:
  topology:
    self_play:
      device_ids:
        - 0
        - 0
        - 0
        - 0
        - 1
        - 1
        - 1
        - 1
        - 2
        - 2
        - 2
        - 2
        - 3
        - 3
        - 3
        - 3
      parallel_games_per_process: 1024
go:
  self_play:
    search:
      parallel_searches: 1
    inference:
      inference_workers: 1
      inference_batch_size: 512
      outstanding_batches_per_worker: 2
```

Expected isolated capacity is approximately 446,725 searches/s. This setting intentionally optimizes total
throughput, not individual game latency. Before using it in an integrated experiment, verify shutdown/model-refresh
latency and contention with DDP training and evaluation.

## Reproduction and evidence

The benchmark harness is `py/tools/run_self_play_search_benchmark.sh`. A representative Go invocation is:

```bash
BENCHMARK_OUTPUT_ROOT=/workspace/r11-benchmarks/go7-sequential-4x1024-batch512-generation20-60s \
GPU_COUNT=4 \
PROCESSES_PER_GPU=4 \
PARALLEL_GAMES_PER_PROCESS=1024 \
PARALLEL_SEARCHES=1 \
INFERENCE_WORKERS=1 \
INFERENCE_BATCH_SIZE=512 \
OUTSTANDING_BATCHES_PER_WORKER=2 \
MEASUREMENT_DURATION_SECONDS=60 \
WARMUP_BATCHES=1 \
BENCHMARK_GENERATION=20 \
PYTHON_BINARY=/workspace/alphazero-engine-venv/bin/python \
bash py/tools/run_self_play_search_benchmark.sh \
  /workspace/r11-benchmark-models/go7.jit.pt \
  py/configs/go-7x7-experiment-template.yaml
```

Raw Go manifests, worker reports, batch histograms, and CPU/GPU telemetry are retained outside Git in
`.codex-diagnostics/r11-go7-throughput-evidence.tar.gz` (160,260 bytes, SHA-256
`f49e3eda997a2a041d747e64596ec21e03e55f18a9e1da54bdecf48c89b43259`). Chess evidence is retained in
`.codex-diagnostics/r11-batching-sweep-evidence.tar.gz` (44,997 bytes, SHA-256
`f0a18498341169a30794cad1acbcf21f3b769cb459035fdca98e9d8d6f7fed65`) and the earlier R11 evidence archives.

Treat these numbers as hardware-, model-, runtime-, and revision-specific. A different GPU, network shape, search
schedule, or concurrent workload requires a new baseline rather than extrapolation from this table.
