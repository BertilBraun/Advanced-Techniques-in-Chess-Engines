# Interactive engine benchmark

`py/tools/benchmark_interactive_engine.py` measures the production workload: one long-lived
engine analyzing one history-aware game position. It does not use the parallel self-play path.

The matrix compares sequential and tree-parallel search through the parallel-leaf count,
inference batch limits and collection timeout, explicit CPU/CUDA placement, fixed wall-clock
budgets, and deterministic fixed-search budgets. CUDA configurations that cannot run are
emitted as `skipped` records instead of aborting the matrix. Parallel leaves are selected
serially with virtual loss, evaluated as one explicit network batch, and backed up serially.
This keeps tree mutation deterministic and makes `--parallel-searches` a model batch-width
control rather than an operating-system thread count.

Each run records completed searches, searches per second, deadline overshoot, depth, root visits,
ordered candidate statistics, root value, and PV. Quality is compared with a high-search
single-thread reference using move agreement, reference-move rank and visit share, value delta,
and common PV prefix. This is an efficiency proxy, not an Elo estimate: any promising parallel
configuration still needs paired games against the reference at equal time or equal searches.

Example from `py`:

```powershell
python .\tools\benchmark_interactive_engine.py `
  --model .\training_data\chess\model_0.jit.pt `
  --output-directory .\benchmark-results\interactive `
  --devices cpu,cuda `
  --parallel-searches 1,4,8,16 `
  --batch-sizes 1,8,32 `
  --batch-timeouts-microseconds 50,500 `
  --time-budgets-seconds 1,3,10 `
  --search-limits 512,2048
```

The output directory contains `manifest.json`, append-safe-per-run `results.jsonl`, and
`summary.json`. The manifest captures the command, git state, model hash, Python/Torch versions,
OS/CPU information, and CUDA device names. Keep these files with published results, and also
record the native compiler and CMake flags from the build log.

For an exported source tree without `.git`, pass `--source-revision <revision>`. The harness
otherwise records `unavailable` instead of failing before the benchmark begins.

## RTX 3060 optimization run, 2026-07-22

The remote run used an NVIDIA GeForce RTX 3060 12 GB, CUDA 13.0, driver 595.71.05,
Torch 2.12.0+cu130 with cuDNN 9.2, Python 3.12.13, GCC 13.3, and the Release build
(`-O3`, LTO). The benchmark model was the production chess shape (12 residual layers,
112 hidden channels, 3,241,835 parameters) initialized with seed 20260722; its SHA-256 was
`d85f782970cbd1cdc07f0d2c71ce3b9480517720fe0a858f61eaa99341d36e7d`. It was not a trained
checkpoint, so these runs measure throughput and search mechanics, not playing strength.

At a one-second wall-clock budget, source `c8b322d` produced:

| Target | Parallel searches | Searches/s | Average model batch | Deadline overshoot |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 55.4 | 1.00 | 10 ms |
| CPU | 4 | 111.9 | 3.90 | 10 ms |
| CPU | 8 | 120.4 | 7.59 | 71 ms |
| CPU | 16 | 123.7 | 14.33 | 43 ms |
| CUDA | 1 | 193.8 | 1.00 | 1 ms |
| CUDA | 4 | 260.7 | 3.95 | 1 ms |
| CUDA | 8 | 353.0 | 7.84 | 0 ms |
| CUDA | 16 | 753.0 | 15.69 | 0 ms |

The previous asynchronous worker implementation averaged approximately one position per model
call and reached about 197 CUDA searches/s at width one; it failed with a broken promise at
eight workers. Explicit leaf batching therefore delivered about 3.8x the best stable timed CUDA
throughput on this hardware while removing concurrent tree mutation. Fixed-search results were
also recorded because a larger parallel width uses staler selection information and may be
weaker per search. A trained checkpoint and paired games remain required before choosing the
production width solely from the throughput result.

A second run used a 4,096-search sequential reference, 1,024-search fixed workloads, and
three-second timed workloads. All widths (1, 4, 8, and 16) chose the reference move at 1,024
fixed searches. In three seconds, widths 1, 8, and 16 agreed with the reference; width 4 did not
(the reference move ranked fifth). Width 16 completed 2,801 searches at 932.4 searches/s versus
775 at 258.3 searches/s for width 1. This is encouraging but insufficient as a strength claim:
the synthetic model and single position cannot quantify the loss from stale parallel selection.

An exploratory three-second sweep continued to 32, 64, and 128 parallel searches. CUDA
throughput reached 1,647, 1,778, and 1,810 searches/s respectively, showing diminishing returns
after width 64. Width 128 overshot the deadline by 53 ms. These widths are intentionally not the
Python bot default: their larger stale-selection window needs trained-checkpoint match testing
before the additional raw searches can be treated as a strength improvement.

A checked-in smoke run and its limitations are documented in
`interactive-engine-local-20260721/README.md`.
