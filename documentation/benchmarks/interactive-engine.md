# Interactive engine benchmark

`py/tools/benchmark_interactive_engine.py` measures the production workload: one long-lived
engine analyzing one history-aware game position. It does not use the parallel self-play path.

The matrix compares sequential and tree-parallel search through the thread count, inference
batch limits and collection timeout, explicit CPU/CUDA placement, fixed wall-clock budgets,
and deterministic fixed-search budgets. CUDA configurations that cannot run are emitted as
`skipped` records instead of aborting the matrix.

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
  --threads 1,2,4,8 `
  --batch-sizes 1,8,32 `
  --batch-timeouts-microseconds 50,500 `
  --time-budgets-seconds 1,3,10 `
  --search-limits 512,2048
```

The output directory contains `manifest.json`, append-safe-per-run `results.jsonl`, and
`summary.json`. The manifest captures the command, git state, model hash, Python/Torch versions,
OS/CPU information, and CUDA device names. Keep these files with published results, and also
record the native compiler and CMake flags from the build log.

A checked-in smoke run and its limitations are documented in
`interactive-engine-local-20260721/README.md`.
