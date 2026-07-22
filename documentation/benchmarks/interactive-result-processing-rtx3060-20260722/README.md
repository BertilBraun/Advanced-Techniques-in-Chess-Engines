# Interactive CUDA result-processing benchmark (RTX 3060, 2026-07-22)

This sweep measures the optimized direct result conversion and pipelined inference implementation
in a complete, serially owned MCTS. It uses the same deterministic 12-layer, 112-channel model as
the preceding RTX 3060 baseline. The model is untrained, so these results establish throughput,
deadline behavior, and deterministic structure, not playing strength.

## Provenance

`manifest.json` records the exact command, source revision, model SHA-256, Python, Torch, operating
system, CPU count, and CUDA device. The relevant environment was:

- NVIDIA GeForce RTX 3060 12 GB, compute capability 8.6, driver 595.71.05
- CUDA toolkit 13.0.88 and Torch 2.12.0+cu130
- eight logical CPU cores and 16 GB system memory
- Python 3.12.13, GCC 13.3, Release build
- model SHA-256 `d85f782970cbd1cdc07f0d2c71ce3b9480517720fe0a858f61eaa99341d36e7d`
- source revision `98fb0e5115d924c0ddaed2a4127dbc5cd097aaa2`

The build and validation commands were:

```text
cmake -S cpp -B cpp/build-rtx -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CLANG_TIDY=OFF
cmake --build cpp/build-rtx -j2
ctest --test-dir cpp/build-rtx --output-on-failure
cd py
PYTHONPATH=. python -m pytest --import-mode=importlib test/test_interactive_engine.py -q
```

All ten native tests and all seven focused Python tests passed. The matrix completed 128 of 128
runs, with no skips or errors. `results.jsonl` contains every timed and fixed-search record.

## Full-MCTS result

| Replicas | Batch | Outstanding/replica | Searches/s | Searches in 3 s | Result processing | Inference wait | Worker utilization |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 2 | **18,115** | 54,255 | 72 ms | 2,666 ms | 87.0% |
| 2 | 128 | 2 | 17,773 | 52,645 | 52 ms | 2,717 ms | 87.3% |
| 3 | 50 | 2 | 17,473 | 52,071 | 89 ms | 2,597 ms | 83.8% |
| 3 | 64 | 2 | 16,405 | 48,789 | 66 ms | 2,670 ms | 83.4% |
| 2 | 128 | 1 | 16,125 | 48,149 | 61 ms | 2,689 ms | 80.4% |
| 2 | 64 | 1 | 15,526 | 46,485 | 69 ms | 2,646 ms | 79.2% |

The preceding comparable integrated baseline was 10,467 searches/s at two replicas and batch 64.
The new best is 73% faster. At the same two-replica, batch-64 configuration, pipelining a second
batch raises throughput another 16.7% over the one-outstanding implementation.

The earlier profile spent about 51 microseconds per evaluated position in policy/WDL conversion.
The winning run spends 1.33 microseconds per position, roughly 38 times less. Selection and board
materialization take 148 ms, encoding 43 ms, result processing 72 ms, and expansion/backup 26 ms.
The remaining dominant cost is the 2,666 ms inference wait, not policy processing.

## Direct-inference reconciliation

The separate model-replica sweep included forward, complete policy/WDL copy, and optionally the new
legal-move filtering and WDL conversion:

| Replicas | Batch | Raw positions/s | Processed positions/s |
|---:|---:|---:|---:|
| 1 | 64 | 18,909 | 18,193 |
| 2 | 64 | 28,084 | 28,524 |
| 2 | 128 | 28,670 | 28,381 |
| 3 | 50 | 33,047 | 32,882 |
| 3 | 64 | 30,498 | 30,392 |
| 4 | 64 | 31,111 | 30,527 |
| 8 | 64 | 17,241 | 17,339 |

The old and new runtimes agree within about 1% for the previously measured raw 2x64 and 3x64
cases. The optimized full MCTS reaches 64.5% of raw 2x64 inference throughput, versus 37.6% for the
old integrated baseline. Processed throughput is effectively the same as raw throughput, confirming
that policy/WDL conversion is no longer the missing threefold factor.

## Configuration and correctness conclusions

Two outstanding batches improve every tested one-to-three-replica configuration, including 10-23%
at the strongest two-replica settings. Four or more replicas with two outstanding batches perform
poorly on this eight-logical-core host, so that combination is not recommended. Production defaults
are therefore two replicas, batch 64, and two outstanding batches; the knobs remain explicit for
different hardware.

All 64 fixed-work records stopped at exactly 4,096 searches. Two independent fixed-4,096 runs of
the production configuration produced identical chosen move (`d2d3`), root value, complete ordered
candidate list, depth, and principal variation (`d2d3 b8c6 b1a3 a8b8`). Across the full matrix, the
largest positive three-second deadline error was 116 ms in a severely oversubscribed eight-replica
case; the winning configuration stopped 5 ms early. Peak CUDA allocator memory ranged from 424 to
704 MiB for two replicas and remained below 2.4 GiB even at eight replicas.

The `smoke-model-*` files preserve a separate accidental small-model ceiling sweep and must not be
compared with the 12x112 baseline. `determinism-a.jsonl` and `determinism-b.jsonl` contain the two
independent production-configuration checks.
