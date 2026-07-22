# Integrated direct interactive evaluation

## Implementation

The production `InteractiveEngine` now uses the indexed evaluation arena and direct inference
pipeline. One serial tree owner performs selection, virtual-loss reservation, board encoding,
expansion, backup, re-rooting, and reclamation. Inference workers own independent TorchScript
model replicas and CUDA streams. They consume stable encoded batch slots and never access tree
nodes. The MCTS hot path no longer uses the cached or non-cached general inference clients.

The Python configuration exposes `inference_workers` and uses `maximum_batch_size` as the direct
worker batch capacity. `AlphaZeroBotCpp` defaults to two workers with batches of 64 based on the
integrated measurements below. Policy mode sends exactly one encoded position through a direct
worker and does not create or mutate an MCTS tree frontier.

Fixed-search analysis stops issuing tickets at the exact requested count and drains every issued
batch. Timed analysis uses an inference-latency estimate to stop issuing work before the deadline,
then drains all submitted batches. Every completion, deadline exit, and exception path is required
to leave zero evaluating nodes and zero virtual loss.

## Methodology

- Hardware: NVIDIA GeForce RTX 3060 12 GB; 8 visible logical CPUs.
- Software: Linux 6.8.0-124, Python 3.12.13, PyTorch 2.12.0+cu130.
- Model: deterministic untrained 12 residual blocks by 112 channels TorchScript model,
  SHA-256 `d85f782970cbd1cdc07f0d2c71ce3b9480517720fe0a858f61eaa99341d36e7d`.
- Position: standard starting position, a fresh long-lived game for each configuration.
- Workload: three-second MCTS analysis, plus exact fixed-search runs used as structural quality
  proxies.
- Baseline: revision `10383d4`, synchronous pointer-tree batches through the cached general
  inference client.
- Integrated matrix: 1-4 direct CUDA workers and per-worker batches 32, 50, and 64.
- `voice-light-compute` remained stopped. Vast management services were not changed.

The model has the production network shape but is not trained. The measurements are valid for
throughput, batching, deadlines, and structural correctness. Move agreement, visit distribution,
value delta, and PV-prefix fields in the raw records are regression clues only; they are not
playing-strength evidence.

## Three-second results

The integrated values below show two independent matrix runs. Run-to-run variation is material,
so configuration choices use the median rather than a single maximum.

| Workers | Batch | Run 1 searches/s | Run 2 searches/s | Median searches/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 2,030 | 1,586 | 1,808 |
| 1 | 50 | 3,385 | 3,241 | 3,313 |
| 1 | 64 | 3,637 | 2,549 | 3,093 |
| 2 | 32 | 4,727 | 3,415 | 4,071 |
| 2 | 50 | 8,647 | 8,597 | 8,622 |
| 2 | 64 | 10,267 | 9,257 | **9,762** |
| 3 | 32 | 8,419 | 8,707 | 8,563 |
| 3 | 50 | 7,612 | 9,927 | 8,769 |
| 3 | 64 | 9,881 | 10,092 | **9,987** |
| 4 | 32 | 7,689 | 7,239 | 7,464 |
| 4 | 50 | 6,345 | 7,205 | 6,775 |
| 4 | 64 | 6,712 | 7,554 | 7,133 |

Two and three workers with batch 64 are effectively the leading region. Including the separate
instrumented run, the three-run median is 10,267 searches/s for 2x64 and 9,970 for 3x64. Two
workers are therefore the conservative default; this also uses one fewer model replica and keeps
less virtual-loss work in flight.
Trained-model paired matches must decide whether the slightly different staleness/throughput trade
off changes strength.

The best baseline configuration was parallel width 128 with maximum client batch 50: 11,285
searches in 3.021 seconds, or 3,736 searches/s. The instrumented integrated 2x64 configuration
completed 31,317 searches in 2.992 seconds, or 10,467 searches/s. This is a **2.80x** throughput
increase against the best rerun baseline. All integrated timed configurations finished 6-22 ms
before their requested deadline; the baseline matrix overshot by 2-28 ms.

Actual model batches were full after the shallow-tree warm-up: averages were 49.7-49.8 for batch
50 and 63.5-63.7 for batch 64. Four replicas raised peak observed framebuffer use to approximately
925 MB, so VRAM capacity was not the limiting factor.

## Serial tree-owner profile

The instrumented run separates cumulative tree-owner work. Inference time is summed across workers
and therefore overlaps wall-clock time.

| Workers x batch | Searches/s | Worker active proxy | Select/materialize | Encode | Policy/WDL processing | Expand/backup | Waiting for completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x50 | 8,700 | 62.5% | 55 ms | 15 ms | 1,342 ms | 29 ms | 1,532 ms |
| 2x64 | **10,467** | 58.0% | 64 ms | 18 ms | 1,605 ms | 21 ms | 1,263 ms |
| 3x50 | 9,575 | 56.8% | 48 ms | 18 ms | 1,474 ms | 20 ms | 1,405 ms |
| 3x64 | 9,970 | 50.5% | 50 ms | 18 ms | 1,504 ms | 21 ms | 1,375 ms |
| 4x50 | 7,021 | 53.9% | 34 ms | 14 ms | 1,220 ms | 15 ms | 1,689 ms |
| 4x64 | 8,583 | 48.5% | 45 ms | 15 ms | 1,319 ms | 18 ms | 1,571 ms |

Parallel tree selectors are not the next optimization. Selection, lazy materialization, encoding,
and backup together consume only about 3-4% of the three-second run. The large serial cost is legal
policy filtering, move decoding, and WDL validation after CUDA completion, at roughly 41-54% of
wall time. Direct inference workers are active only about 48-63% of their available aggregate time,
and adding a fourth worker reduces throughput.

The next experiment should move result preparation off the tree owner without granting workers
tree access: publish each batch with stable legal encoded-move metadata, filter/normalize policy
rows on the inference workers, and return compact move scores plus WDL. Expansion and backup remain
serial. Multiple tree selectors are justified only if profiling after that change shows selection
or encoding starving the CUDA workers.

## Raw records

Machine-readable manifests and JSONL records are stored in
`documentation/benchmarks/integrated-interactive-rtx3060-20260722/`:

- `baseline-*`: revision `10383d4` baseline matrix.
- `integrated-run1-*` and `integrated-run2-*`: independent complete worker/batch matrices.
- `instrumented-*`: final timing-counter matrix.
