# Multi-worker graph replay fix and batch sizing — 8× RTX 4070 SUPER, 2026-08-24

| | |
| --- | --- |
| `experiment_configuration_sha256` | `583f0ad49c7a7eebcaddd79ddb14bdf3e17611393a664e50e58d3fd33bddd682` (`inference_workers`, `inference_batch_size` and games per process overridden on the command line, see Method) |
| Source revision | `b333d2ad` (master) plus the two commits of branch `cuda-graph-multi-worker` |
| Node | Vast.ai `f1e810637724` (`ssh6.vast.ai`), 8× RTX 4070 SUPER 12 GiB, driver 595.71.05, Intel Xeon E5-2673 v4, 80 logical CPUs, 188 GiB RAM |
| Date | 2026-08-24 |

## Method

Same harness, config and generation as
[`self-play-submission-8xrtx4070super-20260824`](../self-play-submission-8xrtx4070super-20260824/README.md):
`vast-chess-4day-production-v2.yaml` at generation 60, 8 GPUs × 4 processes, 90 s measured behind a
barrier after 2 warm-up batches, `OMP_NUM_THREADS=1`. Games per process, `inference_workers` and
`inference_batch_size` are overridden per row.

## The fault, and what identified it

With graph capture enabled for more than one inference worker, runs died with
`cudaErrorIllegalAddress`. Minimisation on a single GPU, 2 workers, 20 s:

| processes | games/process | outcome |
| ---: | ---: | --- |
| 1 | 128–512 | stable |
| 2 | 128–512 | stable |
| 3 | 512 | 1 of 3 died |
| 4 | 128 | stable |
| 4 | 256 | stable |
| 4 | 512 | 3 of 4 died |

Peak GPU memory in the failing case was 500 MiB per process, 1.5–2.0 GiB for the whole device out of
12.3 GiB, so memory pressure was not the cause. Capture itself succeeded in every process — no
`graph capture unavailable` was ever logged.

The decisive experiment held the replay lock across a stream synchronise, forcing the two runners'
graphs to execute one at a time. The previously failing configuration then ran clean. **The cause is
concurrent execution of graphs replayed on separate streams inside one process**, not capture.

The fix gives graph-capturing runners one CUDA stream per device, so the device orders their replays
and the host never blocks. The single-worker gate is removed.

## Results

### The fault is fixed

| configuration | outcome |
| --- | --- |
| 4 processes, 1 GPU, 2 workers, 512 games — three runs | 0 of 4 failed, every run |
| **32 processes, 8 GPUs, 2 workers, 768 games, cap 384** | **0 of 32 failed; 580,945 searches/s** |

### Throughput, full production topology

| configuration | searches/s | CPU cores | avg batch | GPU % |
| --- | ---: | ---: | ---: | ---: |
| node baseline (`nodeB-r1`, unoptimised) | 509,511 | 52.4 | 141 | 90.0 |
| 32×1, 512 games, cap 256 | 619,882 | 19.9 | 222 | 92.9 |
| 24×1 (3 per GPU), 512 games, cap 256 | 602,629 | 17.8 | 218.6 | 92.0 |
| 32×1, 768 games, cap 256 | 623,272 | 19.9 | 251.1 | 92.9 |
| 32×2, 512 games, cap 256 | 530,058 | 18.1 | 140.5 | 93.4 |
| 32×1, 1024 games, cap 512 | 656,682 | 22.6 | 466 | 88.7 |
| **32×1, 768 games, cap 384** — 4 runs | **660,292 / 664,487 / 666,928 / 670,415** | 21.8–22.9 | 342–343 | 89.9–91.8 |
| 32×2, 768 games, cap 384 | 580,945 | 20.2 | 217.7 | 91.2 |

Median of the chosen configuration is **665,708 searches/s**, spread 1.5 %, against a node baseline
of 509,511 — **+30.6 % on 42 % of the CPU**.

### Graph bucket granularity

| cap | buckets | avg batch | pads to | searches/s |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 8 | 251 | 256 | 623,272 |
| 256 | 16 | 251 | 256 | 618,040 |
| 384 | 8 | 343 | 384 | 663,509 / 652,936 |
| 384 | 16 | 343 | 360 | 660,292 / 666,928 / 670,415 |

Finer buckets pay only when the average batch sits well below the cap; sixteen is kept because the
chosen cap leaves a 12 % pad at eight.

## Interpretation

- One inference worker remains the right setting: two are now **safe** but 13 % slower
  (580,945 against 665,708), because a second worker halves the batch.
- Three processes per GPU is worse than four; four with more games per process is best.
- The batch cap only became a constraint after submission stopped being one. At 3.5 ms of host
  dispatch per call the pipeline could not fill 256; at 31 µs it fills 343 of 384.
- Host RAM rises from 58.5 to 68 GiB at 768 games per process — comfortable at 188 GiB, but it is
  the setting to reduce first on a smaller node.
- Cross-node caution: a separate evaluation measured 662,137 searches/s for the **unoptimised** code
  on a different 8×4070 SUPER node with a faster CPU (208 % per process, 66.6 of 80 cores). That is
  not comparable to the figures here, which are all this node against itself.

## Reproduce

```bash
python3 tools/benchmark_self_play_search.py --run-config configs/validation/vast-chess-4day-production-v2.yaml --model /workspace/benchmodel/chess-v2.jit.pt --device 0 --worker-id 0 --inference-device cuda --games 768 --generation 60 --warmup-batches 2 --duration-seconds 90 --inference-workers 1 --inference-batch-size 384
```

## Files

- `topology-and-batch.jsonl` — per-worker JSON for every row of the throughput table.
- `fault-minimisation.txt` — the process/games matrix that localised the fault.
