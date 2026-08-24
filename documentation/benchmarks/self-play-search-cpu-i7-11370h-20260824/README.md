# Self-play search CPU profile and optimisation — i7-11370H (CPU only), 2026-08-24

| | |
| --- | --- |
| `experiment_configuration_sha256` | not applicable — the harness takes explicit search parameters, not a resolved run config (values below mirror `py/configs/validation/vast-chess-4day-production-v2.yaml` at generation 0) |
| Source revision | baseline `b0719ac74789b097ed43e1757afda4dece385542` (`clean`); optimised = baseline + this branch's native commit |
| Node | local development box, **no GPU used**: WSL2 Ubuntu 24.04 on Windows 11, Intel i7-11370H (4 cores / 8 threads, 4 CPUs visible to WSL), 5.8 GiB RAM, gcc 13.3.0, torch 2.7.1+cpu |
| Date | 2026-08-24 |

## Method

The subject is the CPU cost of the native self-play search, isolated from the GPU. A new tool,
`py/tools/benchmark_native_self_play_loop.py`, drives `AlphaZeroCpp.ChessSelfPlaySearch` through the
same discount → request → search → play loop as `SelfPlayWorker.run_batch`, against a **stand-in
TorchScript network**: a fixed per-action logit vector plus a position-dependent WDL, on the CPU
device. The stand-in costs roughly two orders of magnitude less than the trained network on a GPU, so
the search thread is never inference-bound and its own cost is measured directly.

Search parameters: 512 parallel games, adaptive full-search budget 200–250 visits (observation
interval 100, leader window 200), fast searches 50, `parallel_searches` 2, exploration 1.5,
reduced-parent FPU 0.2, forced playouts 1.5, virtual loss 0.5, value discount 0.99, retained root
fraction 0.6, 2 inference workers, inference batch cap 256, 2 outstanding batches per worker. Two
search mixes: `full_search_probability` 0.25 (the production mix from generation ~45 onward) and 1.0
(the generation-0 mix).

Both binaries are **`RelWithDebInfo` (`-O2 -g -DNDEBUG`), `ENABLE_NATIVE_ARCHITECTURE=OFF`** — not a
Release build. This is an A/B on one box; the absolute rates below are **not** comparable to the
production numbers in `search-throughput-rtx4070-20260821/` and must not be quoted as throughput
evidence.

Per-phase cost comes from the native `InferenceStatistics` counters, normalised by simulations
completed, which is far more stable on a contended 4-core box than wall-clock throughput. Sample
window: 2 warm-up batches then 12 measured batches, 5 repetitions alternating baseline and optimised.
`perf record -F 499 -g --call-graph fp` over the whole process supplies the symbol shares.

## Results

### A/B, production mix (`full_search_probability` 0.25), 5 repetitions

Nanoseconds of search-thread CPU per simulation. "best" is the least-contended repetition of each
arm, which is the fairer comparison on a 4-CPU box; medians are shown alongside.

| metric | baseline | optimised | delta |
| --- | ---: | ---: | ---: |
| native search ns/simulation, median | 11,307 | 8,119 | −28.2% |
| native search ns/simulation, best | 9,916 | 7,807 | −21.3% |
| tree selection ns/sim (best run) | 3,323 | 3,062 | −7.9% |
| result processing ns/sim | 2,648 | 1,817 | −31.4% |
| tree backup ns/sim | 1,669 | 1,434 | −14.1% |
| board encoding ns/sim | 881 | 274 | −68.9% |
| inference wait ns/sim | 239 | 199 | −16.7% |
| unaccounted ns/sim | 1,155 | 1,022 | −11.5% |
| mean final root visits | 99.43 | 99.41 | unchanged |
| average inference batch | 83.9 | 83.8 | unchanged |

Simulations completed per seed agree between arms to within 0.3% and mean final root visits to three
significant figures, so the search itself is unchanged; only its cost moved.

### A/B, generation-0 mix (`full_search_probability` 1.0), 3 repetitions

| metric | baseline | optimised | delta |
| --- | ---: | ---: | ---: |
| native search ns/simulation, median | 9,057 | 7,722 | −14.7% |
| average inference batch | 238.6 | 238.6 | unchanged |

### Where the search-thread CPU goes (baseline, production mix)

| phase | share of native search time |
| --- | ---: |
| tree selection (PUCT descent, `materializeChild`) | 32–36% |
| result processing (`processInferencePosition`) | 27–28% |
| tree backup | 15–18% |
| board encoding | 10% |
| waiting on inference | 1–2% |
| unaccounted (task setup, arena prune, result assembly) | 8–14% |

### Inference batch fill is bounded by schedulable games, not by the cap

Average positions per model call, 512 games, baseline binary:

| `full_search_probability` | `parallel_searches` | `inference_batch_size` | average batch |
| ---: | ---: | ---: | ---: |
| 1.0 | 1 | 256 | 237.8 |
| 1.0 | 2 | 256 | 238.7 |
| 1.0 | 4 | 256 | 252.4 |
| 0.25 | 2 | 64 | 60.8 |
| 0.25 | 2 | 128 | 76.1 |
| 0.25 | 2 | 256 | 83.5 |

### Symbol shares, whole process (`perf` self time)

Shares are of a total that shrank by roughly a fifth, so a rising share is not a rising absolute cost.

| object | baseline | optimised |
| --- | ---: | ---: |
| `AlphaZeroCpp.so` | 45.7% | 45.3% |
| `libtorch_cpu.so` (stand-in network, inference threads) | 25.4% | 25.7% |
| `python3.10` | 14.9% | 14.0% |
| `libc.so.6` (mostly `malloc`/`free`) | 8.0% | 10.0% |

| symbol | baseline | optimised |
| --- | ---: | ---: |
| `ChessEncoding::actionId` | 4.92% | below the 0.7% cut |
| `ChessEncoding::encodeInputInto` | 4.01% | 1.08% |
| `selectAvailableLeaf` (both overloads) | 7.06% | 9.22% |
| `completeWorker` (expansion and backup, inlined) | 5.38% | 8.35% |
| `processInferencePosition` | 3.98% | 4.30% |
| `malloc`/`free` family | 6.62% | 8.58% |

## Interpretation

- The self-play search thread costs about 8–10 µs of CPU per simulation on this core at the
  production search mix, after the change. The four-day-run yardstick of ≈100k searches/s per GPU
  implies a per-simulation budget of 10 µs of one core, so on production hardware the search thread
  is at or very near saturation and the CPU is a first-order constraint, not a secondary one. This
  supports the reported symptom (CPU and GPU both pinned) but does **not** measure it: no GPU was
  involved here, and per-core performance differs between this i7 and the Ryzen 7700X nodes.
- What the change buys is a ≈21–28% reduction in search-thread CPU per simulation with an unchanged
  search. That is headroom, not throughput: whether it converts into searches/s on a node depends on
  whether the search thread or the GPU is the binding constraint there.
- The batch-fill table is the more consequential finding. At the production mix the average inference
  batch is 84 positions regardless of whether the cap is 128 or 256, because fast searches retire
  early and the pool of concurrently schedulable games collapses to the full-search games. Raising
  `inference_batch_size` from 64 to 256 in the v2 config therefore stops paying at production
  generations; the lever that does move fill is more concurrent games or a higher
  `parallel_searches`. This is consistent with the 18–44 fill against a cap of 64 measured in
  `search-throughput-rtx4070-20260821/`.
- Removing the batch truncation on a blocked tree (see the native commit) did **not** change fill
  (83.4 → 83.7): the limit is the number of schedulable leaves, not the truncation.
- What remains, in order: tree selection and backup (≈17% of process cycles, dominated by pointer
  chasing over 1.25 kB arena slots), `malloc`/`free` (≈9%), position copies in
  `TreeArena::allocateNode` (≈3.8%), and the Python advance loop (≈14%).
- Nothing here says anything about the GPU-side CPU cost — kernel-launch overhead, the host-side
  dtype conversion in `InferenceRunner::forwardInto`, or `cudaEventSynchronize` spin in
  `waitCompletedOutput`. Those need the node.

## Reproduce

Build the extension, then, from `py` with `OMP_NUM_THREADS=1` and `PYTHONPATH` pointing at the
built module:

```bash
python3 tools/benchmark_native_self_play_loop.py --games 512 --warmup-batches 2 --measured-batches 12 --full-search-probability 0.25 --seed 3001
```

`--full-search-probability 1.0` gives the generation-0 mix; `--inference-batch-size` and
`--parallel-searches` reproduce the fill table; `--collect-statistics` adds the search-quality
aggregates.

## Files

- `mixed-mix-ab.jsonl` — the 5×2 production-mix A/B, one JSON object per run.
- `full-and-mixed-ab.jsonl` — the earlier 3×2×2 run covering both mixes; the production-mix rows in
  it are too short (4 batches) to be stable and are superseded by `mixed-mix-ab.jsonl`.
