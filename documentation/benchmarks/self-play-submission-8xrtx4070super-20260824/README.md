# Self-play throughput after cutting inference submission cost — 8× RTX 4070 SUPER, 2026-08-24

| | |
| --- | --- |
| `experiment_configuration_sha256` | `583f0ad49c7a7eebcaddd79ddb14bdf3e17611393a664e50e58d3fd33bddd682` (both arms; `inference_workers` overridden on the command line, see Method) |
| Source revision | baseline `b0719ac74789b097ed43e1757afda4dece385542` (`clean`); optimised = baseline + the fourteen native commits of this series |
| Node | Vast.ai `f1e810637724`, 8× RTX 4070 SUPER 12 GiB, driver 595.71.05, Intel Xeon E5-2673 v4, 80 logical CPUs, 188 GiB RAM; Python 3.12, torch 2.12.1+cu126 |
| Date | 2026-08-24 |

## Method

`py/configs/validation/vast-chess-4day-production-v2.yaml` at **generation 60**: 25 % full searches,
adaptive budget 200–750 visits, 150 fast searches, `parallel_searches` 4, inference batch cap 256.
Network: a freshly initialised 3,817,101-parameter CNN 12×128 with a dense ch4/rank96 policy head,
built by `py/tools/prepare_benchmark_model.py`. Topology: **32 processes (4 per GPU) × 512 parallel
games**, 2 warm-up batches, 90 s measured, started behind a barrier. Driver:
`py/tools/benchmark_self_play_search.py`, one process per worker, `OMP_NUM_THREADS=1`.

The two arms differ in the native build and in `inference_workers` (2 in the baseline, 1 in the
optimised arm), which the harness overrides on the command line; every other resolved field is
identical, hence the shared configuration hash. CPU is each worker's own `process_cpu_percent`
summed over the 32 workers, so it excludes the sampling helpers. GPU utilisation is the mean over
all eight devices sampled once a second for the measured window.

A second regime pins **the whole run to 24 CPUs** (`taskset --cpu-list 0-23`, 3 per GPU) to
represent a node whose CPU:GPU ratio is tighter than this one's.

## Results

### Headline, full production topology

Three repetitions per arm; the optimised arm spans 0.4 %, the baseline 2.2 %.

| | baseline | optimised | delta |
| --- | ---: | ---: | ---: |
| searches/s, median | 512,679 | **617,782** | **+20.5 %** |
| searches/s, runs | 512,679 / 515,072 / 504,106 | 617,782 / 619,506 / 617,251 | |
| aggregate process CPU | 52.6 cores | **19.8 cores** | **−62 %** |
| searches per CPU-core-second | 9,747 | **31,201** | **×3.2** |
| average inference batch | 141 | 222 | +57 % |
| model calls in the window | 339,390 | 258,179 | −24 % |
| GPU utilisation | 91.6 % | 93.5 % | |
| summed peak RSS | 58.5 GiB | 59.9 GiB | |

### CPU-constrained regime, whole node pinned to 24 CPUs

| | baseline | optimised | delta |
| --- | ---: | ---: | ---: |
| searches/s | 215,265 | **496,036** | **+130 %** |
| aggregate process CPU | 14.6 cores | 14.1 cores | |
| GPU utilisation | 44.5 % | 75.6 % | |

### Per-change attribution

Effects were isolated where the measurement could separate them; at the full topology the six
tree-side changes together fall inside the ±3 % run-to-run noise, so they are reported from the
regimes that can resolve them.

| change | isolated effect | how measured |
| --- | --- | --- |
| action id carried on the edge, stack bitmask in selection, table-driven plane expansion, cached repetition count, node field order, unchecked backup access, split position arena, visit columns | −21 % to −28 % native search CPU per simulation; +5.8 % node throughput | 5-repetition A/B (dev box, see `self-play-search-cpu-i7-11370h-20260824/`); node A/B at 24 CPUs |
| blocking CUDA event instead of the spinning default | 97× less CPU per wait in isolation, ~1.9 percentage points of CPU per process in situ | event micro-benchmark; full-topology A/B |
| single-dtype host copies via persistent staging | 155.5 → 61.9 µs of host CPU per call | isolated single calls at batch 256 |
| **CUDA graph replay of the forward** | **host CPU per call 3537 → 31 µs; single-process throughput 41,146 → 76,768 searches/s (+87 %)** | C++ micro-benchmark; single-process phase run |
| **`inference_workers` 2 → 1** | average batch 141 → 222; halves the CUDA contexts and submitting threads per process | full-topology A/B |

### Why submission dominated

Host CPU per inference call, measured in isolation on this node:

| stage at batch 146 | µs |
| --- | ---: |
| host→device copy | 25.0 |
| int8→bf16 cast | 22.3 |
| **forward** | **4534** |
| output staging | 55.9 |
| device→host | 50.4 |
| event record | 51.9 |

The forward is 94 % of it, and it is per-op dispatch rather than launch: an isolated
`Conv2d(128,128,3)` in bf16 costs 92.5 µs of host CPU at batch 1 and 84.9 µs at batch 256 — flat, so
fixed host work. Profiler self-time per conv: `aten::_convolution` 73 µs, `aten::cudnn_convolution`
43 µs, `cudaLaunchKernel` only 29 µs for five launches. 28 convolutions × ~85 µs ≈ 2.4 ms. The
TorchScript executor specialises (40 → 347 graph nodes after warm-up, 18 type-check/bailout guards)
but produces **no fusion groups**: 202 CUDA kernels per call at every batch size. Baseline wall per
call was 4849 µs against 2000 µs of GPU work, so the device idled 58 % of every call.

### Graph replay, correctness and cost

- Padding is **bit-exact**: a graph captured at 160 fed 146 real rows gives 0.0 max-abs difference,
  verified with random garbage in the 14 pad rows — the network has no cross-sample coupling.
- Padding 146 → 160 costs **+1.5 % GPU**, not the 9.6 % the row count suggests.
- Extra device memory: **38 MiB** for seven extra buckets sharing one pool; ~500 MiB per process
  measured end to end at two workers.
- Capture costs ~6 ms per bucket and is redone on model refresh.
- Mean final root visits agree between arms to three significant figures (286.8 vs 287.4), so the
  search itself is unchanged.

### What did not work

| attempt | outcome |
| --- | --- |
| `torch::jit::optimize_for_inference` | throws `INTERNAL ASSERT FAILED … no op for aten::cudnn_convolution_relu` in torch 2.12, in both Python and C++, and mutates the module in place before throwing |
| `cudnn.benchmark = True` | net loss (4806 → 5885 µs at batch 146) |
| `channels_last` | 202 → 156 kernels but *higher* host CPU, and shifts policy outputs by 0.31, far outside the 2e-2 tolerance |
| two capturing inference workers per process | faults the device with `cudaErrorIllegalAddress` within a minute; capture is therefore gated on a single worker |

## Interpretation

- The binding constraint on this node was never the tree: it was host-side per-op dispatch in the
  inference submission path, which starved the GPU inside every call. Fixing it is worth +20.5 %
  throughput and −62 % CPU together.
- The efficiency figure travels better than the throughput figure. This node has 10 CPUs per GPU and
  ends up GPU-bound at 93 %, which caps what any CPU saving can return. On a node with 3 CPUs per
  GPU the same binaries return **+130 %**. Expect a result between the two depending on the CPU:GPU
  ratio of the rented node, and closer to the upper figure whenever the trainer is also resident.
- `inference_workers: 1` is now the right setting and is part of the measured configuration. The
  second worker existed to overlap host submission with GPU execution; submission is 31 µs, so there
  is nothing left to overlap, and it costs a CUDA context and halves the batch fill.
- Reducing processes per GPU from 4 to 2 gives 567,349 searches/s at 14.0 cores — 8 % below the best
  throughput for 29 % less CPU, which is the better operating point if the trainer needs the cores.
- Numbers here are not comparable to `search-throughput-rtx4070-20260821/`: different network,
  different search schedule, different node.

## Reproduce

On a provisioned node, with a benchmark model built by `py/tools/prepare_benchmark_model.py`:

```bash
python3 tools/benchmark_self_play_search.py --run-config configs/validation/vast-chess-4day-production-v2.yaml --model /workspace/benchmodel/chess-v2.jit.pt --device 0 --worker-id 0 --inference-device cuda --games 512 --generation 60 --warmup-batches 2 --duration-seconds 90 --inference-workers 1
```

`py/tools/run_self_play_search_benchmark.sh` drives the full 32-process topology; the per-phase
counters used for attribution come from `py/tools/benchmark_native_self_play_loop.py`.

## Files

- `headline.jsonl` — per-worker JSON for the three baseline and three optimised repetitions.
- `constrained-24cpu.jsonl` — the same for the 24-CPU regime.
- `topology.jsonl` — processes-per-GPU sweep for the optimised build.
