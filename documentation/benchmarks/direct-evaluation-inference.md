# Direct evaluation inference benchmark

`py/tools/benchmark_direct_inference.py` measures the inference mechanics intended for a
long-lived interactive search. The native benchmark generates deterministic random legal chess
positions, compresses and packs them before the timed forward loop, and reports those producer
costs separately. The synthetic network has the production shape but does not measure playing
strength.

The compared paths are:

- `cached`: the general self-play client with cache lookup, per-position tensors, promises,
  queues, stacking, policy filtering, and move decoding.
- `noncached`: the general client without cache lookup but with the other aggregation and result
  processing machinery.
- `direct`: one model, one dedicated CUDA stream, and reusable pinned host/device input and host
  output buffers. There is no per-position tensor, queue, promise, cache, or `torch::stack` in its
  hot path.
- `pipeline`: the same direct runner on a model-owning thread with three reusable SPSC batch
  slots. The producer writes board encodings directly into a free slot; the inference thread
  never accesses a board or tree node.
- `replicas`: independent model replicas, dedicated CUDA streams, and one synchronous batch per
  worker. This isolates CUDA concurrency before adding shared-tree concurrency.
- `direct_combined_batch`: one direct forward of `workers * worker_batch_size`, used to compare
  concurrent smaller forwards with a single batch containing the same number of positions.

Example from the repository root:

```powershell
python .\py\tools\benchmark_direct_inference.py `
  --executable .\cpp\build\DirectInferenceBenchmark `
  --model .\chess-12x112.jit.pt `
  --output .\direct-inference.json `
  --batch-sizes 16,32,48,50,64,128,256 `
  --workers 1,2,3,4,5,6,7,8 `
  --iterations 100 `
  --seed 7
```

The JSON report contains every successful configuration, explicit failures for configurations
that do not fit, executable and model hashes, the complete command, compiler/Python/Torch/GPU
provenance, encoding and packing rates, timed positions per second, coarse GPU utilization, and
peak resident VRAM. GPU utilization comes from 100 ms `nvidia-smi` samples spanning model load,
warmup, and the timed interval; it is only a coarse indicator, especially for short runs.

## RTX 3060 experiment, 2026-07-22

The experiment used an NVIDIA GeForce RTX 3060 12 GB, driver 595.71.05, Torch 2.12.0+cu130,
CUDA 13.0, Python 3.12.13, GCC 13.3, and a Release build (`-O3`, LTO). The model was the seeded
production chess shape: 12 residual layers, 112 hidden channels, 3,241,835 parameters, 13,032,860
bytes, SHA-256 `d85f782970cbd1cdc07f0d2c71ce3b9480517720fe0a858f61eaa99341d36e7d`.
It is synthetic, so the results measure throughput only.

The corrected full matrix used 100 timed forwards per worker and five untimed warmups. The most
useful comparison is batch 50: one direct runner sustained 14,584 positions/s and its three-slot
SPSC form sustained 14,716 positions/s. This confirms that the handoff itself is effectively
neutral when there is no tree-selection work to overlap. Three batch-50 model replicas sustained
31,404 positions/s versus 23,313 positions/s for one equal-total-size batch of 150, a 35%
advantage from concurrent streams at the same 150 in-flight positions.

| Worker batch | Direct | SPSC | Cached client | Non-cached client | Best replicas | Replica rate | Equal-total direct | Replica delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 4,681 | 4,698 | 985 | 2,726 | 4 | 13,611 | 18,507 | -26% |
| 32 | 9,109 | 9,071 | 1,881 | 4,461 | 4 | 24,868 | 25,967 | -4% |
| 48 | 13,900 | 13,865 | 1,888 | 5,324 | 3 | 30,803 | 26,243 | +17% |
| 50 | 14,584 | 14,716 | 1,889 | 3,635 | 3 | 31,404 | 23,313 | +35% |
| 64 | 18,364 | 18,565 | 2,120 | 6,836 | 3 | 30,164 | 26,964 | +12% |
| 128 | 25,766 | 25,752 | 3,212 | 9,242 | 3 | 29,497 | 31,390 | -6% |
| 256 | 27,754 | 27,664 | 3,983 | 9,096 | 3 | 32,055 | 32,481 | -1% |

Rates are positions/s. “Equal-total direct” is one batch containing `replicas × worker batch`
positions. Multiple streams help most in the middle batch range and hurt or become neutral once
one forward is large enough. Across the entire tested space, one very large batch of 1,792
reached 33,413 positions/s. That is only 6% above three batch-50 replicas while reserving almost
12 times as many leaves, which may be far more damaging to MCTS search quality.

No configuration failed or exhausted 12 GB VRAM. Three batch-50 replicas peaked at 317 MiB in
the coarse sampler, versus 221 MiB for one batch-150 runner. The small footprint reflects this
particular fused synthetic model; trained checkpoint memory should still be measured.

Raw inference throughput is an upper bound for MCTS: policy filtering, legal move decoding,
selection, expansion, and backup are deliberately outside the direct-forward timer. The cached
and non-cached client measurements include that result processing, so they quantify the complete
general-client path rather than only the model forward.

The experiment establishes two useful integration baselines:

1. The direct runner removes substantial general-client overhead and should replace the current
   inference client inside evaluation search.
2. Multiple replicas/streams can increase GPU throughput even before multiple threads touch the
   tree. A single tree-owning producer should therefore be tested while feeding several direct
   inference workers before accepting shared-tree locking and atomic-update complexity.

The full machine-readable matrix is checked in under
`documentation/benchmarks/direct-inference-rtx3060-20260722/`.
