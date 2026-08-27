# Reduced-precision self-play inference: harness and measurement plan

**Nothing in this document has been measured.** The work was done on a machine with no NVIDIA GPU
(Intel Iris Xe, `torch 2.12.1+cpu`), so every number below is either quoted from an existing
benchmark in this repository or derived arithmetically from one. The harnesses exist so the
question can be settled on the node; the plan says exactly what to run.

## The question

Leela Chess Zero reports roughly +50 % inference throughput from int8 with no measurable quality
loss. Self-play is ~99.7 % of this project's compute, so the same multiplier would be worth a great
deal. Does it apply to these networks on an RTX 4070 SUPER?

## What the existing evidence already says

Two independent routes through the repository's own measurements give the same answer, and it is
not the one the Leela result would suggest.

Network arithmetic, measured on CPU by instantiating the three v9 progressive-sizing rungs and
counting convolution and linear multiply-accumulates:

| Model | Parameters | GFLOP / position | `hidden_size` mod 16 |
| --- | ---: | ---: | ---: |
| `chess-cnn-12x128-dense4` | 3.88 M | 0.4301 | 0 |
| `chess-cnn-14x152-dense4` | 6.05 M | 0.7050 | **8** |
| `chess-cnn-18x176-dense4` | 10.06 M | 1.2115 | 0 |

RTX 4070 SUPER peaks, from 7168 CUDA cores at a 2.475 GHz boost clock and the Ada consumer tensor
ratios (bfloat16 dense = 4x FP32, int8 dense = 8x FP32; the latter doubles to NVIDIA's quoted
"568 AI TOPS" with sparsity):

| Tier | Dense peak |
| --- | ---: |
| FP32 CUDA core | 35.5 TFLOPS |
| BF16 tensor | 141.9 TFLOPS |
| INT8 / FP8 tensor | 283.9 TOPS |

Achieved rate, two routes:

| Route | Source | Achieved | Share of BF16 peak |
| --- | --- | ---: | ---: |
| Node throughput 648,472 searches/s ÷ 8 GPUs = 81,059 pos/s, × 0.4301 GFLOP | `self-play-graph-multiworker-8xrtx4070super-20260824/README.md` | 34.9 TFLOPS | **24.6 %** |
| Batch 146 against "2000 µs of GPU work" | `self-play-submission-8xrtx4070super-20260824/README.md` | 31.4 TFLOPS | **22.1 %** |

The forward pass is genuinely the bottleneck — the same two benchmarks report 92.4–93.3 % GPU
utilisation at the production cap and host submission down to 31 µs per call after CUDA graph
replay — but it is running at roughly a quarter of the precision tier it already has. Doubling the
tensor-core peak cannot help a workload that is not limited by that peak. The `self-play-submission`
benchmark also counted 202 CUDA kernels per call with **no fusion groups**, which is consistent with
a large share of the time going to memory-bound elementwise work rather than to convolution math.

Two cheaper levers were rejected on 2026-08-24 for reasons that no longer hold, both quoted from
`self-play-submission-8xrtx4070super-20260824/README.md`:

- `channels_last`: "202 → 156 kernels but *higher* host CPU, and shifts policy outputs by 0.31"
- `cudnn.benchmark = True`: "net loss (4806 → 5885 µs at batch 146)"

Both were measured while host dispatch was the binding constraint at 3537 µs per call. CUDA graph
capture has since cut that to 31 µs, and the algorithm a warm-up selects is baked into the replayed
graph, so its host cost is paid once per captured bucket and never again. Neither rejection is
evidence about the current build.

## Harnesses

### Throughput and roofline

`py/tools/benchmark_reduced_precision_inference.py` has two arms.

The **network arm** times all three rungs at the production batch sizes across every precision and
memory format the native runtime can select, and reports achieved TFLOPS against the device's own
derived bfloat16 peak.

The **roofline arm** times the bare implicit-GEMM shape of one trunk convolution
(`M = batch × 64`, `N = hidden`, `K = hidden × 9`) in bfloat16, float16, int8 (`torch._int_mm`) and
float8 e4m3 (`torch._scaled_mm`). No int8 or float8 convolution exists on this path, so this arm
measures the ceiling such a kernel could reach rather than a runnable network. Unavailable
precisions are recorded with their reason instead of aborting the run.

From `py/`, on an authorised idle GPU:

```bash
uv run python -m tools.benchmark_reduced_precision_inference \
  --gpu-id 0 \
  --batch-sizes 64 241 320 \
  --warmup-iterations 16 \
  --duration-seconds 2 \
  --output-path benchmark-results/reduced-precision-inference.json \
  --acknowledge-gpu-load
```

The native benchmark takes the same knobs, so the whole pipeline can be measured rather than the
model alone:

```bash
./InferenceBenchmark --model model_N.jit.pt --mode pipeline --batch-size 320 \
  --precision bfloat16 --memory-format channels_last --cudnn-benchmark 1
```

### Quality agreement

`py/tools/measure_inference_precision_agreement.py` compares every candidate variant against the
shipped bfloat16 contiguous path on stored positions from the run's own evaluation dataset, and
reports per variant: top-1 policy agreement over legal moves, unrestricted top-1 agreement, mean and
maximum KL divergence of the legal-move policy, and mean and maximum value error. Top-1 is measured
over legal moves because that is the distribution the search consumes; the metric functions are
pure and unit-tested in `py/test/test_inference_precision_configuration.py`.

```bash
uv run python -m tools.measure_inference_precision_agreement \
  --model-path /workspace/.../model_N.jit.pt \
  --dataset-path /workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v1.bin \
  --position-count 480 \
  --output-path benchmark-results/precision-agreement.json \
  --acknowledge-gpu-load
```

## Measurement plan

Run in this order and stop as soon as an arm fails its gate. All of it is single-GPU and idle; none
of it needs a production run started or reconfigured.

1. **Roofline arm, ~5 GPU-minutes.** If int8 GEMM does not reach at least ~2x the bfloat16 GEMM at
   these shapes, no int8 convolution kernel could either, and the question is closed. Expect
   `chess-cnn-14x152-dense4` to be recorded as unavailable: `hidden_size` 152 is not a multiple of
   16, which is the alignment int8 and float8 tensor-core kernels require.
2. **Network arm, ~30 GPU-minutes.** The decisive number is `fraction_of_bfloat16_peak` for
   `bfloat16 / contiguous`, which should reproduce the 22–25 % derived above. Then read
   `bfloat16 / channels_last` and the `cudnn_benchmark` variants. If channels-last closes a
   meaningful part of the gap to peak, that is the win, and it costs no precision change.
3. **Quality agreement, ~10 GPU-minutes,** for any variant step 2 shows is faster. Gate: legal
   top-1 agreement > 99 %, mean KL below ~1e-3, value MAE below ~1e-3. The 0.31 logit shift the
   2026-08-24 note recorded for channels-last is a raw-logit difference and says nothing on its own
   about agreement; this harness is what decides it.
4. **Pipeline arm** via `InferenceBenchmark --mode pipeline` at batch 320 for any surviving variant,
   to confirm the model-only gain survives graph capture, staging and the device-to-host copy.
5. Only then, a full-topology A/B on the node under `deployment/run_control.sh`, with the archive
   fetched under `.codex-diagnostics/`.

Record for every measurement: `experiment_configuration_sha256`, source revision, node identity and
`nvidia-smi`, per `documentation/benchmarks/TEMPLATE.md`. Results belong in a dated directory under
`documentation/benchmarks/`, not in this file.
