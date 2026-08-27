# Convolutional inference throughput — RTX 4070 SUPER, 2026-08-27

| | |
| --- | --- |
| `experiment_configuration_sha256` | `37a17001b1d26811732b73f114e9bfae729653f6968c56cd4160bdedb705a09f` for the hash-neutral baseline; candidate hashes are recorded with their self-play results |
| Source revision | `b20deeb5a44698ca3aee25fbcc44f9e9a74ba08f` (`clean`) |
| Node | Vast.ai `48910449`; 1× RTX 4070 SUPER 12 GiB; driver 580.159.03; 9.6 effective CPUs; 188 GiB RAM |
| Date | 2026-08-27 |

## Provisioning

The fresh node was provisioned from `cnn-inference-throughput`, based on
`int8-selfplay-inference`. The successful clean retry ran from 16:58:00Z to approximately
17:14:30Z: **16 minutes 30 seconds**. The earlier cached dependency download was deliberately
stopped at the original 30-minute time box and was not counted as a successful provision.

| Component | Observed value |
| --- | --- |
| GPU | 1× NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB, compute capability 8.9 |
| Driver / driver maximum CUDA | 580.159.03 / 13.0 |
| Locked PyTorch | 2.12.1+cu126 |
| PyTorch CUDA / cuDNN | 12.6 / 9.10.2 |
| System CUDA toolkit | 12.8.93 |
| CPU | 80 visible logical CPUs, 2× Xeon E5-2673 v4, cgroup quota 9.6 CPUs |
| RAM / disk | 188 GiB RAM; 32 GiB ephemeral overlay; `/workspace` is not volume-backed |
| Engines | Stockfish 18 and 13; KataGo 1.17.1 `cuda12.8-cudnn9.8.0` |

The Release extension compiled for SM 8.9. KataGo reported its CUDA backend and passed the 7×7
and 9×9 analysis smokes. Stub generation emitted default-argument diagnostics, but
`py/AlphaZeroCpp.pyi` matched the committed file byte-for-byte and the checkout remained clean.

## Method

All throughput measurements used the production v9 12×128 convolutional shape and batch cap 320.
The roofline arm used a seeded random model because weights do not affect kernels, FLOPs, or memory
traffic. Native pipeline and self-play comparisons use identical weights within each A/B. The
native pipeline includes CUDA graph replay, staging, device-to-host copies, and result processing.

Kernel timing is one warmed, graph-captured bfloat16 contiguous replay collected with
`tools.profile_inference_kernel_breakdown`. Percentages divide summed kernel time by the GPU span;
overlap is accounted for when measuring the launch gap. BatchNorm is already folded at export.

## Results

### Roofline

The delivered network harness measured the baseline at **96,671 positions/s**, **41.6 TFLOP/s**,
or **29.3%** of the card's 141.9 TFLOP/s dense bfloat16 peak. That is materially above the
previously derived 22–25%, but is still far from compute-bound. Bfloat16 channels-last reached
101,414 positions/s (1.049×); FP16 contiguous reached 98,717 (1.021×); and FP16 channels-last
reached 102,589 (1.061×). Enabling `cudnn.benchmark` inside that shared Python process was neutral,
but the isolated native runs below show that the process-wide cuDNN algorithm cache concealed its
interaction with channels-last.

### Graph-captured kernel breakdown

| Class | Kernels | Time (µs) | GPU span |
| --- | ---: | ---: | ---: |
| Convolution | 25 | 2,155.3 | 63.60% |
| BatchNorm | 0 | 0.0 | 0.00% |
| Activation | 17 | 80.1 | 2.36% |
| Residual add + activation | 12 | 149.6 | 4.41% |
| Global pooling reduction/broadcast/cat | 24 | 175.7 | 5.18% |
| Layout transforms / memory operations | 77 | 463.4 | 13.67% |
| Other | 40 | 356.8 | 10.53% |
| Launch gaps | — | 8.1 | 0.24% |
| **Total** | **195** | **3,388.9 span** | **100.00%** |

The largest individual groups were 24 trunk convolutions (2,121.9 µs), 27 unfused bias-add
kernels (321.7 µs), 50 NCHW→NHWC transforms (245.5 µs), and 25 NHWC→NCHW transforms (214.9 µs).
CUDA graph replay has genuinely eliminated launch gaps; the remaining problem is kernel count and
memory traffic, especially the 75 format conversions.

With BF16 channels-last and cuDNN search enabled, the replay fell from 195 to **124 kernels** and
from 3,388.9 to **2,833.2 µs**. Memory-operation kernels fell from 77 to 2 (0.10% of the span),
while launch gaps remained negligible at 0.18%. This directly confirms that avoiding the internal
NCHW↔NHWC transforms is the mechanism behind the throughput gain.

### Native graph-captured pipeline, batch 320

Each row is three independent processes, 1,000 measured calls per process after warm-up. The
pipeline includes staging, graph replay, device-to-host copies, and result processing.

| Precision / layout / cuDNN search | Positions/s, three runs | Mean | CV | vs baseline |
| --- | --- | ---: | ---: | ---: |
| BF16 / contiguous / off | 97,862; 98,188; 97,791 | 97,947 | 0.21% | 1.000× |
| BF16 / channels-last / off | 103,089; 103,452; 103,959 | 103,500 | 0.42% | 1.057× |
| BF16 / contiguous / on | 97,518; 97,128; 98,219 | 97,621 | 0.56% | 0.997× |
| BF16 / channels-last / on | 113,254; 113,019; 113,484 | 113,252 | 0.21% | **1.156×** |
| FP16 / channels-last / on | 113,258; 113,934; 114,860 | 114,017 | 0.70% | **1.168×** |

The FP16 row and its baseline used a zero-parameter current-shape model because unconstrained
random Kaiming weights overflowed the FP16 value head and correctly failed the runtime's WDL
validation. Zero weights retain identical operators, tensor shapes, kernels, FLOPs, and memory
traffic. Its BF16 baseline was 97,644 positions/s (0.22% CV), consistent with the seeded-random
baseline above.

### End-to-end self-play

The production single-GPU topology is four processes, one inference worker per process, two
outstanding batches per worker, batch cap 320, 512 games per process, and two parallel searches.

| Configuration | Searches/s, three runs | Mean | CV | Average batch | vs baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| BF16 / contiguous / cuDNN search off | 95,750; 94,398; 96,226 | 95,458 | 0.99% | 317.10 | 1.000× |
| FP16 / channels-last / cuDNN search on | 114,360; 112,596; 114,709 | 113,888 | 0.99% | 317.05 | **1.193×** |

A single BF16 / channels-last / cuDNN-search run reached 109,785 searches/s (1.150×). FP16 is
therefore a real part of the integrated win even though it adds little in an isolated single-context
pipeline. The short 10-second smoke was consistent at 1.203× but is not used for the decision.

## Interpretation

The graph is already substantially optimized: BatchNorm contributes no kernels, residual add and
ReLU are fused, and launch gaps are negligible. The observed win comes from choosing a layout and
cuDNN algorithm that avoid format-conversion traffic, not from higher theoretical FP16 throughput.
FP16 adds less than one percentage point over the best BF16 candidate.

### Batch-cap sweep

These are end-to-end BF16-contiguous self-play runs with the production topology. The 320 result
is the three-run mean above; the other caps are single runs.

| Batch cap | Searches/s | Inference positions/s | Average batch | Dense BF16 peak | vs cap 320 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 81,910 | 81,262 | 254.51 | 24.64% | 0.858× |
| **320** | **95,458** | **94,782** | **317.10** | **28.73%** | **1.000×** |
| 384 | 91,545 | 90,982 | 379.70 | 27.58% | 0.959× |
| 448 | 99,205 | 98,631 | 441.24 | 29.90% | 1.039× |

The 448 cap was the fastest alternative, but its 1.039× gain is below 1.15× and is therefore
inside noise and not worth shipping. Keep the production cap of 320.

### Process and worker topology

The sweep used BF16 contiguous inference, cap 320, and 15-second single runs. The production row
is the more reliable three-run, 30-second mean above.

| Processes | Workers/process | Outstanding/worker | Searches/s | Average batch | vs production |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2 | 71,519 | 316.25 | 0.749× |
| 2 | 1 | 2 | 87,267 | 317.43 | 0.914× |
| 4 | 1 | 1 | 89,141 | 319.16 | 0.934× |
| 2 | 2 | 1 | 87,380 | 317.77 | 0.915× |
| 2 | 2 | 2 | 82,314 | 249.89 | 0.862× |
| **4** | **1** | **2** | **95,458** | **317.10** | **1.000×** |

Keep four processes, one inference worker per process, and two outstanding batches per worker.
Extra workers split batches and add contexts without improving throughput on this node.

## Acceptance-gate status

The throughput half of the gate passes for FP16 + channels-last + cuDNN search: 1.193× with
0.99% CV, against a 0.99% baseline CV. The quality half is **blocked and was not run**. No trained
checkpoint compatible with the current v9 interface could be obtained:

- the `four-day-baseline` release and the local archived baseline are trained 12×112 models with
  29 input planes and two outputs;
- the current v9 runtime expects 12×128, 35 input planes, and three outputs;
- an exhaustive search of the local `.codex-diagnostics` archives found no trained v9 TorchScript
  checkpoint.

Running the agreement tool on random or zero weights would make the policy outputs near-uniform
and provide a meaningless pass, so it was deliberately not done. A real evaluation corpus was
generated successfully with Stockfish 18: `chess-stockfish-evaluation-v1.bin` has SHA-256
`40722feac84a2c44859d8f47bfab6a3540881d577b7879e35f95d94ef42ce54f`. Once a compatible trained
v9 checkpoint is supplied, run `measure_inference_precision_agreement.py` against the BF16
contiguous reference and require legal-move top-1 agreement above 99%, policy KL below 1e-3, and
value error below 1e-3.

Accordingly, the candidate is **not approved to ship yet**, despite passing the throughput gate.

## Lever verdicts

| Lever | Best end-to-end multiplier | Variance | Quality gate | Verdict |
| --- | ---: | ---: | --- | --- |
| BF16 contiguous baseline | 1.000× | 0.99% CV, 3 runs | Reference | Baseline |
| Channels-last alone | 1.057× native pipeline | 0.42% CV, 3 runs | Blocked | Inside noise; do not ship |
| cuDNN search alone | 0.997× native pipeline | 0.56% CV, 3 runs | Blocked | No benefit |
| BF16 channels-last + cuDNN search | 1.150× self-play | One run | Blocked | Borderline and unreplicated; do not ship |
| FP16 contiguous | 1.021× network harness | One harness run | Blocked | Inside noise; do not ship alone |
| **FP16 channels-last + cuDNN search** | **1.193× self-play** | **0.99% CV, 3 runs** | **Blocked** | **Throughput passes; do not ship until quality passes** |
| Batch cap 448 | 1.039× self-play | One run | Output-identical config | Inside noise; keep 320 |
| Alternate topology | 0.934× best alternative | One run each | Output-identical config | Keep 4×1×2 |
| Conv-BN folding | Already applied | — | Mathematically equivalent in eval | Keep existing export fold |
| Residual add + activation fusion | Already applied | — | Existing graph | Keep; 4.41% of baseline span |
| `torch.compile` | Not benchmarked | — | Incompatible runtime boundary | Stop without runtime redesign |

The 2026-08-24 blanket rejections are stale. Channels-last alone now helps modestly because CUDA
graphs removed the host-submission penalty; cuDNN search alone does not. Their combination is the
important interaction: BF16 reaches 1.156× in the isolated pipeline and 1.150× in its single
self-play run, while adding FP16 produces the repeated 1.193× result. The combined candidate is
worth gating, but neither switch alone meets the ship threshold.

## Export-time graph optimization

Conv-BatchNorm folding was already implemented before this work. Python's export path builds an
eval-mode `InferenceNetwork` and calls `fuse_model()`, which uses `fuse_conv_batchnorm`, before
scripting and saving it. C++ then calls `torch::jit::freeze` after loading. The measured zero
BatchNorm kernels confirm that folding is active.

The generation hot-swap contract is also already correct: every newly trained generation is
exported and folded on the Python side, then the C++ loader freezes that updated module.
`adoptWeightsInPlace` copies those already-folded constant tensors into the current frozen
module, preserving the addresses captured by CUDA graphs. Folding is therefore recomputed once
per export, not left stale after adoption, and no runtime change is needed.

Residual add + ReLU is already emitted as 12 `fused_add_relu` kernels. No
`torch.jit.optimize_for_inference` call exists, but the relevant BN and residual fusions are
already visible in the captured graph. `torch.compile` was stopped at the compatibility analysis:
Inductor's Python `OptimizedModule`/code cache is not a serialized TorchScript module, while this
runtime loads a TorchScript module in C++, enumerates its frozen constants for in-place adoption,
and captures those addresses in CUDA graphs. Supporting it would require redesigning the runtime
and hot-swap boundary, which is out of scope.

## Global-pooling cost

Global-pooling reduction, broadcast, and concatenation used **5.18%** of the baseline GPU span
(175.7 µs across 24 kernels). It fell to 4.48% with the best BF16 layout candidate. This is
reported only; no architecture change was made.

## Single-GPU limitation

The per-GPU kernel, layout, precision, and batch conclusions transfer directly because this is
the production GPU model. Topology conclusions are weaker: this node has one GPU, so it cannot
reproduce eight-GPU PCIe pressure, 32 CUDA contexts, inter-GPU CPU contention, or trainer steps
competing with self-play. The 4×1×2 recommendation should be confirmed on the next idle multi-GPU
node before changing production topology.

## Reproduce

All remote invocations were made through `deployment/remote_command.sh 154.64.230.50:50623`.
The principal commands were:

```text
python py/tools/benchmark_reduced_precision_inference.py --batch-sizes 320 --output-json ...
python py/tools/profile_inference_kernel_breakdown.py --model-path ... --batch-size 320 --graph-capture ...
cpp/build-benchmarks/benchmark/InferenceBenchmark --model ... --batch-size 320 --precision ...
deployment/benchmark_self_play_search.sh --game chess --gpus 1 --processes ... --duration-seconds ...
```

Each self-play manifest records the effective configuration and source revision. Candidate
self-play configuration SHA-256 was
`e742bf07260b3328fda4e80d28532bc016e2337e357753897eabe4e69df4c163`; the BF16 combined candidate
was `038a0696d2a09c1a4a74820bd8c7c2af4b7ff5e81ce5b9b758796a075cc217b3`.

Validation run for the code supporting these measurements:

- `uv run ruff format .` — 259 files unchanged;
- `uv run ruff check --fix .` — passed;
- `uv run python -m pytest --import-mode=importlib .\test -q` from `py` — 686 passed,
  52 skipped, 83 warnings in 47.04 seconds;
- Release C++ extension, benchmark build, engine smokes, and measured executables on the node —
  passed.

No C++ source was changed, so clang-format and CompileCheck were not applicable.
`cpp/run-clang-tidy.sh` was not run because the work did not proceed to a large C++ change.

## Files

All results were copied off the ephemeral node. `raw/` contains the original profiler JSON/logs,
native benchmark logs, and complete self-play manifests, worker results, CPU samples, and GPU
samples. `evaluation-artifacts/` contains the generated real-position dataset, its manifest, and
the deterministic opening corpus. `SHA256SUMS` inventories every exported artifact.
