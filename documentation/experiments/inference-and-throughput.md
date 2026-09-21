# Inference and throughput experiments

## Ledger

| Technique | Status | What the evidence establishes | Principal evidence |
| --- | --- | --- | --- |
| Native C++ search and direct inference | **Retained** | The project replaced the Python search hot path with a native batched engine and direct model execution. Benchmarks document the successive throughput gains and integration controls. | [pre-port record](../history/pre-cpp-port/chess-port.md), [native baselines](../benchmarks/self-play-cpp-final-tuning-20260720/README.md), [direct inference](../benchmarks/self-play-direct-inference-4x4070-super-20260722/README.md) |
| Batched leaf inference | **Retained** | Batch-cap, timeout, games/process, and worker-count studies established batching as the dominant systems lever. Optimal values remain workload- and model-dependent. | [batching timeout](../benchmarks/self-play-cpp-batching-timeout5000us-20260720T073957Z/README.md), [latency study](../benchmarks/chess-self-play-latency-rtx3060-20260812/README.md), [CPU/search study](../benchmarks/self-play-search-cpu-i7-11370h-20260824/README.md) |
| Multiple self-play processes per GPU | **Retained** | Four processes per GPU are used in the final topology. Benchmarks show process-level fill can exploit headroom, especially for smaller models; completion counters from short cold-start runs are not steady replay throughput. | [Final config](../../py/configs/production/chess-final-config.yaml), [v39 study](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md), [progressive sizing](../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md) |
| TorchScript BF16 fused inference | **Superseded** | This was the validated production path and remains the generation-zero/bootstrap fallback. TensorRT QAT is now the selected post-bootstrap backend. | [kernel controls](../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md), [CNN throughput](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| `torch.compile` inference | **Implemented and rejected** | It improved eager direct-policy forwards in a diagnostic, but the actual production TorchScript path was faster and remained the control until TensorRT. | [direct-policy diagnostic](../benchmarks/chess-direct-policy-inference-rtx4070s-20260818/README.md), [kernel control](../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md) |
| CUDA graph capture | **Retained** | CUDA-graph execution improved isolated CNN inference and is part of the measured native/TensorRT serving machinery. Its isolated rate must not be equated with end-to-end actor rate. | [CNN throughput](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md), [TensorRT benchmark](../benchmarks/chess-tensorrt-int8-rtx4070s-20260912/README.md) |
| TensorRT FP16 | **Retained** | Validated as a compatible native backend and used where an INT8 template is unavailable or inappropriate. It is also the direct denominator for quantization speedups. | [TensorRT benchmark](../benchmarks/chess-tensorrt-int8-rtx4070s-20260912/README.md), [native backend](../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md) |
| Full-trunk post-training INT8 | **Implemented and rejected** | Raw PTQ delivered high core throughput but catastrophically changed policy and value outputs; no speed number makes those engines valid. | [salvage investigation](../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md), [architecture screen](../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md) |
| Early-block INT8, SmoothQuant, FP8 | **Implemented and rejected** | These alternatives did not meet the joint fidelity/throughput gate. | [salvage investigation](../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md) |
| Scaled-post QAT with pre-fold deployment | **Retained** | Quantization-friendly residual blocks learn the replay target, and pre-fold QAT avoids the severe fidelity loss caused by folding only after training. It is the final inference/training design. | [replay screen](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| Post-training fold plus recovery | **Superseded** | A 5k continuation recovered much of the fidelity lost by late folding, while 10k was worse. The final run avoids this recovery path by remaining pre-fold. | [replay screen](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md) |
| TensorRT refit | **Retained** | Refit reduced recurring engine updates to a fraction of a second in the cadence diagnostic, making per-generation publication practical. | [cadence evidence](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/cadence/README.md), [native backend](../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md) |
| Phase/model/batch-specific engine templates | **Retained** | Typed templates are required because QAT phase, model geometry, and batch shape change the graph. Template lifecycle is production infrastructure. | [v39 study](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| TensorRT L4/L5 refit template built from equal Q/DQ scales | **Implemented and rejected** | The V90 collapse was reproduced as a TensorRT scale-equality optimization bug, not ordinary template staleness. Separating scales before build and defaulting to optimization level 3 fixed the controlled failure. | [template investigation](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md) |
| Channels-last and cuDNN autotuning | **Retained** | Production uses both. Shape sweeps measured substantial channels-last gains for relevant CNN widths; gains depend on batch and shape. | [attention viability, width controls](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md), [CNN throughput](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| DDP training | **Retained** | Eight-rank NCCL DDP is the final trainer topology. Benchmarks validated throughput and exposed when static-bucket/compiled variants were not worthwhile. | [DDP production](../benchmarks/ddp-production-training-20260720/README.md), [DDP model throughput](../benchmarks/ddp-model-throughput-20260720/README.md), [attention DDP](../benchmarks/chess-attention-ddp-rtx4070s-20260818/README.md) |

## From Python search to a native batched actor

The early Python MCTS benchmark is a reference implementation and performance floor, not the system that produced
the final run. The production path moved tree ownership, node allocation, selection, backup, rerooting, and batched
inference coordination into C++. A series of baselines then tuned arena allocation, batching timeout, direct
inference, worker topology, and result processing. The benchmark history is useful because it shows cumulative
engineering progress, but many rows use different GPUs, models, visit mixes, and contention. They should not be
plotted as a single controlled speedup curve without those qualifiers.

## Batch fill, latency, and concurrency

Peak model positions per second is not sufficient. Self-play has variable-length searches and games; mixed fast/full
search created a long tail; high process counts introduce staleness and longer individual games; and short runs
right-censor completions. The final topology—four processes per GPU, 512 games per process, batch cap 320, two
outstanding batches—was selected to keep the device fed while half the actors pause during optimizer work.

V39 is the clearest decomposition. Its deployment INT8 actor measured 1.861x the matched floating control in
exclusive search throughput, while live admitted replay positions improved only 23.3% relative to the cited V35
period. Different game length, completion gating, trainer overlap, and replay reuse account for much of that gap.
The technical report should present core inference, exclusive search, completed games, admitted positions, and
generations/hour as separate metrics.

## Why quantization required an architecture change

The first TensorRT INT8 result was fast and unusable. Full-trunk PTQ produced severe policy/value drift; calibration
choice, SmoothQuant, limited QAT, and FP8 did not meet the combined fidelity and speed gate. The successful route was
a bounded scaled post-activation residual block trained with fake quantization. Even then, folding BatchNorm only
after training caused a large discontinuity. Pre-fold deployment graphs—or a bounded recovery continuation—were
needed to restore agreement.

The retained evidence supports three distinct claims:

1. the scaled-post network can learn the frozen replay target under QAT;
2. its INT8 TensorRT engine can preserve useful output fidelity on held-out positions;
3. the native actor obtains a material measured throughput gain, including 14.4% for 12x128 and 39.1% for 14x160
   in later production-topology controls ([small](../benchmarks/v76-v35-small-prefold-backend-20260918/README.md),
   [medium](../benchmarks/v76-v35-medium-prefold-backend-20260918/README.md)).

These do not by themselves prove a strength gain. Quantization buys more self-play compute; the final run measures
whether the end-to-end training system converts it into Elo.

## Template lifecycle and fidelity

TensorRT refit separates graph construction from per-generation weights and makes updates cheap. The V90 incident
initially looked like ordinary template staleness, but the completed investigation corrected that interpretation:
scales *are* refittable, and same-lineage staleness added only modest KL with no measured Stockfish effect. The actual
failure required an optimization-level-4-or-5 template built with equal saturated Q/DQ scales, followed by a refit
that made those scales distinct. TensorRT silently retained an invalid equality-based optimization. Separating scale
constants before template construction and defaulting to optimization level 3 fixed the exact controlled failure;
periodic rebuilds are cheap insurance, not the root fix.

The final configuration names templates by model and QAT phase, separates self-play batch 320 from evaluation batch
64, and permits TorchScript bootstrap. The report should distinguish a template rebuild from the cheap recurring
weight refit and should preserve the investigation's corrected conclusion rather than repeating its initial
staleness hypothesis.

## Remaining evidence gaps

- Final-run telemetry must quantify how often TensorRT, FP16 fallback, and bootstrap TorchScript actually ran.
- The 19x176 final-stage INT8 speed/fidelity result should be reported from the terminal archive or a matched
  production-topology benchmark.
- Historical throughput comparisons need normalization by GPU, model, batch, visit mix, process count, and whether
  the trainer was active.
- No throughput result should be converted directly to Elo without the final learning curve.
