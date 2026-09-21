# 5. Systems optimization

## Native search and direct inference

The largest early gain came from moving rules, search-tree ownership, and inference coordination into C++. A naive
Python PUCT implementation remains only as an order-of-magnitude reference in the
[Python MCTS baseline](../benchmarks/naive-python-mcts-rtx3060-20260816/README.md). Production workers submit tensors
directly to a native batched inference runtime, avoiding Python message serialization and result processing in the
inner loop.

The throughput history is preserved in
[the consolidated history](../benchmarks/throughput-history-chess-20260821/README.md) and the early
[direct-inference study](../benchmarks/self-play-direct-inference-rtx3060-20260722/README.md). Because those records
span different GPUs, revisions, and workloads, this report treats them as an engineering chronology rather than one
clean speedup factor.

## Batching and submission cost

High GPU utilization requires enough concurrent roots, but the host must prepare and submit them cheaply. The project
tested games per process, inference workers, batch caps, outstanding requests, timeouts, memory format, and search
parallelism. Submission profiling found that assembling input batches and replaying CUDA graphs could dominate once
network kernels became fast. Changes that reduced submission work improved node throughput even when raw model
forward speed was unchanged.

The principal evidence is:

- [self-play submission optimization](../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md);
- [multi-worker CUDA graph replay](../benchmarks/self-play-graph-multiworker-8xrtx4070super-20260824/README.md);
- [CPU search profile](../benchmarks/self-play-search-cpu-i7-11370h-20260824/README.md);
- [self-play/training pause trade-off](../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md).

The final topology runs four self-play processes per GPU with 512 parallel games per process, inference batches up to
320, and two outstanding batches. Half of the processes are paused during a training quantum, reflecting a measured
trade between fresh-data production and trainer contention.

Performance must be reported as a ladder rather than one headline rate: model forwards, exclusive-search
simulations, completed games, admitted replay positions, generations/optimizer steps, and finally strength per
wall-clock hour. V39 is the clearest warning. INT8 improved exclusive search far more than admitted replay because
game length, completion gating, trainer overlap, and replay reuse intervened
([decomposition](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md)). Later matched controls measured
the backend for [12x128](../benchmarks/v76-v35-small-prefold-backend-20260918/README.md) and
[14x160](../benchmarks/v76-v35-medium-prefold-backend-20260918/README.md). None of these **T** results alone proves
an Elo gain.

## Training throughput and replay I/O

Persistent DDP ranks avoid repeated process and model startup. The trainer reads vectorized batches from a
memory-mapped columnar store, prefetches batches, and uses one rank per GPU. Early DDP, batch-size, precision, and
contention measurements are in the
[training-throughput benchmark](../benchmarks/chess-training-throughput-rtx3060-20260812/README.md).

Replay ingestion evolved after a dispatcher could become CPU-bound on a large completed-game inbox. The current
materialization design partitions work before expensive conversion, bounds scans and staging, quarantines individual
bad games, and makes systemic rejection rates fatal. This is primarily a reliability optimization: a run that
quietly stops feeding the trainer has zero useful throughput regardless of GPU utilization.

The credit ledger is a scientific accounting boundary as well as scheduling infrastructure. It funds optimizer
quanta from admitted materialized positions, so rejected games, stalled ingestion, or a changed replay ratio cannot
silently appear as ordinary training. Credit-runtime and replay-loader benchmarks establish recovery, accounting,
and sustained loading (**M/T**), not chess strength.

## TorchScript, TensorRT, and precision

TorchScript provided the original trimmed inference artifact and remains the bootstrap/fallback path. TensorRT FP16
then demonstrated a production-compatible speed path with acceptable reference fidelity. Naive post-training INT8
did not meet policy fidelity requirements, which shifted the work toward quantization-aware training and architecture
changes. The completed feasibility evidence is in
[TensorRT FP16/INT8](../benchmarks/chess-tensorrt-int8-rtx4070s-20260912/README.md),
[architecture screening](../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md), and
[native backend measurements](../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md).

The rejected alternatives matter. Full-trunk post-training INT8 was fast but catastrophically changed policy and
value outputs. Calibration sweeps, partial early-block quantization, SmoothQuant, weight-only variants, and FP8 did
not pass the joint speed/fidelity gate. These are local implemented negative results in the
[salvage investigation](../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md), not universal claims about
those methods. `torch.compile` improved an eager diagnostic but did not beat the then-production TorchScript path.

The final networks use activation caps and scaled post-activation residual branches to make quantization tractable.
Only the backbone convolutions are quantized; policy/value heads, linear layers, and the start block remain outside
INT8. Training remains bfloat16. A QAT model is recalibrated each generation, exported to explicit Q/DQ ONNX, and
refit into model- and batch-specific TensorRT engines.

## The pre-fold serving design

BatchNorm folding changes the trainable representation and can introduce a difficult transition. The settled design
keeps the authoritative training model pre-fold for a long horizon and creates a deployment copy for serving. This
allows INT8 self-play from the first trained generation without forcing the optimizer through an early irreversible
fold. The frozen-replay schedule screens established that QAT could learn stably and that fold timing and deployment
learning rate materially affected short-run fitting.

The final config sets the training fold boundary far beyond the expected early serving phase, uses per-generation
recalibration, and warms deployment state separately. These values describe the run; the final online effect belongs
in Chapter 7.

## TensorRT refit failure and fidelity redesign

During progressive growth, a 14x160 candidate appeared hundreds of Elo weaker despite better float training loss.
The immediate symptom was a stale refit template, but deeper controlled tests found the actual cause: TensorRT had
optimized around equal Q/DQ scales in the template source and produced invalid results when refitting distinct
scales at high optimization levels. Refitting reported success.

The fix makes template scales pairwise distinct before building and uses a safer default optimization level. The
investigation also replaced random unmasked fidelity probes with real chess positions and legal-action masking;
illegal logits and near-ties had made the old top-1 metric misleading. The full correction, including superseded
intermediate hypotheses, is in
[INT8 template staleness](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md).

This episode motivates a general rule: inference artifacts must be evaluated as semantic models, not accepted because
conversion APIs return success or because an unrepresentative aggregate error is small.

Refit and rebuild must remain distinct. A template fixes graph structure and tactic choices; recurring refit updates
named weights and quantization constants cheaply. A rebuild is slower and can be periodic insurance, but rebuilding
on a cadence is not the fix for invalid equal-scale optimization. Templates are specific to model geometry, QAT
phase, and batch shape; the final configuration separates self-play and evaluation artifacts.

## Model shape and memory format

CNN width did not scale smoothly with parameter count or arithmetic. Production-card sweeps found sharp
shape-dependent efficiency, including widths that were slower despite fewer operations. Channels-last format and
cuDNN autotuning recovered meaningful throughput for relevant stages but did not explain every discontinuity.
Progressive rungs therefore require direct measurement at their actual batch sizes. See the
[width controls](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md) and
[CNN inference benchmark](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md).

## Evaluation and interactive serving

Evaluation and public play reuse the native engine but operate in different latency regimes. Ladder evaluation
amortizes inference across many games; browser/UCI play needs bounded response time and result processing for one
position. Local protocol smokes, CUDA result-processing controls, and integrated interactive measurements establish
deployability and CPU/result-path costs (**M/T**), not training efficacy. Saturated ladder throughput must not be
reported as single-game latency.

## Hardware and transfer limits

Node comparisons informed rental choice, but a GPU model name is not a transferable performance result. CPU quota,
PCIe/NUMA topology, host contention, driver, power, batch shape, and simultaneous tenants matter. Cross-era
throughput history is chronology only; controlled claims stay attached to one hardware/runtime/workload identity.
The relevant controls are the [four-node comparison](../benchmarks/node-comparison-vast-4nodes-20260821/README.md)
and [same-GPU host comparison](../benchmarks/node-comparison-8xrtx4070super-20260824/README.md).

## What was not retained

- Fully asynchronous training was not implemented as a separate algorithm. The runtime overlaps self-play,
  materialization, evaluation, and blocking training quanta, but the learner still has explicit quantum boundaries.
- An inference cache was audited and rejected because exact repeated inputs were too rare.
- Monte Carlo graph search was implemented on a branch and rejected for this chess workload.
- `torch.compile` was useful in selected trainer studies but is disabled in the final configuration; TensorRT is the
  serving compiler.
