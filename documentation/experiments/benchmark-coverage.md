# Benchmark coverage ledger

This table accounts for every `README.md` under `documentation/benchmarks` as of 2026-09-21. “Status” classifies
the technique or conclusion represented by the benchmark, not whether the artifact should be kept. All evidence
artifacts remain valuable, including rejected and superseded work.

| Benchmark record | Topic | Status | Technical-report use |
| --- | --- | --- | --- |
| [Benchmark index](../benchmarks/README.md) | Evidence navigation | **Infrastructure only** | Existing benchmark catalog and preservation conventions. |
| [Adaptive budget frozen-trunk probe](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md) | Search allocation | **Implemented and rejected** | Offline gate failure before the later online negative result. |
| [R3 adaptive termination audit](../benchmarks/adaptive-search-termination-r3-20260813/README.md) | Search stopping | **Audited and declined** | Proxy opportunity, missing traces, and target-risk limitations. |
| [Adaptive-search validation](../benchmarks/adaptive-search-validation-rtx4070s-20260818/README.md) | Search stopping | **Infrastructure only** | Mechanics validation on a random network; explicitly not threshold evidence. |
| [Architecture contention](../benchmarks/chess-architecture-contended-rtx3060-20260817/README.md) | CNN vs attention | **Inconclusive** | Early hardware comparison, bounded by contention and incomplete replicates. |
| [Attention DDP](../benchmarks/chess-attention-ddp-rtx4070s-20260818/README.md) | Training throughput | **Retained** | Supports ordinary DDP over the tested static-bucket alternative. |
| [Packed QKV](../benchmarks/chess-attention-packed-qkv-rtx3060-20260818/README.md) | Attention kernels | **Superseded** | Training improvement inside the attention line; original FP32 inference conclusion is invalid. |
| [Attention SDPA on RTX 3060](../benchmarks/chess-attention-sdpa-backends-rtx3060-20260818/README.md) | Inference controls | **Superseded** | Corrects an unrepresentative inference-precision comparison. |
| [Attention SDPA on RTX 4070S](../benchmarks/chess-attention-sdpa-backends-rtx4070s-20260818/README.md) | Inference controls | **Inconclusive** | Production-card model-forward/native rates; backend adoption required separate decision. |
| [Attention training](../benchmarks/chess-attention-training-rtx4070s-20260818/README.md) | Training throughput | **Inconclusive** | Batch-256 throughput control, not learning quality. |
| [Attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md) | Architecture/head | **Retained** | Selects the from-to head and bounds attention/CNN conclusions. |
| [Final progressive attention inference](../benchmarks/chess-direct-policy-final-progressive-rtx4070s-20260818/README.md) | Architecture throughput | **Superseded** | Acceptance measurement for an attention ladder not used by the final run. |
| [Direct-policy inference](../benchmarks/chess-direct-policy-inference-rtx4070s-20260818/README.md) | Compile/architecture | **Superseded** | Eager/compiled diagnostic; not the production serving path. |
| [Direct-policy kernel controls](../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md) | TorchScript/SDPA | **Superseded** | Establishes production TorchScript control before TensorRT adoption. |
| [Teacher-imitation distillation](../benchmarks/chess-distillation-probe-rtx3060-20260827/README.md) | Distillation | **Inconclusive** | Dataset-size, search-depth, sampling-defect, and student/teacher gap analysis. |
| [Fixed-batch overfit](../benchmarks/chess-overfit-rtx3090-20260819/README.md) | Objectives/architecture | **Infrastructure only** | Shows architectures/objectives can fit a fixed batch; not generalization or Elo. |
| [V34 replay distillation](../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) | Distillation/compression | **Inconclusive** | Equal-search, equal-time, and equal-MAC proxy comparison. |
| [Chess search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md) | Search | **Retained** | Principal search-depth, parameter, oracle, and parallelism evidence. |
| [Self-play latency](../benchmarks/chess-self-play-latency-rtx3060-20260812/README.md) | Batching | **Retained** | Explains batch fill and mixed-search tail behavior. |
| [SGD post-fold LR](../benchmarks/chess-sgd-postfold-lr-rtx4070s-20260914/README.md) | Optimizer/QAT | **Inconclusive** | Controlled frozen-replay LR ranking, not online Elo. |
| [SGD pre-fold factorial](../benchmarks/chess-sgd-prefold-factorial-rtx4070s-20260914/README.md) | Optimizer/QAT | **Inconclusive** | Controlled fold/schedule proxy and gradient behavior. |
| [SGD replay screen](../benchmarks/chess-sgd-replay-screen-rtx4070s-20260913/README.md) | Optimizer/QAT | **Inconclusive** | Establishes Nesterov SGD viability and candidate rates on frozen replay. |
| [Stockfish ladder baseline](../benchmarks/chess-stockfish-ladder-8xrtx3060-20260816/README.md) | Evaluation | **Infrastructure only** | Early ladder protocol/results context. |
| [TensorRT INT8](../benchmarks/chess-tensorrt-int8-rtx4070s-20260912/README.md) | Quantization | **Retained** | Core FP16/INT8 performance, calibration, and fidelity study. |
| [Terminal v34](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md) | Strength | **Retained** | Last completed public result and comparison baseline until final-run evaluation is frozen. |
| [Training throughput](../benchmarks/chess-training-throughput-rtx3060-20260812/README.md) | Trainer/actor contention | **Retained** | Mixed-contention throughput and replay-credit interpretation. |
| [V34 training dynamics](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md) | Learning/data | **Retained** | Late-training diagnosis and final-run design motivation. |
| [CNN inference throughput](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md) | Kernels/CUDA graphs | **Retained** | Production-shape memory-format, graph, batch, and kernel controls. |
| [Credit runtime stage 6](../benchmarks/credit-runtime-stage6-20260724/README.md) | Runtime | **Superseded** | Early credit-ledger stage evidence. |
| [Credit runtime stage 7](../benchmarks/credit-runtime-stage7-20260724/README.md) | Runtime | **Retained** | Integrated credit/runtime validation and recovery behavior. |
| [Cut-game value target](../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md) | Targets/termination | **Retained** | Target-choice study for censored games and forced full-search cut position. |
| [DDP model throughput](../benchmarks/ddp-model-throughput-20260720/README.md) | Training throughput | **Retained** | Model-shape DDP throughput control. |
| [DDP production training](../benchmarks/ddp-production-training-20260720/README.md) | Training throughput | **Retained** | Production DDP integration and contention baseline. |
| [Generation 936 deep match](../benchmarks/deep-match-generation936-50k-nodes-rtx4070s-20260906/README.md) | Evaluation | **Retained** | High-node strength evidence and evaluation-cost context. |
| [Direct inference RTX 3060](../benchmarks/direct-inference-rtx3060-20260722/README.md) | Inference | **Superseded** | Early direct-inference smoke. |
| [Go 7x7 baseline](../benchmarks/go-7x7-training-baseline-2xrtx3060-20260810/README.md) | Go/platform | **Inconclusive** | Demonstrates multi-game platform scope; Go programme was paused. |
| [INT8 template staleness](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md) | TensorRT lifecycle | **Implemented and rejected** | Corrects the initial staleness hypothesis and rejects unsafe L4/L5 equal-scale refit templates. |
| [Integrated interactive engine](../benchmarks/integrated-interactive-rtx3060-20260722/README.md) | Serving | **Infrastructure only** | End-to-end interactive engine integration. |
| [Interactive engine local](../benchmarks/interactive-engine-local-20260721/README.md) | Serving | **Infrastructure only** | Local interactive protocol smoke. |
| [Interactive result processing](../benchmarks/interactive-result-processing-20260722/README.md) | Serving | **Infrastructure only** | CPU/result-processing control. |
| [Interactive result processing RTX 3060](../benchmarks/interactive-result-processing-rtx3060-20260722/README.md) | Serving | **Infrastructure only** | GPU-backed result-processing measurement. |
| [Ladder batching](../benchmarks/ladder-batching-rtx4070s-20260906/README.md) | Evaluation throughput | **Retained** | Evaluation batching/concurrency evidence. |
| [Generation 936 ladder Elo](../benchmarks/ladder-elo-generation936-rtx4070s-20260906/README.md) | Evaluation | **Retained** | Calibrated ladder result for one mature checkpoint. |
| [Ladder Elo versus generation](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/README.md) | Evaluation/learning curve | **Retained** | Strength trajectory and ladder-fit methodology. |
| [Ladder reference config](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/reference-config/README.md) | Evaluation provenance | **Infrastructure only** | Preserves the reference configuration used by that ladder. |
| [Model refresh](../benchmarks/model-refresh-20260723/README.md) | Runtime | **Infrastructure only** | Checkpoint refresh mechanics. |
| [Naive Python MCTS](../benchmarks/naive-python-mcts-rtx3060-20260816/README.md) | Search reference | **Superseded** | Correctness/reference performance floor, not production search. |
| [8x4070S node comparison](../benchmarks/node-comparison-8xrtx4070super-20260824/README.md) | Hardware | **Infrastructure only** | Host/GPU comparison and transfer limits. |
| [Four-node comparison](../benchmarks/node-comparison-vast-4nodes-20260821/README.md) | Hardware | **Infrastructure only** | Rental-node selection evidence. |
| [Parallel searches batch-1600 rerun](../benchmarks/parallel-searches-rerun-batch1600-rtx4070s-20260906/README.md) | Search parallelism | **Retained** | Large-batch control for throughput/quality trade. |
| [Parallel searches sweep](../benchmarks/parallel-searches-sweep-rtx4070s-20260906/README.md) | Search parallelism | **Retained** | Primary final-era parallelism sweep. |
| [Progressive sizing throughput](../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md) | Model sizing | **Retained** | Direct evidence for the small-model early-throughput premise. |
| [Replay loader](../benchmarks/replay-loader-20260724/README.md) | Replay infrastructure | **Retained** | Loader throughput and storage-path control. |
| [Resignation canary](../benchmarks/resignation-audit-canary-20260723/README.md) | Resignation | **Infrastructure only** | Verifies audit persistence with deliberately aggressive threshold. |
| [Search throughput](../benchmarks/search-throughput-rtx4070-20260821/README.md) | Search parallelism | **Retained** | Early production-mix parallel-search throughput. |
| [C++ baseline 071550](../benchmarks/self-play-cpp-baseline-4x8x3x96-20260720T071550Z/README.md) | Native self-play | **Superseded** | First detailed native baseline. |
| [C++ baseline 073130](../benchmarks/self-play-cpp-baseline-4x8x3x96-20260720T073130Z/README.md) | Native self-play | **Superseded** | Corrected/follow-up baseline. |
| [C++ batching timeout](../benchmarks/self-play-cpp-batching-timeout5000us-20260720T073957Z/README.md) | Native batching | **Retained** | Timeout/batch-fill tuning evidence. |
| [C++ final tuning](../benchmarks/self-play-cpp-final-tuning-20260720/README.md) | Native self-play | **Retained** | Consolidated native search tuning result. |
| [Direct inference 4x4070S](../benchmarks/self-play-direct-inference-4x4070-super-20260722/README.md) | Native inference | **Retained** | Multi-GPU direct-inference production comparison. |
| [Direct inference RTX 3060](../benchmarks/self-play-direct-inference-rtx3060-20260722/README.md) | Native inference | **Retained** | Single-GPU direct-inference control. |
| [Graph multiworker](../benchmarks/self-play-graph-multiworker-8xrtx4070super-20260824/README.md) | Worker/process topology | **Retained** | Multiworker self-play graph/submission throughput; not MCGS evidence. |
| [MCTS node arena](../benchmarks/self-play-mcts-node-arena-20260720/README.md) | Native memory | **Retained** | Arena allocation optimization. |
| [Search CPU study](../benchmarks/self-play-search-cpu-i7-11370h-20260824/README.md) | CPU/batching | **Retained** | Shows inference/batch-fill limitation under a stand-in CPU workload. |
| [Submission 8x4070S](../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md) | Submission/runtime | **Retained** | Full-node submission and worker scaling evidence. |
| [Self-play throughput 4x3060](../benchmarks/self-play-throughput-4xrtx3060-20260809/README.md) | Self-play throughput | **Retained** | Earlier workload decomposition and batch-size limits. |
| [Self-play pause trade-off](../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md) | Actor/trainer overlap | **Retained** | Regime-dependent pause policy; supports, but does not universalize, final topology. |
| [Supervised testbed](../benchmarks/supervised-testbed-rtx4070-20260821/README.md) | Recovery screening | **Inconclusive** | Short trimmed supervised readings; document as diagnostic only. |
| [INT8 architecture screen](../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md) | Quantization architecture | **Retained** | Diagnoses graph bottlenecks and motivates scaled post-activation blocks. |
| [INT8 replay screen](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md) | QAT/fidelity | **Retained** | Principal frozen-replay QAT, fold, fidelity, refit, and throughput evidence. |
| [INT8 cadence control](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/cadence/README.md) | TensorRT refit | **Retained** | Measures recurring refit/update cost. |
| [INT8 failed variants](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/failures/README.md) | Evidence hygiene | **Implemented and rejected** | Prevents failed/superseded smokes from being mistaken for completed arms. |
| [INT8 salvage](../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md) | Quantization | **Implemented and rejected** | PTQ, partial QAT, SmoothQuant, and FP8 negative results. |
| [Native TensorRT backend](../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md) | TensorRT integration | **Retained** | Native refit/execution, FP16 control, and bounded INT8 game smoke. |
| [Chess throughput history](../benchmarks/throughput-history-chess-20260821/README.md) | Systems history | **Infrastructure only** | Cross-era overview; use only with workload/hardware caveats. |
| [V35-code/V42-g0 A/B](../benchmarks/v35-code-v42-generation0-controlled-ab-20260913/README.md) | Regression audit | **Infrastructure only** | Reproducible controlled-arm protocol; this README records setup rather than a completed result. |
| [V39 INT8 self-play](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md) | End-to-end throughput | **Retained** | Best decomposition from core INT8 speed to live admitted replay throughput. |
| [V76/V35 medium pre-fold backend](../benchmarks/v76-v35-medium-prefold-backend-20260918/README.md) | INT8 throughput | **Retained** | Production-topology 14x160 TensorRT advantage. |
| [V76/V35 small pre-fold backend](../benchmarks/v76-v35-small-prefold-backend-20260918/README.md) | INT8 throughput | **Retained** | Production-topology 12x128 TensorRT advantage and fidelity context. |

## Coverage limitations

This ledger guarantees discoverability, not equivalence. Several directories preserve microbenchmarks, smokes, or
diagnostics rather than efficacy experiments. Conversely, three important decisions do not have dedicated benchmark
directories and must be cited to their decision records:

- graph search and inference caching: [archived decision](../plan/archive/chess-four-day-baseline-and-next-run-plan.md#10-transposition-graph-search-and-inference-cache-decision---rejected);
- the final learned-stopping result: [adaptive-search conclusion](../analysis/adaptive-search-conclusion-20260904.md);
- the current final recipe: [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml).
