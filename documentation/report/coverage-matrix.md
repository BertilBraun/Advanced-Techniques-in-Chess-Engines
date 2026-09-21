# Research and artifact coverage matrix

This appendix makes omissions reviewable. It maps every substantive analysis record and every README in the
[benchmark coverage ledger](../experiments/benchmark-coverage.md) to a report destination. “Primary” means the
artifact supports narrative or a claim. “Supporting” means it supplies a control, smoke, provenance record, or
historical bound and need not be narrated independently.

Evidence grades are defined in [Chapter 2](02-methodology-and-evidence.md): **S** strength, **O** online learning,
**P** proxy, **T** throughput, **M** mechanics, and **R** rationale.

## Analysis records

| Analysis | Destination | Role/grade |
| --- | --- | --- |
| [Adaptive budget negative result](../analysis/adaptive-search-budget-negative-result-20260901.md) | [Search](04a-search.md#predicted-budgets) | Primary O/P/T |
| [Adaptive-search conclusion](../analysis/adaptive-search-conclusion-20260904.md) | [Search](04a-search.md#learned-early-stopping), [lineage](04d-lineage-and-decisions.md#search-campaign-v13v30) | Primary O/T/M |
| [Chess conversion investigation](../analysis/chess-conversion-investigation-20260826.md) | [Data](04b-data-and-replay.md#endgame-conversion-and-target-poisoning) | Primary O/P |
| [Elo scale and reporting](../analysis/chess-elo-scale-and-reporting-20260911.md) | [Methodology](02-methodology-and-evidence.md#strength-measurement) | Primary S/R |
| [Chess search findings](../analysis/chess-search-findings-20260827.md) | [Search](04a-search.md#fixed-depth-and-the-value-of-search) | Primary S/P/T |
| [Compute-poor reference recipes](../analysis/reference-recipes-for-a-compute-poor-run.md) | [Data](04b-data-and-replay.md), [model](04c-networks-and-training.md) | Primary R; external-transfer limits |
| [V35–V42 executable bisect](../analysis/v35-v42-executable-bisect-20260913.md) | [Model](04c-networks-and-training.md#bootstrap-calibration-and-deterministic-initialization), [lineage](04d-lineage-and-decisions.md#v35v42-why-short-runs-were-not-causal) | Primary M/P |
| [V35–V42 regression audit](../analysis/v35-v42-regression-audit-20260913.md) | Same as executable bisect | Primary M/P |
| [V8 training-data comparison](../analysis/v8-training-data-comparison-20260826.md) | [Data](04b-data-and-replay.md#endgame-conversion-and-target-poisoning) | Primary P |

## Benchmark records 1–20

| Benchmark record | Destination | Role/grade |
| --- | --- | --- |
| [Benchmark index](../benchmarks/README.md) | [Methodology](02-methodology-and-evidence.md#units-of-evidence) | Supporting navigation |
| [Adaptive budget frozen-trunk probe](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md) | [Search](04a-search.md#predicted-budgets) | Supporting P gate |
| [R3 adaptive termination audit](../benchmarks/adaptive-search-termination-r3-20260813/README.md) | [Search](04a-search.md#threshold-stopping-opportunity-audit-not-negative-elo-result) | Primary P/R |
| [Adaptive-search validation](../benchmarks/adaptive-search-validation-rtx4070s-20260818/README.md) | [Search](04a-search.md#threshold-stopping-opportunity-audit-not-negative-elo-result) | Supporting M; random-network validation only |
| [Architecture contention](../benchmarks/chess-architecture-contended-rtx3060-20260817/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting/inconclusive T/P |
| [Attention DDP](../benchmarks/chess-attention-ddp-rtx4070s-20260818/README.md) | [Systems](05-systems-optimization.md#training-throughput-and-replay-io) | Supporting T |
| [Packed QKV](../benchmarks/chess-attention-packed-qkv-rtx3060-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting, superseded T |
| [Attention SDPA RTX 3060](../benchmarks/chess-attention-sdpa-backends-rtx3060-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting correction T |
| [Attention SDPA RTX 4070S](../benchmarks/chess-attention-sdpa-backends-rtx4070s-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting T |
| [Attention training](../benchmarks/chess-attention-training-rtx4070s-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting T, not quality |
| [Attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Primary P/T |
| [Final progressive attention inference](../benchmarks/chess-direct-policy-final-progressive-rtx4070s-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting, superseded T |
| [Direct-policy inference](../benchmarks/chess-direct-policy-inference-rtx4070s-20260818/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Supporting compile control T |
| [Direct-policy kernel controls](../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary historical backend control T |
| [Teacher-imitation distillation](../benchmarks/chess-distillation-probe-rtx3060-20260827/README.md) | [Model](04c-networks-and-training.md#distillation-and-compression) | Primary P/T with defects |
| [Fixed-batch overfit](../benchmarks/chess-overfit-rtx3090-20260819/README.md) | [Model](04c-networks-and-training.md#auxiliary-objectives) | Supporting M/P only |
| [V34 replay distillation](../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) | [Model](04c-networks-and-training.md#distillation-and-compression) | Primary P/T/S bounded |
| [Chess search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md) | [Search](04a-search.md#fixed-depth-and-the-value-of-search) | Primary S/P/T |
| [Self-play latency](../benchmarks/chess-self-play-latency-rtx3060-20260812/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary T |
| [SGD post-fold LR](../benchmarks/chess-sgd-postfold-lr-rtx4070s-20260914/README.md) | [Model](04c-networks-and-training.md#adamw-sgd-and-learning-rate-evidence) | Primary P/M |

## Benchmark records 21–40

| Benchmark record | Destination | Role/grade |
| --- | --- | --- |
| [SGD pre-fold factorial](../benchmarks/chess-sgd-prefold-factorial-rtx4070s-20260914/README.md) | [Model](04c-networks-and-training.md#adamw-sgd-and-learning-rate-evidence) | Primary P/M |
| [SGD replay screen](../benchmarks/chess-sgd-replay-screen-rtx4070s-20260913/README.md) | [Model](04c-networks-and-training.md#adamw-sgd-and-learning-rate-evidence) | Primary P/M |
| [Stockfish ladder baseline](../benchmarks/chess-stockfish-ladder-8xrtx3060-20260816/README.md) | [Lineage](04d-lineage-and-decisions.md#baseline-platform-and-four-day-yardstick) | Supporting historical S/M |
| [TensorRT INT8](../benchmarks/chess-tensorrt-int8-rtx4070s-20260912/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary P/T |
| [Terminal v34](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md) | [Lineage](04d-lineage-and-decisions.md#v29-and-v34-strength-versus-wall-clock), [results](07-final-run-results.md) | Primary S baseline |
| [Training throughput](../benchmarks/chess-training-throughput-rtx3060-20260812/README.md) | [Systems](05-systems-optimization.md#training-throughput-and-replay-io) | Primary T |
| [V34 training dynamics](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md) | [Lineage](04d-lineage-and-decisions.md#v29-and-v34-strength-versus-wall-clock) | Primary O/T observational |
| [CNN inference throughput](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md) | [Systems](05-systems-optimization.md#model-shape-and-memory-format) | Primary T |
| [Credit runtime stage 6](../benchmarks/credit-runtime-stage6-20260724/README.md) | [Data](04b-data-and-replay.md#replay-infrastructure-as-evidence) | Supporting superseded M/T |
| [Credit runtime stage 7](../benchmarks/credit-runtime-stage7-20260724/README.md) | [Data](04b-data-and-replay.md#replay-infrastructure-as-evidence) | Supporting M/T |
| [Cut-game value target](../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md) | [Data](04b-data-and-replay.md#endgame-conversion-and-target-poisoning) | Primary P/M |
| [DDP model throughput](../benchmarks/ddp-model-throughput-20260720/README.md) | [Systems](05-systems-optimization.md#training-throughput-and-replay-io) | Supporting T |
| [DDP production training](../benchmarks/ddp-production-training-20260720/README.md) | [Systems](05-systems-optimization.md#training-throughput-and-replay-io) | Primary T/M |
| [Generation-936 deep match](../benchmarks/deep-match-generation936-50k-nodes-rtx4070s-20260906/README.md) | [System evaluation](03-system-and-methods.md#evaluation) | Primary S |
| [Direct inference RTX 3060](../benchmarks/direct-inference-rtx3060-20260722/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting superseded T |
| [Go 7x7 baseline](../benchmarks/go-7x7-training-baseline-2xrtx3060-20260810/README.md) | [Scope](01-motivation-and-scope.md#scope), [lineage](04d-lineage-and-decisions.md#baseline-platform-and-four-day-yardstick) | Supporting M/O, incomplete programme |
| [INT8 template staleness](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md) | [Systems](05-systems-optimization.md#tensorrt-refit-failure-and-fidelity-redesign) | Primary P/M and incident S |
| [Integrated interactive engine](../benchmarks/integrated-interactive-rtx3060-20260722/README.md) | [Systems](05-systems-optimization.md#evaluation-and-interactive-serving) | Supporting M/T |
| [Interactive engine local](../benchmarks/interactive-engine-local-20260721/README.md) | [Systems](05-systems-optimization.md#evaluation-and-interactive-serving) | Supporting M |
| [Interactive result processing](../benchmarks/interactive-result-processing-20260722/README.md) | [Systems](05-systems-optimization.md#evaluation-and-interactive-serving) | Supporting M/T |

## Benchmark records 41–60

| Benchmark record | Destination | Role/grade |
| --- | --- | --- |
| [Interactive result processing RTX 3060](../benchmarks/interactive-result-processing-rtx3060-20260722/README.md) | [Systems](05-systems-optimization.md#evaluation-and-interactive-serving) | Supporting M/T |
| [Ladder batching](../benchmarks/ladder-batching-rtx4070s-20260906/README.md) | [System evaluation](03-system-and-methods.md#evaluation) | Primary T/M/S protocol |
| [Generation-936 ladder Elo](../benchmarks/ladder-elo-generation936-rtx4070s-20260906/README.md) | [System evaluation](03-system-and-methods.md#evaluation) | Primary S |
| [Ladder Elo versus generation](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/README.md) | [Lineage](04d-lineage-and-decisions.md#v29-and-v34-strength-versus-wall-clock) | Primary S/O observational |
| [Ladder reference config](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/reference-config/README.md) | [Reproducibility](09-reproducibility.md#reproducing-evaluation) | Supporting provenance M |
| [Model refresh](../benchmarks/model-refresh-20260723/README.md) | [System failure semantics](03-system-and-methods.md#failure-and-restart-semantics) | Supporting M |
| [Naive Python MCTS](../benchmarks/naive-python-mcts-rtx3060-20260816/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting reference T |
| [8x4070S node comparison](../benchmarks/node-comparison-8xrtx4070super-20260824/README.md) | [Systems](05-systems-optimization.md#hardware-and-transfer-limits) | Supporting T |
| [Four-node comparison](../benchmarks/node-comparison-vast-4nodes-20260821/README.md) | [Systems](05-systems-optimization.md#hardware-and-transfer-limits) | Supporting T |
| [Parallel-search batch-1600 rerun](../benchmarks/parallel-searches-rerun-batch1600-rtx4070s-20260906/README.md) | [Search](04a-search.md#parallelism-and-batch-fill) | Primary S/T correction |
| [Parallel-search sweep](../benchmarks/parallel-searches-sweep-rtx4070s-20260906/README.md) | [Search](04a-search.md#parallelism-and-batch-fill) | Primary, qualified S/T |
| [Progressive-sizing throughput](../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md) | [Model](04c-networks-and-training.md#progressive-sizing) | Primary T |
| [Replay loader](../benchmarks/replay-loader-20260724/README.md) | [Data](04b-data-and-replay.md#replay-infrastructure-as-evidence) | Supporting T/M |
| [Resignation canary](../benchmarks/resignation-audit-canary-20260723/README.md) | [Data](04b-data-and-replay.md#calibrated-resignation) | Supporting M only |
| [Search throughput](../benchmarks/search-throughput-rtx4070-20260821/README.md) | [Search](04a-search.md#parallelism-and-batch-fill) | Supporting T |
| [C++ baseline 071550](../benchmarks/self-play-cpp-baseline-4x8x3x96-20260720T071550Z/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting historical T |
| [C++ baseline 073130](../benchmarks/self-play-cpp-baseline-4x8x3x96-20260720T073130Z/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting corrected T |
| [C++ batching timeout](../benchmarks/self-play-cpp-batching-timeout5000us-20260720T073957Z/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Supporting T |
| [C++ final tuning](../benchmarks/self-play-cpp-final-tuning-20260720/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Primary historical T/M |
| [Direct inference 4x4070S](../benchmarks/self-play-direct-inference-4x4070-super-20260722/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Primary T |

## Benchmark records 61–79

| Benchmark record | Destination | Role/grade |
| --- | --- | --- |
| [Direct inference RTX 3060](../benchmarks/self-play-direct-inference-rtx3060-20260722/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Primary T |
| [Graph multiworker](../benchmarks/self-play-graph-multiworker-8xrtx4070super-20260824/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Supporting T; not graph-search evidence |
| [MCTS node arena](../benchmarks/self-play-mcts-node-arena-20260720/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting T/M |
| [Search CPU study](../benchmarks/self-play-search-cpu-i7-11370h-20260824/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary T; stand-in CPU caveat |
| [Submission 8x4070S](../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary T |
| [Self-play throughput 4x3060](../benchmarks/self-play-throughput-4xrtx3060-20260809/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Supporting historical T |
| [Self-play pause trade-off](../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md) | [Data](04b-data-and-replay.md#overlap-is-not-full-asynchrony) | Primary T/O; regime-specific |
| [Supervised testbed](../benchmarks/supervised-testbed-rtx4070-20260821/README.md) | [Lineage](04d-lineage-and-decisions.md#recovery-and-the-v7v8-conversion-failure) | Supporting/inconclusive P |
| [INT8 architecture screen](../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary P/T |
| [INT8 replay screen](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md) | [Model](04c-networks-and-training.md#quantization-friendly-residual-blocks), [systems](05-systems-optimization.md#the-pre-fold-serving-design) | Primary P/T/M |
| [INT8 cadence control](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/cadence/README.md) | [Systems](05-systems-optimization.md#tensorrt-refit-failure-and-fidelity-redesign) | Supporting T/M |
| [INT8 failed variants](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/failures/README.md) | [Methodology](02-methodology-and-evidence.md#corrections-and-supersession) | Supporting evidence hygiene |
| [INT8 salvage](../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary negative P/T |
| [Native TensorRT backend](../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary T/M |
| [Chess throughput history](../benchmarks/throughput-history-chess-20260821/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting chronology, not controlled curve |
| [V35-code/V42-g0 A/B](../benchmarks/v35-code-v42-generation0-controlled-ab-20260913/README.md) | [Lineage](04d-lineage-and-decisions.md#v35v42-why-short-runs-were-not-causal) | Supporting protocol M, no efficacy result |
| [V39 INT8 self-play](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary T decomposition |
| [V76/V35 medium pre-fold backend](../benchmarks/v76-v35-medium-prefold-backend-20260918/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary T for 14x160 |
| [V76/V35 small pre-fold backend](../benchmarks/v76-v35-small-prefold-backend-20260918/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary T/P for 12x128 |

## Coverage conclusion

All nine substantive analysis records and all 79 benchmark README records have a destination above. This does not
make every record equally important. Supporting smokes and superseded controls remain discoverable without inflating
them into narrative results; primary artifacts carry the report's claims. Architecture, plan, and history records
are cited in the relevant topic chapters when they are the only evidence for an implemented decision, such as graph
search, or when they define current ownership and invariants.
