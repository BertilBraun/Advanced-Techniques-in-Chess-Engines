# Research and artifact coverage matrix

This appendix makes omissions reviewable. It maps every substantive analysis record and every README in the
[benchmark coverage ledger](../experiments/benchmark-coverage.md) to a report destination. “Primary” means the
artifact supports narrative or a claim. “Supporting” means it supplies a control, smoke, provenance record, or
historical bound and need not be narrated independently.

Evidence dimensions are defined in [Chapter 2](02-methodology-and-evidence.md). They are written out here rather
than encoded as letter grades.

## Analysis records

| Analysis | Destination | Role/grade |
| --- | --- | --- |
| [Adaptive budget negative result](../analysis/adaptive-search-budget-negative-result-20260901.md) | [Search](04a-search.md#predicted-budgets) | Primary online-learning, proxy, and throughput evidence |
| [Adaptive-search conclusion](../analysis/adaptive-search-conclusion-20260904.md) | [Search](04a-search.md#learned-early-stopping), [decision provenance](04d-lineage-and-decisions.md#adaptive-search-controller-audit) | Primary online-learning, throughput, and mechanics evidence |
| [Chess conversion investigation](../analysis/chess-conversion-investigation-20260826.md) | [Data](04b-data-and-replay.md#endgame-conversion-and-target-poisoning) | Primary observational and proxy evidence |
| [Elo scale and reporting](../analysis/chess-elo-scale-and-reporting-20260911.md) | [Methodology](02-methodology-and-evidence.md#strength-measurement) | Primary strength evidence and reporting rationale |
| [Chess search findings](../analysis/chess-search-findings-20260827.md) | [Search](04a-search.md#fixed-depth-and-the-value-of-search) | Primary strength, proxy, and throughput evidence |
| [Compute-poor reference recipes](../analysis/reference-recipes-for-a-compute-poor-run.md) | [Data](04b-data-and-replay.md), [model](04c-networks-and-training.md) | Primary design rationale with external-transfer limits |
| [Controlled initialization bisect](../analysis/v35-v42-executable-bisect-20260913.md) | [Model](04c-networks-and-training.md#bootstrap-calibration-and-deterministic-initialization), [decision provenance](04d-lineage-and-decisions.md#deterministic-initialization-and-causal-hygiene) | Primary mechanics and proxy evidence |
| [Initialization regression audit](../analysis/v35-v42-regression-audit-20260913.md) | Same as executable bisect | Primary mechanics and proxy evidence |
| [Post-rework training-data comparison](../analysis/v8-training-data-comparison-20260826.md) | [Data](04b-data-and-replay.md#endgame-conversion-and-target-poisoning) | Primary proxy evidence |

## Benchmark records 1–20

| Benchmark record | Destination | Role/grade |
| --- | --- | --- |
| [Benchmark index](../benchmarks/README.md) | [Methodology](02-methodology-and-evidence.md#units-of-evidence) | Supporting navigation |
| [Adaptive budget frozen-trunk probe](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md) | [Search](04a-search.md#predicted-budgets) | Supporting proxy gate |
| [Adaptive termination opportunity audit](../benchmarks/adaptive-search-termination-r3-20260813/README.md) | [Search](04a-search.md#threshold-stopping-opportunity-audit-not-negative-elo-result) | Primary proxy evidence and rationale |
| [Adaptive-search validation](../benchmarks/adaptive-search-validation-rtx4070s-20260818/README.md) | [Search](04a-search.md#threshold-stopping-opportunity-audit-not-negative-elo-result) | Supporting mechanics; random-network validation only |
| [Architecture contention](../benchmarks/chess-architecture-contended-rtx3060-20260817/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting, inconclusive throughput and proxy evidence |
| [Attention DDP](../benchmarks/chess-attention-ddp-rtx4070s-20260818/README.md) | [Systems](05-systems-optimization.md#training-throughput-and-replay-io) | Supporting throughput evidence |
| [Packed QKV](../benchmarks/chess-attention-packed-qkv-rtx3060-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting, superseded throughput evidence |
| [Attention SDPA RTX 3060](../benchmarks/chess-attention-sdpa-backends-rtx3060-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting throughput correction |
| [Attention SDPA RTX 4070S](../benchmarks/chess-attention-sdpa-backends-rtx4070s-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting throughput evidence |
| [Attention training](../benchmarks/chess-attention-training-rtx4070s-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting throughput evidence, not quality |
| [Attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Primary proxy and throughput evidence |
| [Final progressive attention inference](../benchmarks/chess-direct-policy-final-progressive-rtx4070s-20260818/README.md) | [Model](04c-networks-and-training.md#cnn-versus-attention) | Supporting, superseded throughput evidence |
| [Direct-policy inference](../benchmarks/chess-direct-policy-inference-rtx4070s-20260818/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Supporting compilation throughput control |
| [Direct-policy kernel controls](../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary historical backend-throughput control |
| [Teacher-imitation distillation](../benchmarks/chess-distillation-probe-rtx3060-20260827/README.md) | [Model](04c-networks-and-training.md#distillation-and-compression) | Primary proxy and throughput evidence with defects |
| [Fixed-batch overfit](../benchmarks/chess-overfit-rtx3090-20260819/README.md) | [Model](04c-networks-and-training.md#auxiliary-objectives) | Supporting mechanics and proxy evidence only |
| [Previous-baseline replay distillation](../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) | [Model](04c-networks-and-training.md#distillation-and-compression) | Primary bounded proxy, throughput, and strength evidence |
| [Chess search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md) | [Search](04a-search.md#fixed-depth-and-the-value-of-search) | Primary strength, proxy, and throughput evidence |
| [Self-play latency](../benchmarks/chess-self-play-latency-rtx3060-20260812/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary throughput evidence |
| [SGD post-fold learning rate](../benchmarks/chess-sgd-postfold-lr-rtx4070s-20260914/README.md) | [Model](04c-networks-and-training.md#adamw-sgd-and-learning-rate-evidence) | Primary proxy and mechanics evidence |

## Benchmark records 21–40

| Benchmark record | Destination | Role/grade |
| --- | --- | --- |
| [SGD pre-fold factorial](../benchmarks/chess-sgd-prefold-factorial-rtx4070s-20260914/README.md) | [Model](04c-networks-and-training.md#adamw-sgd-and-learning-rate-evidence) | Primary proxy and mechanics |
| [SGD replay screen](../benchmarks/chess-sgd-replay-screen-rtx4070s-20260913/README.md) | [Model](04c-networks-and-training.md#adamw-sgd-and-learning-rate-evidence) | Primary proxy and mechanics |
| [Stockfish ladder baseline](../benchmarks/chess-stockfish-ladder-8xrtx3060-20260816/README.md) | [Lineage](04d-lineage-and-decisions.md#baseline-platform-and-wall-clock-yardstick) | Supporting historical strength and mechanics |
| [TensorRT INT8](../benchmarks/chess-tensorrt-int8-rtx4070s-20260912/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary proxy and throughput |
| [Previous-baseline terminal evaluation](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md) | [Lineage](04d-lineage-and-decisions.md#wall-clock-strength-and-data-throughput), [results](07-final-run-results.md) | Primary strength baseline |
| [Training throughput](../benchmarks/chess-training-throughput-rtx3060-20260812/README.md) | [Systems](05-systems-optimization.md#training-throughput-and-replay-io) | Primary throughput |
| [Previous-baseline training dynamics](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md) | [Lineage](04d-lineage-and-decisions.md#wall-clock-strength-and-data-throughput) | Primary online-learning and throughput observational |
| [CNN inference throughput](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md) | [Systems](05-systems-optimization.md#model-shape-and-memory-format) | Primary throughput |
| [Credit runtime stage 6](../benchmarks/credit-runtime-stage6-20260724/README.md) | [Data](04b-data-and-replay.md#replay-infrastructure-as-evidence) | Supporting superseded mechanics and throughput |
| [Credit runtime stage 7](../benchmarks/credit-runtime-stage7-20260724/README.md) | [Data](04b-data-and-replay.md#replay-infrastructure-as-evidence) | Supporting mechanics and throughput |
| [Cut-game value target](../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md) | [Data](04b-data-and-replay.md#endgame-conversion-and-target-poisoning) | Primary proxy and mechanics |
| [DDP model throughput](../benchmarks/ddp-model-throughput-20260720/README.md) | [Systems](05-systems-optimization.md#training-throughput-and-replay-io) | Supporting throughput |
| [DDP production training](../benchmarks/ddp-production-training-20260720/README.md) | [Systems](05-systems-optimization.md#training-throughput-and-replay-io) | Primary throughput and mechanics |
| [Generation-936 deep match](../benchmarks/deep-match-generation936-50k-nodes-rtx4070s-20260906/README.md) | [System evaluation](03-system-and-methods.md#evaluation) | Primary strength |
| [Direct inference RTX 3060](../benchmarks/direct-inference-rtx3060-20260722/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting superseded throughput |
| [Go 7x7 baseline](../benchmarks/go-7x7-training-baseline-2xrtx3060-20260810/README.md) | [Scope](01-motivation-and-scope.md#scope), [lineage](04d-lineage-and-decisions.md#baseline-platform-and-wall-clock-yardstick) | Supporting mechanics and observational online-learning, incomplete programme |
| [INT8 template staleness](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md) | [Systems](05-systems-optimization.md#tensorrt-refit-failure-and-fidelity-redesign) | Primary proxy and mechanics, plus incident strength |
| [Integrated interactive engine](../benchmarks/integrated-interactive-rtx3060-20260722/README.md) | [Systems](05-systems-optimization.md#evaluation-and-interactive-serving) | Supporting mechanics and throughput |
| [Interactive engine local](../benchmarks/interactive-engine-local-20260721/README.md) | [Systems](05-systems-optimization.md#evaluation-and-interactive-serving) | Supporting mechanics |
| [Interactive result processing](../benchmarks/interactive-result-processing-20260722/README.md) | [Systems](05-systems-optimization.md#evaluation-and-interactive-serving) | Supporting mechanics and throughput |

## Benchmark records 41–60

| Benchmark record | Destination | Role/grade |
| --- | --- | --- |
| [Interactive result processing RTX 3060](../benchmarks/interactive-result-processing-rtx3060-20260722/README.md) | [Systems](05-systems-optimization.md#evaluation-and-interactive-serving) | Supporting mechanics and throughput |
| [Ladder batching](../benchmarks/ladder-batching-rtx4070s-20260906/README.md) | [System evaluation](03-system-and-methods.md#evaluation) | Primary throughput, mechanics, and strength protocol |
| [Generation-936 ladder Elo](../benchmarks/ladder-elo-generation936-rtx4070s-20260906/README.md) | [System evaluation](03-system-and-methods.md#evaluation) | Primary strength |
| [Ladder Elo versus generation](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/README.md) | [Lineage](04d-lineage-and-decisions.md#wall-clock-strength-and-data-throughput) | Primary strength and observational online-learning observational |
| [Ladder reference config](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/reference-config/README.md) | [Reproducibility](09-reproducibility.md#reproducing-evaluation) | Supporting provenance mechanics |
| [Model refresh](../benchmarks/model-refresh-20260723/README.md) | [System failure semantics](03-system-and-methods.md#failure-and-restart-semantics) | Supporting mechanics |
| [Naive Python MCTS](../benchmarks/naive-python-mcts-rtx3060-20260816/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting reference throughput |
| [8x4070S node comparison](../benchmarks/node-comparison-8xrtx4070super-20260824/README.md) | [Systems](05-systems-optimization.md#hardware-and-transfer-limits) | Supporting throughput |
| [Four-node comparison](../benchmarks/node-comparison-vast-4nodes-20260821/README.md) | [Systems](05-systems-optimization.md#hardware-and-transfer-limits) | Supporting throughput |
| [Parallel-search batch-1600 rerun](../benchmarks/parallel-searches-rerun-batch1600-rtx4070s-20260906/README.md) | [Search](04a-search.md#parallelism-and-batch-fill) | Primary strength and throughput correction |
| [Parallel-search sweep](../benchmarks/parallel-searches-sweep-rtx4070s-20260906/README.md) | [Search](04a-search.md#parallelism-and-batch-fill) | Primary, qualified strength and throughput |
| [Progressive-sizing throughput](../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md) | [Model](04c-networks-and-training.md#progressive-sizing) | Primary throughput |
| [Replay loader](../benchmarks/replay-loader-20260724/README.md) | [Data](04b-data-and-replay.md#replay-infrastructure-as-evidence) | Supporting throughput and mechanics |
| [Resignation canary](../benchmarks/resignation-audit-canary-20260723/README.md) | [Data](04b-data-and-replay.md#calibrated-resignation) | Supporting mechanics only |
| [Search throughput](../benchmarks/search-throughput-rtx4070-20260821/README.md) | [Search](04a-search.md#parallelism-and-batch-fill) | Supporting throughput |
| [C++ baseline 071550](../benchmarks/self-play-cpp-baseline-4x8x3x96-20260720T071550Z/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting historical throughput |
| [C++ baseline 073130](../benchmarks/self-play-cpp-baseline-4x8x3x96-20260720T073130Z/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting corrected throughput |
| [C++ batching timeout](../benchmarks/self-play-cpp-batching-timeout5000us-20260720T073957Z/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Supporting throughput |
| [C++ final tuning](../benchmarks/self-play-cpp-final-tuning-20260720/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Primary historical throughput and mechanics |
| [Direct inference 4x4070S](../benchmarks/self-play-direct-inference-4x4070-super-20260722/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Primary throughput |

## Benchmark records 61–79

| Benchmark record | Destination | Role/grade |
| --- | --- | --- |
| [Direct inference RTX 3060](../benchmarks/self-play-direct-inference-rtx3060-20260722/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Primary throughput |
| [Graph multiworker](../benchmarks/self-play-graph-multiworker-8xrtx4070super-20260824/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Supporting throughput; not graph-search evidence |
| [MCTS node arena](../benchmarks/self-play-mcts-node-arena-20260720/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting throughput and mechanics |
| [Search CPU study](../benchmarks/self-play-search-cpu-i7-11370h-20260824/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary throughput; stand-in CPU caveat |
| [Submission 8x4070S](../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary throughput |
| [Self-play throughput 4x3060](../benchmarks/self-play-throughput-4xrtx3060-20260809/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Supporting historical throughput |
| [Self-play pause trade-off](../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md) | [Data](04b-data-and-replay.md#overlap-is-not-full-asynchrony) | Primary throughput and observational online-learning; regime-specific |
| [Supervised testbed](../benchmarks/supervised-testbed-rtx4070-20260821/README.md) | [Lineage](04d-lineage-and-decisions.md#endgame-conversion-and-target-stream-audit) | Supporting/inconclusive proxy |
| [INT8 architecture screen](../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary proxy and throughput |
| [INT8 replay screen](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md) | [Model](04c-networks-and-training.md#quantization-friendly-residual-blocks), [systems](05-systems-optimization.md#the-pre-fold-serving-design) | Primary proxy, throughput, and mechanics |
| [INT8 cadence control](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/cadence/README.md) | [Systems](05-systems-optimization.md#tensorrt-refit-failure-and-fidelity-redesign) | Supporting throughput and mechanics |
| [INT8 failed variants](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/failures/README.md) | [Methodology](02-methodology-and-evidence.md#corrections-and-supersession) | Supporting evidence hygiene |
| [INT8 salvage](../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary negative proxy and throughput |
| [Native TensorRT backend](../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md) | [Systems](05-systems-optimization.md#torchscript-tensorrt-and-precision) | Primary throughput and mechanics |
| [Chess throughput history](../benchmarks/throughput-history-chess-20260821/README.md) | [Systems](05-systems-optimization.md#native-search-and-direct-inference) | Supporting chronology, not controlled curve |
| [Controlled initialization comparison](../benchmarks/v35-code-v42-generation0-controlled-ab-20260913/README.md) | [Lineage](04d-lineage-and-decisions.md#deterministic-initialization-and-causal-hygiene) | Supporting protocol mechanics, no efficacy result |
| [INT8 self-play topology decomposition](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary throughput decomposition |
| [Medium-model pre-fold backend](../benchmarks/v76-v35-medium-prefold-backend-20260918/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary throughput for 14x160 |
| [Small-model pre-fold backend](../benchmarks/v76-v35-small-prefold-backend-20260918/README.md) | [Systems](05-systems-optimization.md#batching-and-submission-cost) | Primary throughput and proxy for 12x128 |

## Coverage conclusion

All nine substantive analysis records and all 79 benchmark README records have a destination above. This does not
make every record equally important. Supporting smokes and superseded controls remain discoverable without inflating
them into narrative results; primary artifacts carry the report's claims. Architecture, plan, and history records
are cited in the relevant topic chapters when they are the only evidence for an implemented decision, such as graph
search, or when they define current ownership and invariants.
