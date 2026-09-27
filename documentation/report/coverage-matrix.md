# Research and artifact coverage matrix

This appendix makes omissions reviewable. It maps every substantive analysis record and every README in the
[benchmark coverage ledger](../experiments/benchmark-coverage.md) to a report destination. “Primary” means the
artifact supports narrative or a claim. “Supporting” means it supplies a control, smoke, provenance record, or
historical bound and need not be narrated independently.

Evidence dimensions are defined in [Chapter 2](02-methodology-and-evidence.md). They are written out here rather
than encoded as letter grades.

## Analysis records

| Analysis | Destination | Role/evidence |
| --- | --- | --- |
| [Adaptive budget negative result](../analysis/adaptive-search-budget-negative-result-20260901.md) | [Search allocation](04a-search.md#three-attempts-to-allocate-search-adaptively) | Primary online-learning, proxy, and throughput evidence |
| [Adaptive-search conclusion](../analysis/adaptive-search-conclusion-20260904.md) | [Search allocation](04a-search.md#three-attempts-to-allocate-search-adaptively) | Primary online-learning, throughput, and mechanics evidence |
| [Chess conversion investigation](../analysis/chess-conversion-investigation-20260826.md) | [Game termination](04b-data-and-replay.md#outcome-targets-at-the-ply-cap) | Primary observational and proxy evidence |
| [Elo scale and reporting](../analysis/chess-elo-scale-and-reporting-20260911.md) | [Playing strength](02-methodology-and-evidence.md#final-playing-strength) | Primary strength evidence and reporting rationale |
| [Chess search findings](../analysis/chess-search-findings-20260827.md) | [Fixed-budget search](04a-search.md#the-fixed-budget-baseline) | Primary strength, proxy, and throughput evidence |
| [Compute-poor reference recipes](../analysis/reference-recipes-for-a-compute-poor-run.md) | [Data curriculum](04b-data-and-replay.md), [Network and training](04c-networks-and-training.md) | Primary design rationale with external-transfer limits |
| [Controlled initialization bisect](../analysis/v35-v42-executable-bisect-20260913.md) | [Optimization controls](04c-networks-and-training.md#bootstrap-and-optimization-controls) | Primary mechanics and proxy evidence |
| [Initialization regression audit](../analysis/v35-v42-regression-audit-20260913.md) | [Optimization controls](04c-networks-and-training.md#bootstrap-and-optimization-controls) | Primary mechanics and proxy evidence |
| [Late-game training-data comparison](../analysis/v8-training-data-comparison-20260826.md) | [Game termination](04b-data-and-replay.md#outcome-targets-at-the-ply-cap) | Primary proxy evidence |

## Benchmark records 1–20

| Benchmark record | Destination | Role/evidence |
| --- | --- | --- |
| [Benchmark index](../benchmarks/README.md) | [Provenance](02-methodology-and-evidence.md#reading-the-component-experiments) | Supporting navigation |
| [Adaptive budget frozen-trunk probe](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md) | [Search allocation](04a-search.md#three-attempts-to-allocate-search-adaptively) | Supporting proxy gate |
| [Adaptive termination opportunity audit](../benchmarks/adaptive-search-termination-r3-20260813/README.md) | [Search allocation](04a-search.md#three-attempts-to-allocate-search-adaptively) | Primary proxy evidence and rationale |
| [Adaptive-search validation](../benchmarks/adaptive-search-validation-rtx4070s-20260818/README.md) | [Search allocation](04a-search.md#three-attempts-to-allocate-search-adaptively) | Supporting mechanics; random-network validation only |
| [Architecture contention](../benchmarks/chess-architecture-contended-rtx3060-20260817/README.md) | [Network architecture](04c-networks-and-training.md#convolution-attention-and-global-context) | Supporting, inconclusive throughput and proxy evidence |
| [Attention DDP](../benchmarks/chess-attention-ddp-rtx4070s-20260818/README.md) | [Training supply](05-systems-optimization.md#replay-materialization-and-training-supply) | Supporting throughput evidence |
| [Packed QKV](../benchmarks/chess-attention-packed-qkv-rtx3060-20260818/README.md) | [Network architecture](04c-networks-and-training.md#convolution-attention-and-global-context) | Supporting, superseded throughput evidence |
| [Attention SDPA RTX 3060](../benchmarks/chess-attention-sdpa-backends-rtx3060-20260818/README.md) | [Network architecture](04c-networks-and-training.md#convolution-attention-and-global-context) | Supporting throughput correction |
| [Attention SDPA RTX 4070S](../benchmarks/chess-attention-sdpa-backends-rtx4070s-20260818/README.md) | [Network architecture](04c-networks-and-training.md#convolution-attention-and-global-context) | Supporting throughput evidence |
| [Attention training](../benchmarks/chess-attention-training-rtx4070s-20260818/README.md) | [Network architecture](04c-networks-and-training.md#convolution-attention-and-global-context) | Supporting throughput evidence, not quality |
| [Attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md) | [Network architecture](04c-networks-and-training.md#convolution-attention-and-global-context) | Primary proxy and throughput evidence |
| [Progressive attention inference](../benchmarks/chess-direct-policy-final-progressive-rtx4070s-20260818/README.md) | [Network architecture](04c-networks-and-training.md#convolution-attention-and-global-context) | Supporting, superseded throughput evidence |
| [Direct-policy inference](../benchmarks/chess-direct-policy-inference-rtx4070s-20260818/README.md) | [Inference runtime](05-systems-optimization.md#inference-runtimes-and-precision) | Supporting compilation throughput control |
| [Direct-policy kernel controls](../benchmarks/chess-direct-policy-kernel-controls-rtx4070s-20260818/README.md) | [Inference runtime](05-systems-optimization.md#inference-runtimes-and-precision) | Primary historical backend-throughput control |
| [Corrected CNN width and depth/batch sweeps](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md#trunk-width-against-throughput-and-why-136-was-a-bad-choice) | [Architecture throughput tables](appendix-c-supporting-comparisons.md#architecture-shape-and-inference-throughput) | C1 normalizes published rates to 103,490 positions/s and compares with inverse width squared; C2 retains the transcribed batch-specific ratios |
| [Teacher-imitation distillation](../benchmarks/chess-distillation-probe-rtx3060-20260827/README.md) | [Compact models](04c-networks-and-training.md#distillation-and-compact-models) | Primary proxy and throughput evidence with defects |
| [Fixed-batch overfit](../benchmarks/chess-overfit-rtx3090-20260819/README.md) | [Training-only heads](04c-networks-and-training.md#value-and-training-only-heads) | Supporting mechanics and proxy evidence only |
| [Previous-baseline replay distillation](../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) | [Compact models](04c-networks-and-training.md#distillation-and-compact-models) | Primary bounded proxy, throughput, and strength evidence |
| [Chess search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md) | [Fixed-budget search](04a-search.md#the-fixed-budget-baseline) | Primary strength, proxy, and throughput evidence |
| [Self-play latency](../benchmarks/chess-self-play-latency-rtx3060-20260812/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Primary throughput evidence |
| [SGD post-fold learning rate](../benchmarks/chess-sgd-postfold-lr-rtx4070s-20260914/README.md) | [Optimization controls](04c-networks-and-training.md#bootstrap-and-optimization-controls) | Primary proxy and mechanics evidence |

## Benchmark records 21–40

| Benchmark record | Destination | Role/evidence |
| --- | --- | --- |
| [SGD pre-fold factorial](../benchmarks/chess-sgd-prefold-factorial-rtx4070s-20260914/README.md) | [Optimization controls](04c-networks-and-training.md#bootstrap-and-optimization-controls) | Primary proxy and mechanics |
| [SGD replay screen](../benchmarks/chess-sgd-replay-screen-rtx4070s-20260913/README.md) | [Optimization controls](04c-networks-and-training.md#bootstrap-and-optimization-controls) | Primary proxy and mechanics |
| [Stockfish ladder baseline](../benchmarks/chess-stockfish-ladder-8xrtx3060-20260816/README.md) | [Playing strength](02-methodology-and-evidence.md#final-playing-strength) | Supporting historical strength and mechanics |
| [TensorRT INT8](../benchmarks/chess-tensorrt-int8-rtx4070s-20260912/README.md) | [Inference runtime](05-systems-optimization.md#inference-runtimes-and-precision) | Primary proxy and throughput |
| [Previous-baseline terminal evaluation](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md) | [Training outcome](06-final-chess-recipe.md#progress-across-training-campaigns) | Primary strength baseline |
| [Training throughput](../benchmarks/chess-training-throughput-rtx3060-20260812/README.md) | [Training supply](05-systems-optimization.md#replay-materialization-and-training-supply) | Primary throughput |
| [Previous-baseline training dynamics](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md) | [Replay memory and pacing](04b-data-and-replay.md#replay-capacity-and-reuse) | Primary observational online-learning and throughput evidence |
| [CNN inference throughput](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md) | [Inference runtime](05-systems-optimization.md#inference-runtimes-and-precision) | Primary throughput |
| [Superseded replay-credit runtime](../benchmarks/credit-runtime-stage6-20260724/README.md) | [Data integrity](05-systems-optimization.md#replay-materialization-and-training-supply) | Supporting superseded mechanics and throughput |
| [Replay-credit runtime](../benchmarks/credit-runtime-stage7-20260724/README.md) | [Data integrity](05-systems-optimization.md#replay-materialization-and-training-supply) | Supporting mechanics and throughput |
| [Cut-game value target](../benchmarks/cut-game-value-target-rtx4070super-20260825/README.md) | [Game termination](04b-data-and-replay.md#outcome-targets-at-the-ply-cap) | Primary proxy and mechanics |
| [DDP model throughput](../benchmarks/ddp-model-throughput-20260720/README.md) | [Training supply](05-systems-optimization.md#replay-materialization-and-training-supply) | Supporting throughput |
| [DDP production training](../benchmarks/ddp-production-training-20260720/README.md) | [Training supply](05-systems-optimization.md#replay-materialization-and-training-supply) | Primary throughput and mechanics |
| [Deep-search strength match](../benchmarks/deep-match-generation936-50k-nodes-rtx4070s-20260906/README.md) | [Evaluation](03-system-and-methods.md#evaluation) | Primary strength |
| [Direct inference RTX 3060](../benchmarks/direct-inference-rtx3060-20260722/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting superseded throughput |
| [Go 7x7 baseline](../benchmarks/go-7x7-training-baseline-2xrtx3060-20260810/README.md) | [Scope](01-motivation-and-scope.md#scope-and-contributions) | Supporting mechanics and observational online-learning, incomplete programme |
| [INT8 template staleness](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md) | [TensorRT refit failure](05a-three-failures.md#prediction-drift-under-tensorrt-refitting) | Primary proxy and mechanics, plus incident strength |
| [Integrated interactive engine](../benchmarks/integrated-interactive-rtx3060-20260722/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting mechanics and throughput |
| [Interactive engine local](../benchmarks/interactive-engine-local-20260721/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting mechanics |
| [Interactive result processing](../benchmarks/interactive-result-processing-20260722/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting mechanics and throughput |

## Benchmark records 41–60

| Benchmark record | Destination | Role/evidence |
| --- | --- | --- |
| [Interactive result processing RTX 3060](../benchmarks/interactive-result-processing-rtx3060-20260722/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting mechanics and throughput |
| [Ladder batching](../benchmarks/ladder-batching-rtx4070s-20260906/README.md) | [Evaluation](03-system-and-methods.md#evaluation) | Primary throughput, mechanics, and strength-protocol evidence |
| [Ladder strength evaluation](../benchmarks/ladder-elo-generation936-rtx4070s-20260906/README.md) | [Evaluation](03-system-and-methods.md#evaluation) | Primary strength evidence |
| [Ladder learning curve](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/README.md) | [Training outcome](06-final-chess-recipe.md#progress-across-training-campaigns) | Primary strength and observational online-learning evidence |
| [Ladder reference config](../benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/reference-config/README.md) | [Reproducibility](appendix-d-reproducibility.md#reproducing-evaluation) | Supporting provenance mechanics |
| [Model refresh](../benchmarks/model-refresh-20260723/README.md) | [Model publication](03-system-and-methods.md#progressive-models-and-publication) | Supporting mechanics |
| [Naive Python MCTS](../benchmarks/naive-python-mcts-rtx3060-20260816/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting reference throughput |
| [8x4070S node comparison](../benchmarks/node-comparison-8xrtx4070super-20260824/README.md) | [End-to-end throughput](05-systems-optimization.md#throughput-allocation-within-the-learning-loop) | Supporting throughput |
| [Four-node comparison](../benchmarks/node-comparison-vast-4nodes-20260821/README.md) | [End-to-end throughput](05-systems-optimization.md#throughput-allocation-within-the-learning-loop) | Supporting throughput |
| [Parallel-search oversized-batch rerun](../benchmarks/parallel-searches-rerun-batch1600-rtx4070s-20260906/README.md) | [Parallel search](04a-search.md#parallel-leaves-buying-latency-with-search-quality) | Primary strength and throughput correction |
| [Parallel-search sweep](../benchmarks/parallel-searches-sweep-rtx4070s-20260906/README.md) | [Parallel search](04a-search.md#parallel-leaves-buying-latency-with-search-quality) | Primary, qualified strength and throughput evidence |
| [Progressive-sizing throughput](../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md) | [Progressive model sizing](04c-networks-and-training.md#progressive-model-sizing) | Primary throughput |
| [Replay loader](../benchmarks/replay-loader-20260724/README.md) | [Data integrity](05-systems-optimization.md#replay-materialization-and-training-supply) | Supporting throughput and mechanics |
| [Resignation canary](../benchmarks/resignation-audit-canary-20260723/README.md) | [Resignation calibration](04b-data-and-replay.md#resignation-calibration) | Supporting mechanics only |
| [Search throughput](../benchmarks/search-throughput-rtx4070-20260821/README.md) | [Parallel search](04a-search.md#parallel-leaves-buying-latency-with-search-quality) | Supporting throughput |
| [Initial C++ self-play baseline](../benchmarks/self-play-cpp-baseline-4x8x3x96-20260720T071550Z/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting historical throughput |
| [Corrected C++ self-play baseline](../benchmarks/self-play-cpp-baseline-4x8x3x96-20260720T073130Z/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting corrected throughput |
| [C++ batching timeout](../benchmarks/self-play-cpp-batching-timeout5000us-20260720T073957Z/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Supporting throughput |
| [C++ self-play tuning](../benchmarks/self-play-cpp-final-tuning-20260720/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Primary historical throughput and mechanics |
| [Direct inference 4x4070S](../benchmarks/self-play-direct-inference-4x4070-super-20260722/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Primary throughput |

## Benchmark records 61–79

| Benchmark record | Destination | Role/evidence |
| --- | --- | --- |
| [Direct inference RTX 3060](../benchmarks/self-play-direct-inference-rtx3060-20260722/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Primary throughput |
| [Graph multiworker](../benchmarks/self-play-graph-multiworker-8xrtx4070super-20260824/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Supporting throughput; not graph-search evidence |
| [MCTS node arena](../benchmarks/self-play-mcts-node-arena-20260720/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting throughput and mechanics |
| [Search CPU study](../benchmarks/self-play-search-cpu-i7-11370h-20260824/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Primary throughput; stand-in CPU caveat |
| [Submission 8x4070S](../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Primary throughput |
| [Self-play throughput 4x3060](../benchmarks/self-play-throughput-4xrtx3060-20260809/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Supporting historical throughput |
| [Self-play pause trade-off](../benchmarks/selfplay-pause-tradeoff-rtx4070s-20260902/README.md) | [Training overlap](04b-data-and-replay.md#actor-trainer-overlap) | Primary throughput and observational online-learning evidence; regime-specific |
| [Supervised testbed](../benchmarks/supervised-testbed-rtx4070-20260821/README.md) | [Policy representations](04c-networks-and-training.md#three-policy-representations) | Supporting, inconclusive proxy evidence |
| [INT8 architecture screen](../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md) | [Inference runtime](05-systems-optimization.md#inference-runtimes-and-precision) | Primary proxy and throughput |
| [INT8 replay screen](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md) | [Quantization architecture](04c-networks-and-training.md#quantization-as-an-architectural-constraint), [Inference runtime](05-systems-optimization.md#inference-runtimes-and-precision) | Primary proxy, throughput, and mechanics |
| [INT8 cadence control](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/cadence/README.md) | [TensorRT refit failure](05a-three-failures.md#prediction-drift-under-tensorrt-refitting) | Supporting throughput and mechanics |
| [INT8 failed variants](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/raw/failures/README.md) | [Provenance](02-methodology-and-evidence.md#reading-the-component-experiments) | Supporting evidence hygiene |
| [INT8 salvage](../benchmarks/tensorrt-int8-salvage-rtx4070s-20260912/README.md) | [Inference runtime](05-systems-optimization.md#inference-runtimes-and-precision) | Primary negative proxy and throughput |
| [Native TensorRT backend](../benchmarks/tensorrt-native-backend-rtx4070s-20260912/README.md) | [Inference runtime](05-systems-optimization.md#inference-runtimes-and-precision) | Primary throughput and mechanics |
| [Chess throughput history](../benchmarks/throughput-history-chess-20260821/README.md) | [Native search](05-systems-optimization.md#native-ownership-of-the-search-loop) | Supporting chronology, not a controlled curve |
| [Controlled initialization comparison](../benchmarks/v35-code-v42-generation0-controlled-ab-20260913/README.md) | [Optimization controls](04c-networks-and-training.md#bootstrap-and-optimization-controls) | Supporting protocol mechanics, no efficacy result |
| [INT8 self-play topology decomposition](../benchmarks/v39-selfplay-throughput-rtx4070s-20260913/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Primary throughput decomposition |
| [Medium-model pre-fold backend](../benchmarks/v76-v35-medium-prefold-backend-20260918/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Primary throughput for the 14-block, 160-channel model |
| [Small-model pre-fold backend](../benchmarks/v76-v35-small-prefold-backend-20260918/README.md) | [Batch submission](05-systems-optimization.md#batching-and-host-side-submission) | Primary throughput and proxy evidence for the 12-block, 128-channel model |

## Coverage conclusion

All nine substantive analysis records and all 79 benchmark README records have a destination above. This does not
make every record equally important. Supporting smokes and superseded controls remain discoverable without inflating
them into narrative results; primary artifacts carry the report's claims. Architecture, plan, and history records
are cited in the relevant topic chapters when they are the only evidence for an implemented decision, such as graph
search, or when they define current ownership and invariants.
