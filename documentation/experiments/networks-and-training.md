# Network and training experiments

## Ledger

| Technique | Status | What the evidence establishes | Principal evidence |
| --- | --- | --- | --- |
| Convolutional residual trunk | **Retained** | After bootstrap defects and runtime confounds were separated, attention remained viable but did not earn replacement of the CNN for the final run. | [attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md), [contended comparison](../benchmarks/chess-architecture-contended-rtx3060-20260817/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| Pure attention trunk | **Implemented and rejected** | Multiple shapes and backends were implemented. Some early results were invalidated by FP32 inference or generation-zero prior defects; the controlled viability study still selected a CNN-derived design for production. | [attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md), [SDPA control](../benchmarks/chess-attention-sdpa-backends-rtx4070s-20260818/README.md) |
| Packed-QKV attention | **Superseded** | Retained as the cleaner attention primitive in that implementation, but attention itself is not in the final chess recipe. The original inference conclusion was invalid because it measured FP32 rather than production BF16. | [packed-QKV benchmark](../benchmarks/chess-attention-packed-qkv-rtx3060-20260818/README.md) |
| Hybrid / transformer trunk | **Proposed only** | No separate hybrid or transformer production experiment is documented beyond the attention-family studies. | [research ledger](../../THINGS_TO_TRY.md) |
| Shared policy/value trunk | **Retained** | The production CNN shares its trunk. No controlled late-block separation study is documented, so the project cannot claim trunk sharing beat partial separation. | [Final config](../../py/configs/production/chess-final-config.yaml), [research ledger](../../THINGS_TO_TRY.md) |
| From-to attention policy head | **Retained** | Frozen-replay studies found the policy-head change useful at small throughput cost; it became the final head. Evidence is a supervised proxy, not isolated online Elo. | [attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| Dense low-rank policy head | **Superseded** | An earlier run-local bake-off selected a rank-96 dense bottleneck, but the evidence bundle was not preserved under `documentation/benchmarks` and the final recipe later moved to the from-to head. | [later head comparison](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md) |
| Global-pooling residual context | **Retained** | Present every second residual block in all final stages. It was motivated by external evidence and integrated into later screens, but no repository benchmark isolates its chess strength gain. | [Final config](../../py/configs/production/chess-final-config.yaml), [reference-recipe analysis](../analysis/reference-recipes-for-a-compute-poor-run.md) |
| Scaled post-activation residual blocks | **Retained** | This quantization-friendly block learned normally under QAT and preserved usable deployment fidelity after the pre-fold design was adopted. | [INT8 replay screen](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md), [architecture screen](../benchmarks/tensorrt-int8-architecture-screen-rtx4070s-20260912/README.md) |
| Progressive 12x128 → 14x160 → 19x176 sizing | **Retained** | Small networks provide higher early actor throughput; progressive training/promotion is implemented durably. The exact final ladder and Elo-plateau trigger have not been isolated against one fixed network. | [Final config](../../py/configs/production/chess-final-config.yaml), [throughput benchmark](../benchmarks/progressive-sizing-throughput-rtx4070super-20260823/README.md), [architecture](../architecture/progressive-model-sizing.md) |
| Elo-plateau candidate start and loss-based promotion | **Retained** | Implemented production control logic with crash recovery. Its reliability is tested; its independent strength benefit is not a completed experiment. | [progressive architecture](../architecture/progressive-model-sizing.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| Next-policy auxiliary head | **Retained** | Trained at weight 0.15 and omitted from inference artifacts. It is part of the final bundle, but no clean online ablation isolates its contribution. | [Final config](../../py/configs/production/chess-final-config.yaml), [v8 data audit](../analysis/v8-training-data-comparison-20260826.md), [overfit benchmark](../benchmarks/chess-overfit-rtx3090-20260819/README.md) |
| Remaining-game-length auxiliary head | **Retained** | Trained at weight 0.1 with censored cut-game labels. Overfit/data audits validate mechanics; they do not prove Elo gain. | [Final config](../../py/configs/production/chess-final-config.yaml), [v8 data audit](../analysis/v8-training-data-comparison-20260826.md), [overfit benchmark](../benchmarks/chess-overfit-rtx3090-20260819/README.md) |
| Other auxiliary heads | **Proposed only** | Future action, uncertainty, root-Q, material, survival, king-safety, control-map, and related heads were not final experiments. | [research ledger](../../THINGS_TO_TRY.md) |
| SGD with Nesterov momentum | **Retained** | Frozen-replay screens showed that Nesterov SGD can fit the QAT target and selected useful schedules. Online strength comes from the final run, not the screens alone. | [SGD replay screen](../benchmarks/chess-sgd-replay-screen-rtx4070s-20260913/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| AdamW production training | **Superseded** | Earlier successful lineages used AdamW; the final run uses SGD. Historical success prevents calling AdamW rejected in general. | [v34 dynamics](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| Frozen-replay LR/fold screens | **Inconclusive** | They provide controlled proxy rankings—delayed folding and higher post-fold rates learned faster—but do not directly establish self-play Elo. The final recipe subsequently changed fold timing and inherited deployment LR. | [pre-fold factorial](../benchmarks/chess-sgd-prefold-factorial-rtx4070s-20260914/README.md), [post-fold sweep](../benchmarks/chess-sgd-postfold-lr-rtx4070s-20260914/README.md), [Final config](../../py/configs/production/chess-final-config.yaml) |
| Bootstrap prior calibration and deterministic initialization | **Retained** | Early attention runs exposed near-uniform exported priors, while later audits found the configured seed did not reach generation-zero construction. Shape calibration and deterministic model creation are correctness/reproducibility measures. | [attention viability](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md), [adaptive conclusion](../analysis/adaptive-search-conclusion-20260904.md), [v35-v42 audit](../analysis/v35-v42-regression-audit-20260913.md) |
| Replay distillation | **Inconclusive** | Student imitation and v34 replay probes quantified fitting, search-depth gaps, throughput, and sampling defects. They did not produce a retained production distillation stage. | [teacher-imitation probe](../benchmarks/chess-distillation-probe-rtx3060-20260827/README.md), [v34 replay distillation](../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) |
| Wider value head (32 vs 2 channels) | **Implemented and rejected** | A short matched probe gained only 0.00309 total loss while costing parameters and throughput; it did not earn inclusion. | [INT8 replay screen](../benchmarks/tensorrt-int8-replay-screen-rtx4070s-20260912/README.md) |
| EMA/SWA, gradient accumulation, dynamic loss balancing | **Proposed only** | Listed as candidates; no completed efficacy experiment is documented. | [research ledger](../../THINGS_TO_TRY.md) |

## CNN versus attention

The architecture story has two corrections. First, the earliest attention training failure was not clean evidence
against attention: generation-zero BatchNorm export produced a nearly uniform prior, changing the self-play targets
before training had a chance to compare trunks. Second, some inference comparisons ran FP32 despite production using
BF16. Those results are preserved but cannot support the original deployment claims.

The later viability study used a fixed teacher dataset, paired held-out cross-entropy, corrected bootstrap priors,
and production-card throughput controls. Its actionable result was the from-to policy head; the selected final trunk
remained convolutional. This supports “attention was investigated and did not earn adoption,” not “transformers are
intrinsically worse for chess.”

## Head and context design

The final network combines a shared convolutional trunk, global-pooling bias every second block, a from-to attention
policy head, and a small WDL value head. Evidence strength differs by component. The from-to head has a controlled
frozen-replay comparison and measured forward cost. Global pooling is present throughout later successful and QAT
screens but lacks a one-variable chess ablation. Shared trunks were the implemented default; partially separated
late trunks were discussed but not tested. The technical report should not group all three under one “architecture
ablation.”

## Progressive sizing

The throughput benchmark established the premise: at early self-play conditions a small network can generate far
more searches per second than a large one. The architecture then added persistent per-model trainers, identical
replay batches for active/candidate comparison, plateau-triggered candidate start, paired loss EMAs, ordered
promotion, private checkpoints, and crash-idempotent publication. Those are strong implementation facts.

What remains unmeasured is the causal strength-per-dollar gain of the exact 12x128 → 14x160 → 19x176 policy versus
a fixed model. Terminal results show that the mechanism can produce a strong model, not what a counterfactual fixed
run would have achieved.

## Auxiliary objectives

Next-policy and remaining-length targets are cheap labels already available from trajectories. Overfit studies
showed that the production objective is trainable, and replay audits verified eligibility/censoring behavior. There
is no matched long online ablation in the current evidence. They should be described as retained, motivated
auxiliaries rather than quoted with external KataGo multipliers as if those transferred directly to chess.

## Optimizer, folding, and learning-rate screens

The frozen v34 replay made controlled SGD/QAT comparisons possible. Nesterov SGD learned stably; delayed folding
improved the historical pre-fold schedule; and post-fold target rates through 0.08 improved the short proxy without
divergence. These screens were valuable engineering selection tools, but the final configuration differs: it uses a
long pre-fold phase (`fold_after_optimizer_steps: 1000000`) and inherits the main linear learning rate after folding.
Accordingly, the screen winners are historical stepping stones, not a literal description of the final schedule.

The v35-to-v42 regression audit also cautions against reading short independent runs causally. The configured seed
had not seeded generation-zero construction, so apparently identical arms began from different weights. Later work
fixed that reproducibility defect.

## Remaining evidence gaps

- No final-lineage one-variable Elo ablations exist for global pooling, the from-to head, trunk sharing, or either
  auxiliary head.
- The exact progressive ladder lacks a fixed-model counterfactual.
- The final report should separate proxy-selected optimizer schedules from the schedule actually used by the final
  configuration.
- The old low-rank policy-head bake-off is missing a tracked evidence bundle; any numerical claim from it should be
  treated as historical unless the artifact is recovered.
