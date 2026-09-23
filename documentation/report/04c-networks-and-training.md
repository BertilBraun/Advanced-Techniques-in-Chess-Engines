# 4C. Network and training investigations

This chapter separates trunk, head, context, initialization, optimizer, auxiliary, and compression evidence. Their
status summary is [Network and training experiments](../experiments/networks-and-training.md).

## CNN versus attention

The first attention results were not clean architecture comparisons. Generation-zero BatchNorm export made the
attention prior nearly uniform while the CNN control was accidentally extremely sharp. Some throughput comparisons
also used FP32 although production used BF16. These records remain evidence of the failure mode, not evidence that
attention is inherently unsuitable.

The later viability study calibrated bootstrap priors, used a fixed teacher dataset, paired held-out cross-entropy,
and measured production-card throughput separately. A bare attention trunk did not beat the CNN at a matched head.
The best attention cell improved the proxy, but most of its advantage came from the from-to policy head and it paid a
large throughput cost. The project retained the CNN. This is proxy and throughput evidence, not a terminal online
trunk ablation.

Packed-QKV and SDPA-backend work improved or clarified the attention implementation but became superseded when the
attention family was not selected. Hybrid CNN/transformer trunks remained proposals.

## Policy, context, and value heads

The from-to head replaced a large dense projection with structured origin/destination scoring. On the unchanged
12x128 CNN it captured most of the held-out-policy improvement for a small forward-throughput cost and became the
final head. An earlier low-rank dense-head bake-off was not preserved as a tracked benchmark; its exact numbers
should not appear as equivalent evidence.

Global-pooling context every second block is retained and externally motivated but lacks a one-variable chess
ablation. The policy and value heads share the convolutional trunk; a partially split late trunk was discussed but
not tested. A 32-channel value head received a short matched frozen-replay probe and did not justify its added cost,
so the final value head stays at two channels.

## Quantization-friendly residual blocks

Ordinary post-training INT8 could not preserve the network outputs. The successful architecture uses scaled
post-activation residual branches and caps activations at 6, making QAT ranges bounded. Frozen-replay screens show
that the block learns under fake quantization and can serve a faithful pre-fold INT8 graph. This is a joint
architecture/deployment result, not evidence that the residual block is stronger in float chess.

## Progressive sizing

The throughput premise is measured: small early models can generate substantially more search on the production
GPU. The mechanism is also durable: active and candidate trainers consume the same replay-batch identity; candidate
start is triggered by searched-Elo gain per hour; paired candidate-versus-active matches govern promotion; private
candidate checkpoints survive restart; publication is ordered and idempotent. The former loss-EMA gate was removed
after unequal candidate training made its losses incomparable and promoted a much weaker larger model.

Function-preserving growth later mapped the trained 14x160 network into 19x176, recovered INT8 fidelity with QAT,
and removed the need for the larger model to relearn its parent's function. The larger continuation reached parity
but did not improve the strength curve, so the reported model remains 14x160. This bounds the conclusion to the
tested recipe: capacity was not the immediate bottleneck, but the experiment does not establish a general limit on
larger networks.

The exact 12x128 → 14x160 → 19x176 ladder has no fixed-model counterfactual. Its causal strength-per-dollar gain is
unresolved. The evidence establishes throughput and controller mechanics; the completed campaign establishes online
learning and strength only for the assembled bundle.

## Bootstrap calibration and deterministic initialization

Generation zero creates the first self-play targets, so its output distribution is part of the algorithm. The final
bootstrap path measures policy shape on 516 real encoded probes, selects among candidate initializations under policy
and WDL constraints, and calibrates policy scale toward a top-three-mass target. It may sharpen or dampen a model;
it is not an architecture-specific constant.

The adaptive-search postmortem found that the configured random seed did not reach network construction. Nominally
identical independent runs therefore began from different tensors. That defect was fixed, and the subsequent
controlled audit moved toward exact initialization, checkpoint, replay, and inference-artifact comparisons. The
[regression audit](../analysis/v35-v42-regression-audit-20260913.md) and
[executable bisect](../analysis/v35-v42-executable-bisect-20260913.md) are methodological evidence: uncontrolled
short-run rankings should not be treated causally.

## Auxiliary objectives

The final model trains next-policy at weight 0.15 and normalized remaining game length at weight 0.1. Both heads are
removed from inference. Fixed-batch overfit and replay audits establish gradients, eligibility, symmetry, and
censoring. No long matched online ablation isolates either head. Other proposed heads—future action, uncertainty,
root Q, material, survival, king safety, control maps, and search correction—were not final experiments.

## AdamW, SGD, and learning-rate evidence

AdamW produced the verified previous four-day baseline and is therefore superseded, not disproven. Frozen-replay
screens using that model's data showed that Nesterov SGD could train the QAT network and ranked candidate warmups,
fold boundaries, and deployment rates.
Delayed folding helped the historical pre-fold schedule; within the tested short horizon, higher post-fold target
rates improved proxy fitting without instability.

The final schedule is not the screen winner copied literally. It keeps the authoritative training model pre-fold
until one million optimizer steps and the deployment copy inherits the main linear learning rate. The screens provide
proxy, mechanics, and selection evidence. Final online strength belongs to the assembled run.

Gradient clipping was frequent in several frozen screens but did not prevent learning. This is a diagnostic to plot,
not proof that the clipping threshold is optimal. EMA/SWA, gradient accumulation, and dynamic loss balancing remained
proposals.

## Distillation and compression

The teacher-imitation probe varied student size, data volume, auxiliary imitation, and search depth. It quantified
how deeper search widened the teacher/student gap but inherited dataset sampling defects and did not produce a final
training stage.

The later replay-compression study of the previous four-day baseline published a 0.47M-parameter student and
evaluated equal searches, approximate equal serving time, and equal network compute. The answer depended on the
constraint: the student was much smaller
and faster but did not match the teacher. Compression is therefore measured and useful, yet inconclusive as a
replacement for direct final-model training
([probe](../benchmarks/chess-distillation-probe-rtx3060-20260827/README.md),
[replay compression](../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md)).
