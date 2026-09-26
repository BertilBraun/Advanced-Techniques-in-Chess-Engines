# 4.3. Choosing what the network predicts

![Comparison of dense reduced-action, spatial move-plane, and from-to policy heads](figures/policy-representations.svg)

Figure 7: Dense heads project to a move list, move-plane heads predict spatial move types, and the retained from-to
head scores square pairs before gathering canonical actions and masking illegal moves.

Network architecture constrains both prediction quality and the volume of search affordable during training.
We compared policy representations, convolutional and attention trunks, global context, and value heads, then
examined model growth and quantization. The policy head was a major source of parameter cost in small networks;
Figure 7 summarizes the three representations evaluated.

## Three policy representations

All three policy families were implemented against the chess move interface, whose selected encoding contains
1,880 canonical actions. Controlled fitting experiments compare held-out policy cross-entropy in nats.

The **dense head** flattened a small spatial projection into 1,880 logits. Dense heads trained successful models,
but on a small trunk their projection could dominate the parameter count. Variants changed projection width,
spatial reduction, and final-map rank. A rank-96 variant reduced head size from roughly 484,000 to 207,000
parameters.

The **move-plane head** represented sliding directions, knight moves, and promotions as spatial move types. It
could preserve board structure, but its experiments did not show an advantage over the dense control. In a short
supervised comparison, the repaired plane head reached 2.1828 held-out policy cross-entropy after about 2,500
steps versus 2.0824 for the dense control, while its curve was still improving faster. An online plane-head trial
was retired after slower learning and weaker play.

The two implementations differed: one gathered 1,880 canonical logits from 76 planes (56 sliding directions,
eight knight moves, and 12 promotions); the other exposed all 4,864 plane-square cells and normalized only legal
moves. Both express the same basic idea: predict move types at board locations rather than score a flat action list.

The retained **from-to head** uses query-key dot products over the 64 trunk squares, with a fixed gather into the
canonical action space and separate promotion offsets. This retains spatial structure without a large dense
projection. On the controlled convolutional trunk, the head used 51,072 parameters rather than 483,680 for the dense
alternative. Section 3.5 and Appendix D specify its representation and action mapping.

Holding that trunk fixed, the from-to head improved the held-out policy gap by 0.0298 nats (paired 95% interval
0.0285--0.0311). Spending the saved parameters on a wider trunk added only 0.0018 nats in the three measured cells;
the missing fourth cell prevents fully separating the two effects. Forward throughput fell by approximately 1.9%
at batch 512 and 9% at batch 64. The from-to head therefore improved policy fit and reduced parameter count, with
a serving penalty that was smaller at the large batches used for self-play.

## Board input and rule state

The selected model receives 52 side-to-move-canonical board planes: pieces, castling rights, en passant, checks,
repetition, the eight most recent moves, material counts, and the fifty-move counter. Rule-sensitive planes keep
positions with different legal or draw states distinguishable. [Appendix D](appendix-d-reproducibility.md), Table D1
lists all 52 planes and their encodings. File reflection is the only augmentation; it also
mirrors action targets and exchanges kingside and queenside castling planes. Some earlier component comparisons
used a 29-plane input, so their absolute scores are not input-matched to the final model.

## Convolution, attention, and global context

The trunk comparison evaluated whether attention improved board-wide feature learning enough to justify its
serving cost. Both designs shared their trunk across policy, value, and auxiliary heads. Because dense primary and
auxiliary policy heads could dominate a small model's parameters, head choice had to be controlled alongside the trunk.

The attention alternative treated the 64 squares as tokens, with learned row and column embeddings,
pre-normalized self-attention, and GELU feed-forward blocks. No-bias, relative-offset, and input-dependent
Smolgen-style attention biases were implemented. These biases respectively add no positional preference, depend on
the offset between squares, or depend on the board itself. The preserved comparison covers the first and third.

With bootstrap policy shape, head, runtime, and precision controlled, the convolutional trunk beat bare attention
by 0.0060 nats on held-out teacher data. Smolgen gave the best attention cell, but changing the attention model's
dense head to from-to improved the held-out gap by a much larger 0.1573 nats. Attention also consumed more memory
and served more slowly at the relevant batches. The tested convolutional design was the better choice for this
workload; the result does not rule out attention in other chess systems.

Local convolution receives global context every second residual block: board-wide means and maxima from one
quarter of the channels are projected back as biases on local features. Squeeze-excitation was another implemented
context mechanism, using board-wide information to rescale channels. Global pooling appeared to learn faster,
with no clear final-strength difference; this remains a qualitative observation because the comparison results
are unavailable. The pooled features give local convolutions access to the whole board, following the KataGo
precedent [2]. Those context operations remain in floating point when the main convolutions are quantized.

## Value and training-only heads

The outcome head predicts win, draw, and loss rather than a single scalar. Search converts this distribution to an
expected value when necessary, while training and diagnostics retain the draw probability. A compact
two-channel spatial reduction and 48-unit hidden layer was retained. A matched 32-channel probe added 97,020
parameters and reduced measured training throughput by 1.31%, while improving total loss by only 0.00309 in one
short seed and slightly worsening WDL loss. This justified keeping the smaller head. The earlier scalar-to-WDL
transition has no preserved isolated strength comparison.

Training also predicts the next move's searched policy and normalized remaining game length, with loss weights
0.15 and 0.1. The next-policy target uses the following state's own side-to-move action space,
legality mask, and symmetry transformation. Both heads train the shared representation without making inference
more expensive, because deployment removes them. Other auxiliaries were set aside while debugging potential
interference, not because they were shown harmful. The retained pair has not received a long matched self-play
ablation.

## Bootstrap and optimization controls

Initialization shapes the first search priors and, through them, the first self-play targets. In an initial
architecture comparison, the attention policy put only about 0.11 probability mass on its top three moves while
the convolutional control was effectively one-hot. Neither extreme represented chess knowledge, but each induced
different data. The retained bootstrap uses architecture-appropriate initialization, deterministic construction,
a small final policy projection, and calibration on 516 encoded positions toward a common policy concentration.
Those positions set numerical scale without supplying supervised chess targets.

The retained optimizer is Nesterov SGD. Frozen-replay tests helped choose warm-up and learning-rate settings before
committing to self-play runs. AdamW had also trained successful models; the choice of SGD does not mean AdamW failed.
Gradient clipping limits unusually large updates. The final run completed 408,500 optimizer steps, with the learning
rate, losses, and gradient norms shown in Appendix A.

## Quantization as an architectural constraint

INT8 deployment required changes to the residual architecture as well as quantization-aware training (QAT).
The experiments compared prediction fidelity and compiled throughput: an architecture that tolerates rounding
but introduces expensive precision conversions may still be unsuitable for self-play.

Post-training quantization of the ordinary residual tower accelerated inference but severely distorted predictions. Activation ranges
grew from about 0.78 near the input to roughly 40--43 late in the network; depending on calibration, full-trunk INT8
preserved only 13.3--25.9% policy top-one agreement. Weight-only quantization was faithful but slower than TensorRT
FP16. Quantization therefore became a network-design problem rather than a final export switch.

A scaled pre-activation block bounded residual activation ranges,
but its normalization, clipping, scaling, and repeated precision conversions fragmented the compiled TensorRT graph.
It expanded an 83-layer FP16 graph to 330 layers
and an INT8 graph to 470 layers, with many reformats, yet still failed fidelity. The retained scaled post-activation
block instead keeps the conventional sequence of convolution, normalization, capped activation, convolution,
normalization, scaled residual addition, and capped activation. Activations are capped at six, and residual branches
are scaled by the inverse square root of depth. The scale can be folded into the second convolution at export, which
preserves more efficient compiler tactics.

Batch-normalization folding changed the quantization behaviour as well as the compiled graph. The final run keeps
the trainable model unfused and performs folding and recalibration on a deployment copy. Separate fitting
experiments also evaluated continued training in the folded topology.

The scaled post-activation block learned normally under quantization-aware training. After continuation in the folded deployment topology,
a production-sized smoke test reached roughly 135,000 INT8 positions per second, compared with 60,000 for
TorchScript BF16 and 99,000 for TensorRT FP16. These are model-core rates, not end-to-end self-play rates. Folding
only after training damaged agreement, and global context and the heads remain outside the INT8 trunk. The result is
evidence for an architecture compatible with efficient INT8 inference, rather than a playing-strength advantage
over the ordinary floating-point residual block.

## Progressive model sizing

![Small-to-medium promotion is supported while the larger-model transition remains unresolved](figures/progressive-model-sizing.svg)

Figure 8: A small model reduces early self-play cost, and a medium candidate trains on the same replay before
paired-match promotion. The larger candidate may avoid catch-up with function-preserving growth, but the limited
continuation did not demonstrate a strength gain; the reported checkpoint remains medium-sized.

Following KataGo [2], progressive sizing uses a small network for early self-play while a larger candidate trains
on the same replay before promotion. The throughput advantage is most useful while the smaller network has enough
capacity to absorb the available experience. The small-to-medium transition worked repeatedly in this project.

Candidate start follows a stage-specific searched-Elo plateau; promotion instead requires two passing paired
matches against the active model. Section 6.3 explains why matches replaced training loss as the promotion criterion;
Appendix D states the retained thresholds.

An independently initialized larger candidate needed substantial catch-up. Function-preserving growth instead
initializes the larger model to compute the same predictions as the medium model. It avoids relearning that
function, but might also bias optimization toward the smaller model's existing representation. The
limited grown-model continuation reached parity without a clear strength gain. The reported model therefore remains
medium-sized; choosing between catch-up and growth needs a matched-compute comparison.

## Distillation and compact models

Distillation evaluated the tradeoff between reduced inference cost and approximation of a stronger teacher.
We tested two sources of supervision: the larger teacher's direct predictions and the targets
already collected in replay. In the first, students learned the teacher's legal-move probabilities and WDL
predictions from games played without search. Increasing this dataset from one million to six million positions
mattered more than a small capacity sweep.
The strongest 1.33-million-parameter student trailed its teacher by 176.1 Elo at 25 searches each and by 38.4 Elo
when it received the measured shallow equal-compute allowance of 58 searches. At 250 searches each the gap widened
to 257.6 Elo. Deeper search amplified the better prior rather than washing out approximation error. Fixed-ply
sampling gave the dataset only one side-to-move parity, and the teacher had a known conversion weakness, so these
absolute gaps do not transfer to the final model.

Replay-target compression instead trained on sparse MCTS visits, outcomes, root values, and replay metadata from a
frozen ten-million-row window. The selected 474,069-parameter student was 13.20 times smaller than its teacher. It
trailed by 291.3 Elo at 64 searches each and by 166.2 Elo when its measured saturated serving advantage allowed 186
searches against 64. It reached statistical parity only at an equal-multiply-accumulate allowance of 850 searches,
which counted neural arithmetic but omitted tree work and inference overhead. The smaller model's theoretical
compute advantage was therefore much larger than the extra search it could actually perform in the same time.

A final compression study trained a 470,295-parameter student on a separate frozen 20-million-row replay
snapshot. At
10,000 searches it reached 2,683 benchmark Elo after roughly 7.5 epochs and 2,697 after roughly 23
epochs. The 14-Elo central difference lay well inside the match intervals, while held-out policy loss had become
nearly flat. Tripling passes over this fixed buffer therefore produced no measurable playing gain. The longer
student reached 2,873 benchmark Elo at 100,000 searches against the one opponent tested there. Chapter 8 presents
these final student results alongside the full model.

The resulting compact models reduced inference cost, but their realized search advantage was substantially smaller
than their parameter or arithmetic ratios. The teacher's strength advantage also increased with search depth in
the measured comparisons. Distillation was separate from the primary self-play training run.

## Decision

The retained network combines the rule-complete 52-plane input, a shared convolutional trunk, periodic global
context, the from-to policy head, a compact WDL head, and training-only next-policy and remaining-length objectives.
Scaled post-activation blocks make the trunk compatible with quantization-aware deployment. Progressive sizing
successfully exploited a small model before handing off to the medium model, while the value of the larger stage
remains unresolved. Component tests support parts of this design, while final playing strength belongs to the
assembled system.
