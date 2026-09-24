# 8. Final training and evaluation results

Teacher training, the ten-row terminal matrix, the parallel-search sweep, and both distilled student experiments
are complete and checksum-covered. The protocols and all ten terminal rows are reproduced in Appendix B.

## What was selected

The strongest fully retained checkpoint is a **6.32-million-parameter, 14-block, 160-channel convolutional model**
with scaled post-activation residual blocks, global context conditioning, and a chess from-to attention policy head.
It was trained with quantization-aware training. Searched results use its INT8 TensorRT artifact; policy-only results
use the corresponding float TorchScript export because no native search service is involved. The inference path is
therefore stated with each result rather than silently treating unlike backends as identical.

The ladder's strongest region spans several nearby checkpoints rather than a single isolated spike. Checkpoint 1026
was the last checkpoint in that region for which the full model, optimizer, QAT, ONNX, and TensorRT set had been
retained. A subsequently grown 19-block, 176-channel model recovered its parent's strength but remained flat and did
not justify replacing the selected checkpoint.

## Training outcome

The report-scoped final lineage contains **180 observations through exactly 2.5 effective days**. It rose from 798
to **2,372.2** at the cutoff and reached a peak of **2,407.6** inside that interval. Later points belong to capacity
and training experiments that did not improve the accepted result and are excluded from the comparison rather than
presented as continued final-model training.

The previous four-day baseline is cut at exactly **3.0 days**, retaining 143 clean observations and ending at
2,265.4 Elo. Later points from its noisy terminal evaluation interval are excluded. The two plotted endpoints differ
by 106.8 Elo, and the final endpoint is 348.2 Elo above the early baseline's terminal observation. These are
descriptions of the trimmed curves, not valid cross-campaign strength estimates, because the previous baseline used a
single-rung fit while the final campaign's generic ladder series used a three-rung bracketed fit.

A retrospective plateau comparison on the same three-rung estimator places the previous baseline at 2,283.9 Elo
and the final recipe at 2,358.0 Elo: **+74.1 Elo**, with an approximately **±15 Elo transfer/sensitivity allowance**.
The allowance reflects uncertainty in transferring the estimator correction; it is not a game-level bootstrap
confidence interval. The report therefore uses **about +74 Elo under a matched estimator** as the cross-campaign
headline and does not compare raw peaks. Repeating the arithmetic with only the report figure's clean 3.0- and
2.5-day windows gives **+75.8 Elo**; the conclusion is not driven by the later points omitted from the plot. The
plateau arithmetic and its transfer assumption are given in Appendix B.

![64-search ladder Elo across five chess training campaigns](figures/chess-ladder-progress-paper.svg)

Figure 8: Each curve uses the bias-corrected 0.95 exponential moving average used by the project's earlier
training-dynamics plots. The final recipe is cut at 2.5 days and the previous four-day baseline at 3.0 days. The
figure is descriptive; the +74.1
Elo claim comes from the matched-estimator plateau audit rather than from subtracting its displayed endpoints.

That stitched curve is not the entire compute history. It deliberately excludes two reverted branches while retaining
their raw time in the source export:

1. An early INT8 conversion collapse was detected after one ladder observation and reverted.
2. A from-scratch capacity increase was promoted because training losses were compared across candidates that had
   seen different numbers of examples. The larger candidate was still roughly 270 Elo weaker. Training returned to
   the last valid checkpoint and removed 4,040,112 contaminated replay rows.

The second incident changed the method, not merely the operational state. Promotion now uses a match between the
artifacts intended for deployment rather than training-loss parity, and the tested capacity-growth recovery starts
from a function-preserving widening of the parent. The corrected larger model reached parity but did not break
through the plateau during its limited continuation. This does not identify the limiting factor: longer training,
post-growth optimization, target quality, replay composition, and useful additional capacity remain confounded.

The selected checkpoint records **408,500 completed optimizer steps**. With the configured global batch of 2,048
this corresponds to **836,608,000 training presentations**. The frozen coordinator events cover all 817 contiguous
training quanta on that path. Summing their per-quantum ingested-game counters gives **3,249,647 completed games**;
the credit ledger ends at approximately **209.15 million net materialized positions**, or about **4.00 training
presentations per net materialized position**. Replay held **16 million live rows** at the selected checkpoint.
These are selected-lineage counters, not totals for every experiment or every game generated on the node. The
cumulative position scalar is float32 in TensorBoard, so its final few integer digits are not meaningful. Appendix A
shows the training diagnostics and clarifies the boundary of each count.

The network-evaluation total is estimated rather than counted. Using a conservative 100 searched plies per completed
game and roughly 600 simulations per ply gives about 195 billion search simulations across 3.25 million games,
reported as an estimate in the abstract. Most require a neural-network evaluation; terminal leaves and reuse
make this an approximate scale measure, not an exact forward-pass counter.

Across 480 small-model quanta, median measured trainer throughput was 16,977 samples/s; across 337 medium-model
quanta it was 11,194. Model shape, schedule, and concurrent workload all differed, so this is not an isolated
model-size effect.

The narrow cost attached to the selected checkpoint is **$43.20**, calculated as 60 accepted-lineage hours at
`$0.72/h`. It excludes the reverted work, later growth experiment, distillation, evaluation, and idle rental time.
Until total spend is reconciled, it must not be described as the project's total compute cost.

## Strength across four decades of search

The terminal protocol used 100 games per row from 50 colour-swapped opening pairs against single-threaded Stockfish
13 at fixed node limits. For each search budget, the reported rating is the opponent rung whose score lies closest to
0.500. Anchor ratings come from
Marco Meloni's fixed-node Stockfish 13 benchmark [9],
which connects Stockfish through Fruit 2.2.1 to the historical SSDF scale. The resulting values are benchmark Elo,
not FIDE ratings or estimates on a current unrestricted-engine list.

![Final model playing strength across measured search budgets](figures/final-search-curve-paper.svg)

Figure 9: The connected points use the opponent rung with score nearest 0.5; pale diamonds show the other
measured rung at each budget. Vertical bars are the reported 95% confidence intervals. The categorical horizontal
axis keeps policy-only play visible alongside the searched conditions; it does not imply equal compute spacing.
Policy-only uses the float export, while searched points use INT8 TensorRT. The parallel-search count also changes
with budget, so this is a measured operating curve rather than an isolated node-budget experiment.

The observed policy-only-to-deepest operating-point difference is 1,593 benchmark Elo, with smaller increments at
each later search budget. This is not an isolated search effect: the policy-only and searched artifacts differ, and
parallelism changes across searched conditions. The bootstrap intervals quantify match sampling conditional on the
fixed, graph-read Stockfish anchor ratings; they do not include anchor-calibration uncertainty. The 100,000-search
estimate is unusually well anchored: an independent 100,000-node opponent
gives 3,247 Elo, only four points below the 200,000-node result. That local agreement supports the top headline, but
does not validate extrapolation beyond the measured anchors.

Appendix B reports the complete ten-row W/D/L matrix and match-bootstrap intervals.

## Two protocol effects that matter

### Opponent rungs disagree at lower budgets

At 100, 1,000, 10,000, and 100,000 searches, the harder opponent rung reads 128, 102, 46, and 4 Elo higher than the
easier rung. Ideal transitive Elo would give the same estimate from both. Draw behavior, anchor calibration error,
matchup effects, and sampling noise are possible contributors, but the project did not isolate the cause. Choosing
the score closest to 0.5 limits extrapolation. The disagreement matters at low budgets but is immaterial to the
deepest result.

### Parallelism buys time by spending strength

The headline curve is an operating curve, not a pure search-budget ablation: it uses one parallel search at 100 and
1,000 searches, four at 10,000, and sixteen at 100,000. A controlled 1,000-search sweep measured 2,823 Elo with one
parallel search, 2,804 with four, and 2,778 with sixteen. Four-way parallelism therefore cost 19 Elo; sixteen-way cost
45 Elo.

The two recorded timing totals give **18.1–18.5 minutes** with one parallel search, **3.4–3.8** with four, and
**1.3–1.8** with sixteen. These are descriptive ranges across the operational recap and result manifests, not
confidence intervals. The corresponding paired speedup ratios span approximately **4.8–5.3x** for four-way and
**10.1–13.9x** for sixteen-way parallelism. At only 100 searches, sixteen-way parallelism cost 235 Elo, so
parallelism cannot be changed silently across a compute curve. The available points suggest that a fixed parallel
count becomes less harmful as the total budget grows, but they are too sparse to define how much parallelism is
effectively free at each budget. That frontier would require a dedicated budget-by-parallelism grid.

## What the student establishes—and what it does not

Both students have **470,295 parameters**, 13.4 times fewer than the teacher, and trained on the same 20-million-row
replay snapshot. At 10,000 searches against the same 20,000-node anchor, extending training from 36,621 to 110,000
steps moved the estimate only from **2,683 [2,637, 2,731]** to **2,697 [2,640, 2,753]** benchmark Elo. The 14-Elo
central change lies inside match uncertainty, while the longer student's held-out policy loss had nearly flattened.
At 100,000 searches it reached **2,873 [2,819, 2,935]**, but that point is unbracketed. The student used TorchScript
and the teacher INT8 TensorRT, so this is not an architecture-only comparison. Appendix B gives the W/D/L counts and
the skipped-rung boundary. Parameter compression is 13.4x; Elo itself is not a meaningful percentage scale.
