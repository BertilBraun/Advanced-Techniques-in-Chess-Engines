# 8. Final training and evaluation results

After 2.5 days of final training, the selected 6.32-million-parameter chess model reached 3,251 benchmark Elo at
100,000 searches per move in paired games against a fixed-node Stockfish 13 ladder. The results below trace its
training progress, measure its strength across search budgets, and examine the parallel-search and distilled-student
controls. Appendix B gives every terminal match row.

## What was selected

The strongest fully retained checkpoint is a **6.32-million-parameter, 14-block, 160-channel convolutional model**
with scaled post-activation residual blocks, global context conditioning, and a chess from-to attention policy head.
It was trained with quantization-aware training. Searched results use its INT8 TensorRT artifact; policy-only results
use the corresponding float TorchScript export because no native search service is involved. The inference path is
therefore stated with each result rather than silently treating unlike backends as identical.

Several nearby checkpoints reached the ladder's strongest region, so the selection does not rest on a single spike.
Checkpoint 1026 was the last in that region with the complete model, optimizer, QAT, ONNX, and TensorRT artifacts
retained. A subsequently grown 19-block, 176-channel model recovered its parent's strength but did not surpass it.

## Training outcome

The final lineage rose from 798 to **2,372.2** on the 64-search training ladder over **2.5 effective days**, peaking
at **2,407.6** within the plotted interval. The figure retains 180 observations and stops before later capacity and
training experiments that did not improve the selected result.

The previous baseline is plotted through **3.0 days**, retaining 143 observations and ending at 2,265.4 Elo before
its noisy terminal evaluation interval. The plotted endpoints differ by 106.8 Elo; the final endpoint is 348.2 Elo
above the early baseline's terminal observation. Those visual comparisons do not measure cross-campaign strength:
the older curve used a single-rung fit, while the final curve used a three-rung bracketed fit.

To compare the campaigns more fairly, we put their plateaus on the same three-rung estimator. That retrospective
comparison yields 2,283.9 versus 2,358.0 Elo, or **about +74 Elo** for the final recipe. The estimated transfer
sensitivity is approximately **±15 Elo**, not a match-bootstrap confidence interval. Restricting the calculation
to the clean plotted windows yields +75.8 Elo. Appendix B gives the arithmetic and the transfer assumption; neither
estimate should be read as an isolated gain from any one engineering change.

![64-search ladder Elo across five chess training campaigns](figures/chess-ladder-progress-paper.svg)

Figure 8: Each curve uses the bias-corrected 0.95 exponential moving average used by the project's earlier
training-dynamics plots. The final recipe is cut at 2.5 days and the previous four-day baseline at 3.0 days. The
figure is descriptive; the +74.1
Elo claim comes from the matched-estimator plateau audit rather than from subtracting its displayed endpoints.

The plotted lineage excludes reverted INT8 and capacity-promotion branches. The latter exposed the weakness of
training-loss promotion and led to the paired deployment-artifact matches described in Chapter 6. A later
function-preserving larger-model continuation reached parity without a demonstrated improvement.

The selected checkpoint followed **408,500 optimizer steps** in 817 training quanta. At a global batch of 2,048,
that is **836,608,000 training presentations**. Over the same lineage, self-play completed **3,249,647 games** and
materialized approximately **209.15 million net positions**, or about four training presentations per position.
Replay held **16 million live rows** at selection. These figures describe the selected path, rather than every
discarded experiment or game generated on the node; Appendix A shows their trajectories and counting boundaries.

The network-evaluation total is estimated rather than counted. Using a conservative 100 searched plies per completed
game and roughly 600 simulations per ply gives about 195 billion search simulations across 3.25 million games,
reported as an estimate in the abstract. Most require a neural-network evaluation; terminal leaves and reuse
make this an approximate scale measure, not an exact forward-pass counter.

Across 480 small-model quanta, median measured trainer throughput was 16,977 samples/s; across 337 medium-model
quanta it was 11,194. Model shape, schedule, and concurrent workload all differed, so this is not an isolated
model-size effect.

The selected training path cost **$43.20** in node rental: 60 hours at `$0.72/h`. This is not the project's total
spend; it excludes discarded experiments, later growth, distillation, evaluation, and idle rental time.

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

### Parallelism trades time against strength

The headline curve is an operating curve, not a pure search-budget ablation: it uses one parallel search at 100 and
1,000 searches, four at 10,000, and sixteen at 100,000. A controlled 1,000-search sweep measured 2,823 Elo with one
parallel search, 2,804 with four, and 2,778 with sixteen. The central estimates fell by 19 and 45 Elo, respectively,
although the match intervals overlap.

The two recorded timing totals give **18.1–18.5 minutes** with one parallel search, **3.4–3.8** with four, and
**1.3–1.8** with sixteen. These are descriptive ranges across the operational recap and result manifests, not
confidence intervals. The corresponding paired speedup ratios span approximately **4.8–5.3x** for four-way and
**10.1–13.9x** for sixteen-way parallelism. At only 100 searches, the sixteen-way central estimate fell by 235 Elo, so
parallelism cannot be changed silently across a compute curve. The available points suggest that a fixed parallel
count becomes less harmful as the total budget grows, but they are too sparse to define how much parallelism is
effectively free at each budget. That frontier would require a dedicated budget-by-parallelism grid.

## What the student establishes—and what it does not

Both students have **470,295 parameters**, 13.4 times fewer than the teacher, and trained on the same separate,
frozen 20-million-row replay snapshot. The selected teacher checkpoint held 16 million live replay rows; the student
snapshot is not that checkpoint's training window. At 10,000 searches against the same 20,000-node anchor,
extending training from 36,621 to 110,000
steps moved the estimate only from **2,683 [2,637, 2,731]** to **2,697 [2,640, 2,753]** benchmark Elo. The 14-Elo
central change lies inside match uncertainty, while the longer student's held-out policy loss had nearly flattened.
At 100,000 searches it reached **2,873 [2,819, 2,935]**, but that point is unbracketed. The student used TorchScript
and the teacher INT8 TensorRT, so this is not an architecture-only comparison. Appendix B gives the W/D/L counts and
the skipped-rung boundary. Parameter compression is 13.4x; Elo itself is not a meaningful percentage scale.
