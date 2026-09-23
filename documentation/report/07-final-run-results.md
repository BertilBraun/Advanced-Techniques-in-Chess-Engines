# 7. Final training and evaluation results

> **Evidence state.** Teacher training and the ten-row terminal evaluation matrix are complete. Seven matrix rows are
> checksum-covered by the current local archive; the policy-only rows and deepest second anchor were produced after
> that pull and await re-fetch. A longer student experiment remains in progress. The quantitative authority is the
> [final-run result record](../results/final-chess-run.md), backed by the
> [compact evidence index](../evidence/final-chess-20260923/README.md).

## What was selected

The strongest fully retained checkpoint is a **6.32-million-parameter, 14-block, 160-channel convolutional model**
with scaled post-activation residual blocks, global context conditioning, and a chess from-to attention policy head.
It was trained with quantization-aware training and evaluated through its INT8 TensorRT artifact. This distinction is
important: the reported strength belongs to the deployed numerical path, not a more accurate float surrogate.

The ladder's strongest region spans several nearby checkpoints rather than a single isolated spike. Generation 1026
was the last checkpoint in that region for which the full model, optimizer, QAT, ONNX, and TensorRT set had been
retained. A subsequently grown 19-block, 176-channel model recovered its parent's strength but remained flat and did
not justify replacing the selected checkpoint.

## Training outcome

The report-scoped final lineage contains **180 observations through exactly 2.5 effective days**. It rose from 798
to **2,372.2** at the cutoff and reached a peak of **2,407.6** inside that interval. Later points belong to capacity
and training experiments that did not improve the accepted result and are excluded from the comparison rather than
presented as continued final-model training.

The previous four-day baseline is cut at exactly **3.0 days**, retaining 143 clean observations and ending at
2,265.4 Elo. Later points from its noisy terminal evaluation interval are excluded. Under those declared boundaries,
the final recipe ends **106.8 Elo higher in half a day less training**, and **348.2 Elo above the early baseline's
terminal observation**.

![64-search ladder Elo across five chess training campaigns](../showcase/chess-ladder-progress.svg)

Figure 7.1: Raw ladder observations are shown faintly beneath centered seven-point means. The final recipe is cut at
2.5 days and the previous four-day baseline at 3.0 days. The tracked
[publication input](../evidence/final-chess-20260923/ladder-elo-report-trimmed.json) records those rules and retains
the source-series identities; internal run labels do not appear in the figure.

That stitched curve is not the entire compute history. It deliberately excludes two reverted branches while retaining
their raw time in the source export:

1. An early INT8 conversion collapse was detected after one ladder observation and reverted.
2. A from-scratch capacity increase was promoted because training losses were compared across candidates that had
   seen different numbers of examples. The larger candidate was still roughly 270 Elo weaker. Training returned to
   the last valid checkpoint and removed 4,040,112 contaminated replay rows.

The second incident changed the method, not merely the operational state. Promotion now uses a match between deployed
artifacts rather than training-loss parity, and capacity growth starts from a function-preserving widening of the
parent. The corrected larger model reached parity but did not break through the plateau. This is evidence against
capacity being the immediate bottleneck under the tested recipe; it is not evidence that larger networks cannot help
under a different learning-rate, target-quality, or replay regime.

The selected checkpoint records 408,500 completed optimizer steps. With the configured global batch of 2,048 this
corresponds to **836,608,000 training presentations**. Final games, admitted positions, replay occupancy, and
resume-reconciled reuse remain to be derived from the archives.

The narrow cost attached to the selected checkpoint is **$43.20**, calculated as 60 accepted-lineage hours at
`$0.72/h`. It excludes the reverted work, later growth experiment, distillation, evaluation, and idle rental time.
Until total spend is reconciled, it must not be described as the project's total compute cost.

## Strength across four decades of search

The terminal protocol used 100 games per row from 50 colour-swapped opening pairs against single-threaded Stockfish
13 at fixed node limits. For each search budget, the reported rating is the opponent rung whose score lies closest to
0.500.

| Search budget | Headline score and anchor | Benchmark Elo (95% CI) | Increment |
| ---: | --- | ---: | ---: |
| Policy only | 0.440 vs 1,000 nodes | **1,658 [1,597, 1,717]** | -- |
| 100 | 0.480 vs 10,000 nodes | **2,456 [2,393, 2,518]** | +798 |
| 1,000 | 0.450 vs 50,000 nodes | **2,925 [2,875, 2,974]** | +469 |
| 10,000 | 0.520 vs 100,000 nodes | **3,114 [3,063, 3,166]** | +189 |
| 100,000 | 0.530 vs 200,000 nodes | **3,251 [3,206, 3,297]** | +137 |

Search therefore adds 1,593 benchmark Elo from policy-only play to the deepest measured condition, with diminishing
returns at each decade. The 100,000-search estimate is unusually well anchored: an independent 100,000-node opponent
gives 3,247 Elo, only four points below the 200,000-node result. That local agreement supports the top headline, but
does not validate extrapolation beyond the measured anchors.

The complete ten-row matrix, including W/D/L and archive-capture status, is in the
[result record](../results/final-chess-run.md#terminal-evaluation-protocol).

## Two protocol effects that matter

### Weak opponents bias low-budget estimates

At 100, 1,000, 10,000, and 100,000 searches, the harder opponent rung reads 128, 102, 46, and 4 Elo higher than the
easier rung. The shrinking gap is consistent with draw distortion against weak opposition. It is a serious concern at
low budgets but immaterial to the deepest result.

### Parallelism buys time by spending strength

The headline curve is an operating curve, not a pure search-budget ablation: it uses one parallel search at 100 and
1,000 searches, four at 10,000, and sixteen at 100,000. A controlled 1,000-search sweep measured 2,823 Elo with one
parallel search, 2,804 with four, and 2,778 with sixteen. Four-way parallelism therefore cost 19 Elo; sixteen-way cost
45 Elo.

The operational timing definition reports approximately 5.3x speedup for four-way and 13.9x for sixteen-way
parallelism. The result manifests' broader aggregate-duration fields imply smaller 4.8x and 10.1x ratios. Both are
preserved pending a final timing-definition choice. At only 100 searches, sixteen-way parallelism cost 235 Elo, so
parallelism cannot be changed silently across a compute curve.

## What the student establishes—and what it does not

The completed first distilled model has **470,295 parameters**, 13.4 times fewer than the teacher. At 10,000
searches it reached **2,683 Elo [2,626, 2,738]** against its closest anchor. The teacher reaches 3,114 under the same
nominal search count, although the inference backends differ: the student used bfloat16-trained TorchScript while the
teacher used INT8 TensorRT. The result demonstrates substantial compression, but it is not a controlled
architecture-only comparison.

A longer student run and its queued matches were unfinished at the cutoff. Held-out loss had crossed above training
loss early in that run, but the gap alone cannot establish memorisation or strength regression. Its evaluation must
be added as a separate experiment rather than replacing the completed first student.

## Figures still required

The numerical result is ready; the visual account is not. Publication still requires:

1. Raw and smoothed final-lineage ladder Elo against both stitched and raw time, with discarded intervals visible.
2. Total, policy, WDL, and auxiliary losses alongside learning rate and optimizer step.
3. Games, fresh positions, training presentations, replay occupancy/age, and trainer/self-play throughput.
4. Quantization fidelity, backend changes, capacity-growth attempts, and other material interventions.
5. A cost view that separates accepted training, discarded work, distillation, evaluation, and idle rental time.

The root README should remain unchanged until the post-pull result directories are fetched, the result tables are
checksum-complete, and the headline figures are generated from committed compact inputs.
