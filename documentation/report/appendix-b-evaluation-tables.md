# Appendix B. Evaluation detail

## Fixed-node match protocol

The ten teacher matches are shown in Table 1. Every row used 100 games from 50 paired openings, once from each colour. Stockfish 13 used one thread and
1,024 MiB hash. Its fixed node limits were assigned the historical benchmark-Elo anchors of 1,700 at 1,000 nodes,
1,890 at 2,000, 2,220 at 5,000, 2,470 at 10,000, 2,700 at 20,000, 2,960 at 50,000, 3,100 at 100,000, and
3,230 at 200,000 [9]. The displayed 95% intervals bootstrap match outcomes while holding these graph-read anchors
fixed; they exclude calibration uncertainty. The headline for each candidate budget uses the tested opponent whose
observed score is closest to 0.5. Searched games used the INT8 TensorRT artifact; policy-only games used the
corresponding float TorchScript export.

For score `s` against an opponent anchor `R`, benchmark Elo is `R + 400 log10(s / (1 - s))`. Each 95% interval
resamples the 50 colour-swapped opening pairs 10,000 times, then transforms the 2.5th and 97.5th percentiles of
the resulting match scores with the anchor held fixed.

The two deepest-search anchors agree within four Elo. At 100, 1,000, 10,000, and 100,000 searches, the harder
opponent rung reads 128, 102, 46, and 4 Elo higher than the easier rung. Ideal transitive Elo would give the same
estimate from both. Draw behavior, anchor calibration error, matchup effects, and sampling noise are possible
contributors, but the project did not isolate the cause. Choosing the score closest to 0.5 limits extrapolation.
The ladder is an attainable operating curve, not a constant-parallelism search ablation.

## Matched-estimator comparison

The cross-campaign headline does not subtract the displayed 64-search curve endpoints. A retrospective plateau
comparison on the same three-rung estimator placed the previous baseline at 2,283.9 and the final recipe at
2,358.0 benchmark Elo, a difference of +74.1. Restricting both plateaus to the clean report-figure windows (3.0
and 2.5 days) gave +75.8. The earlier baseline's original series used a single-rung fit. A +2.7 Elo estimator
transfer, summarized from 35 paired observations in the operator recap, brings it onto the three-rung scale. The
underlying paired-observation table is not preserved with the report; approximately ±15 Elo is a sensitivity
allowance for this transfer, not a bootstrap interval. This limitation does not apply to the terminal match
intervals above, which come from their game records.

For orientation only, the preceding four-day baseline is plotted through 3.0 days, retaining 143 observations and
ending at 2,265.4 Elo before its noisy terminal interval; subtracting this from the final endpoint gives 106.8 Elo.
The final endpoint also sits 348.2 Elo above the early baseline's terminal observation. Neither visual subtraction
measures cross-campaign strength because the original estimators differ.

## Distilled student controls

Both students had 470,295 parameters and trained on the same frozen 20-million-row replay snapshot without QAT.
They were evaluated with TorchScript and four parallel searches. The 36,621-step student scored 31/33/36 against
20,000 Stockfish nodes at 10,000 searches: 2,683 [2,637, 2,731] benchmark Elo. The 110,000-step student scored
35/29/36 under the same protocol: 2,697 [2,640, 2,753]. At 100,000 searches, the longer student scored 59/28/13
against 20,000 nodes for 2,873 [2,819, 2,935]. That deep point is unbracketed because the planned 50,000-node
match was skipped; it is not an architecture-only comparison with the INT8 teacher.
