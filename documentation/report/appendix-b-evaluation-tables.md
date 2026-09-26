# Appendix B. Evaluation detail

This appendix specifies the match protocol, rating calculation, cross-campaign adjustment, and student controls
underlying the results in Chapters 2 and 8.

## Fixed-node match protocol

Each of the ten teacher matches in Table 1 comprises 100 games from 50 openings played with colours reversed.
Stockfish 13 used one thread and
1,024 MiB hash. Its fixed node limits were assigned the historical benchmark-Elo anchors of 1,700 at 1,000 nodes,
1,890 at 2,000, 2,220 at 5,000, 2,470 at 10,000, 2,700 at 20,000, 2,960 at 50,000, 3,100 at 100,000, and
3,230 at 200,000 [9]. These anchors were read from the published calibration curve. The reported rating at each
candidate budget uses the tested opponent against which the score is closest to 0.5. Searched games used the
INT8 TensorRT artifact; policy-only games used the
corresponding float TorchScript export.

For score `s` against an opponent anchor `R`, benchmark Elo is `R + 400 log10(s / (1 - s))`. Each 95% interval
resamples the 50 colour-swapped opening pairs 10,000 times, then transforms the 2.5th and 97.5th percentiles of
the resulting match scores with the anchor held fixed. The intervals therefore quantify match sampling
uncertainty, excluding uncertainty in the historical calibration.

At 100, 1,000, 10,000, and 100,000 searches, the harder opponent implies a rating 128, 102, 46, and 4 Elo higher
than the easier opponent. The discrepancy narrows with search budget, reaching close agreement at the deepest
point. Draw behaviour, calibration error, matchup effects, and sampling variation could contribute; the matches
do not distinguish them. Selecting the score nearest 0.5 limits extrapolation from lopsided results.

## Matched-estimator comparison

The plateau comparison uses a common three-rung estimator, yielding 2,283.9 benchmark Elo for the previous
baseline and 2,358.0 for the final recipe, a difference of +74.1. Restriction to the plotted 3.0- and 2.5-day
windows gives +75.8. The baseline's original single-rung fit is adjusted by +2.7 Elo, based on a summary of
35 paired observations. Approximately ±15 Elo represents sensitivity to that transfer, not a bootstrap interval.

Figure 11 uses bias-corrected 0.95 exponential moving averages. The final curve retains 180 observations through
2.5 days; the preceding baseline retains 143 through 3.0 days, ending at 2,265.4 Elo before its unreliable late
interval. The raw endpoint differences are 106.8 Elo against that baseline and 348.2 against the early baseline.
Because the original estimators differ, the adjusted plateau comparison above is used for quantitative reporting.
Points beyond the plotted windows belong to subsequent experiments or the unreliable late interval.

## Distilled student controls

Both students had 470,295 parameters and trained on the same frozen 20-million-row replay snapshot without QAT,
separate from the teacher checkpoint's 16-million-row live window. They were evaluated with TorchScript and four
parallel searches, whereas the searched teacher used INT8 TensorRT. The 36,621-step student scored 31/33/36 against
20,000 Stockfish nodes at 10,000 searches: 2,683 [2,637, 2,731] benchmark Elo. The 110,000-step student scored
35/29/36 under the same protocol: 2,697 [2,640, 2,753]. At 100,000 searches, the longer student scored 59/28/13
against 20,000 nodes for 2,873 [2,819, 2,935]. Counts are wins/draws/losses and brackets denote 95% intervals.
The 100,000-search result uses this single opponent, without a second-rung check.
