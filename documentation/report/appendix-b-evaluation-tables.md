# Appendix B. Evaluation detail

This appendix specifies the match protocol and rating calculation underlying the results in Chapters 2 and 8.

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
