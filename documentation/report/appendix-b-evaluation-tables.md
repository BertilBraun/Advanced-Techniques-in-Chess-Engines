# Appendix B. Evaluation detail

This appendix specifies the match protocol and rating calculation underlying the results in Chapters 2 and 8.

## Fixed-node match protocol

Each of the ten teacher matches in Table \ref{tab:02-methodology-and-evidence-1} comprises 100 games from 50 openings played with colours reversed.
The opening set is `chess-stockfish-8moves-v3-openings-v33.json` (SHA-256 prefix `490425ed0f466f55`).
Each opening supplies eight full moves (16 plies) before evaluation play. Games reaching the 300-ply evaluation
cap are recorded as draws and included in the fixed-opponent score.
Stockfish 13 used one thread and
1,024 MiB hash. Its fixed node limits were assigned the historical benchmark-Elo anchors of 1,700 at 1,000 nodes,
1,890 at 2,000, 2,220 at 5,000, 2,470 at 10,000, 2,700 at 20,000, 2,960 at 50,000, 3,100 at 100,000, and
3,230 at 200,000 [9]. These anchors were read from the published calibration curve. The reported rating at each
candidate budget uses the tested opponent against which the score is closest to 0.5. Searched games used the
INT8 TensorRT artifact; policy-only games used the
corresponding float TorchScript export.

Evaluation uses an exploration constant of 1.0, no root noise, and deterministic maximum-visit move selection;
policy-only play selects the highest-probability legal action. Each inference queue uses one worker, batch
capacity 64, and one outstanding batch. Teacher leaf parallelism is specified in Table \ref{tab:02-methodology-and-evidence-1}.

For score `s` against an opponent anchor `R`, benchmark Elo is `R + 400 log10(s / (1 - s))`. Each 95% interval
resamples the 50 colour-swapped opening pairs 10,000 times, then transforms the 2.5th and 97.5th percentiles of
the resulting match scores with the anchor held fixed. The intervals therefore quantify match sampling
uncertainty, excluding uncertainty in the historical calibration.

At 100, 1,000, 10,000, and 100,000 searches, the harder opponent implies a rating 128, 102, 46, and 4 Elo higher
than the easier opponent. The discrepancy narrows with search budget, reaching close agreement at the deepest
point. Draw behaviour, calibration error, matchup effects, and sampling variation could contribute; the matches
do not distinguish them. Selecting the score nearest 0.5 limits extrapolation from lopsided results.

## Student matches

The final student uses TorchScript and four parallel leaf searches. Both reported matches use the same
20,000-node Stockfish opponent, anchored at 2,700 benchmark Elo, with 100 games each.
At 10,000 searches its W/D/L count is 35/29/36, yielding 2,697 [2,640, 2,753] Elo.
At 100,000 searches it scores 59/28/13, yielding 2,873 [2,819, 2,935] Elo.
Intervals use the paired bootstrap described above. The deepest student result used one opponent rung.

## Thinking-time estimate

The under-five-second estimate for 100,000 searches on an RTX 4070 SUPER extrapolates from approximately
five seconds for 80,000 searches with TorchScript and approximately 1.8× faster inference with TensorRT.
Proportional scaling gives approximately 3.5 seconds:

```math
5\,\mathrm{s}\times\frac{100{,}000}{80{,}000}\div 1.8 \approx 3.5\,\mathrm{s}.
```

This assumes inference acceleration transfers sufficiently to overall search; 3.5 seconds is not a measured
latency. The earlier batched timing is a throughput-derived estimate rather than an isolated single-game test.
