# Appendix B. Evaluation detail

## Fixed-node match protocol

Every teacher row used 100 games from 50 paired openings, once from each colour. Stockfish 13 used one thread and
1,024 MiB hash. Its fixed node limits were assigned the historical benchmark-Elo anchors of 1,700 at 1,000 nodes,
1,890 at 2,000, 2,220 at 5,000, 2,470 at 10,000, 2,700 at 20,000, 2,960 at 50,000, 3,100 at 100,000, and
3,230 at 200,000 [9]. The displayed 95% intervals bootstrap match outcomes while holding these graph-read anchors
fixed; they exclude calibration uncertainty. The headline for each candidate budget uses the tested opponent whose
observed score is closest to 0.5. Searched games used the INT8 TensorRT artifact; policy-only games used the
corresponding float TorchScript export.

| Model searches | Parallel | Opponent nodes | W/D/L | Score | Benchmark Elo (95% CI) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Policy only | -- | 1,000 | 32/24/44 | 0.440 | **1,658 [1,608, 1,710]** |
| Policy only | -- | 2,000 | 16/22/62 | 0.270 | 1,717 [1,638, 1,790] |
| 100 | 1 | 5,000 | 51/28/21 | 0.650 | 2,328 [2,276, 2,384] |
| 100 | 1 | 10,000 | 39/18/43 | 0.480 | **2,456 [2,400, 2,512]** |
| 1,000 | 1 | 20,000 | 47/40/13 | 0.670 | 2,823 [2,774, 2,873] |
| 1,000 | 1 | 50,000 | 21/48/31 | 0.450 | **2,925 [2,875, 2,977]** |
| 10,000 | 4 | 50,000 | 45/40/15 | 0.650 | 3,068 [3,023, 3,120] |
| 10,000 | 4 | 100,000 | 30/44/26 | 0.520 | **3,114 [3,065, 3,163]** |
| 100,000 | 16 | 100,000 | 51/38/11 | 0.700 | 3,247 [3,192, 3,305] |
| 100,000 | 16 | 200,000 | 25/56/19 | 0.530 | **3,251 [3,206, 3,297]** |

The two deepest-search anchors agree within four Elo; the shallow pairs disagree more substantially. The ladder is
an attainable operating curve, not a constant-parallelism search ablation.

## Matched-estimator comparison

The cross-campaign headline does not subtract the displayed 64-search curve endpoints. A retrospective plateau
comparison on the same three-rung estimator placed the previous baseline at 2,283.9 and the final recipe at
2,358.0 benchmark Elo, a difference of +74.1. Restricting both plateaus to the clean report-figure windows (3.0
and 2.5 days) gave +75.8. The earlier baseline's original series used a single-rung fit. A +2.7 Elo estimator
transfer, summarized from 35 paired observations in the operator recap, brings it onto the three-rung scale. The
underlying paired-observation table is not preserved with the report; approximately ±15 Elo is a sensitivity
allowance for this transfer, not a bootstrap interval. This limitation does not apply to the terminal match
intervals above, which come from their game records.

## Distilled student controls

Both students had 470,295 parameters and trained on the same frozen 20-million-row replay snapshot without QAT.
They were evaluated with TorchScript and four parallel searches. The 36,621-step student scored 31/33/36 against
20,000 Stockfish nodes at 10,000 searches: 2,683 [2,637, 2,731] benchmark Elo. The 110,000-step student scored
35/29/36 under the same protocol: 2,697 [2,640, 2,753]. At 100,000 searches, the longer student scored 59/28/13
against 20,000 nodes for 2,873 [2,819, 2,935]. That deep point is unbracketed because the planned 50,000-node
match was skipped; it is not an architecture-only comparison with the INT8 teacher.
