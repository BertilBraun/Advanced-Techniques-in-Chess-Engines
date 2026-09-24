# Final chess run

> **Status: training and evaluation complete; result evidence captured.** The selected teacher, all ten headline-matrix
> rows, the parallel-search sweep, and both student experiments are checksum-covered by the local evidence bundles.

The complete readable recipe remains
[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). The compact evidence index is
[`documentation/evidence/final-chess-20260923`](../evidence/final-chess-20260923/README.md); it records the archive
hashes, model hashes, protocol, capture status, and machine-readable result table without putting the large archives
in Git.

## Result in one view

The reported checkpoint is a **14-block, 160-channel scaled-post-activation convolutional network** with global
context conditioning and a chess from-to attention policy head. It has **6,315,378 parameters** and was evaluated
through its pre-fold **INT8 QAT TensorRT** deployment artifact for searched play. Policy-only evaluation used the
matching float TorchScript export because it does not invoke the native search service.

| Search budget per move | Headline benchmark Elo | 95% CI | Gain over previous budget |
| ---: | ---: | ---: | ---: |
| Policy only | **1,658** | [1,608, 1,710] | -- |
| 100 | **2,456** | [2,400, 2,512] | +798 |
| 1,000 | **2,925** | [2,875, 2,977] | +469 |
| 10,000 | **3,114** | [3,065, 3,163] | +189 |
| 100,000 | **3,251** | [3,206, 3,297] | +137 |

This is protocol-specific benchmark Elo calibrated against
[Marco Meloni's fixed-node Stockfish 13 curve](https://www.melonimarco.it/en/2021/03/08/stockfish-and-lc0-test-at-different-number-of-nodes/),
which links Stockfish through Fruit 2.2.1 to the historical SSDF scale. It is not a FIDE rating and is not directly
comparable with CCRL, online-server, or unrestricted contemporary-engine ratings. The curve also mixes search
parallelism, as documented below; it should not be interpreted as a controlled single-variable scaling law.

## Selected model and evidence

The stitched multi-rung ladder was strongest in a window around generations 1020--1080. **Generation 1026** was
selected because it was the last checkpoint in that window retained with the complete training, optimizer, QAT,
ONNX, and TensorRT artifact set. A later, larger model reached parity but did not establish a stronger plateau during
its limited continuation.

| Field | Final value |
| --- | --- |
| Checkpoint | Generation 1026 |
| Architecture | 14 residual blocks, 160 channels, global pooling every second block |
| Policy head | Chess from-to attention, key size 128 |
| Parameters | 6,315,378 |
| Completed optimizer steps in checkpoint QAT state | 408,500 |
| Training presentations at configured global batch 2,048 | 836,608,000 |
| Training precision | bfloat16 with pre-fold INT8 QAT |
| Searched evaluation deployment | TensorRT INT8, batch 64 |
| Policy-only evaluation deployment | Float TorchScript, SHA-256 `1cb9fe4b23c91e4162097c7425b397516bb28dd2560cbb066ec8422548961816` |
| Model SHA-256 | `c92a363b041a18d0ef93b852ac1c6d58716ae9a22b4e62d543de297c4ec5f904` |
| INT8 ONNX SHA-256 | `d634abacae3c874eac6ded89f6af861eb81b509da638b5ad710587b1a08be658` |
| TensorRT engine SHA-256 | `357652b119b4e4570127587bf0a04757ece6c01dd03abe3a0a905254249d84a5` |

The checkpoint manifest and archive-level hashes are listed in the
[evidence index](../evidence/final-chess-20260923/README.md#selected-checkpoint).
The matching INT8 ONNX is also preserved in the public
[Hugging Face repository at immutable revision `dc8fcccc`](https://huggingface.co/BertilBraun/alphazero-chess/blob/dc8fccccb67ab5ec9e36267a165a9700b7dbf55f/production/final-generation-1026/model.int8.onnx).
Its LFS SHA-256 matches the frozen evidence. The repository's model card, `latest` aliases, and checksum index still
describe the preceding public checkpoint, so the final artifact is published but the public release metadata is not
yet fully synchronized. That release rework is in progress; the final model card and live deployment are expected to
reference generation 1026 before publication.

## Training trajectory and excluded work

The report-scoped stitched ladder contains **180 observations through exactly 2.5 effective days**, rising from
798 to **2,372.2** at the cutoff and peaking at **2,407.6**. Later observations belong to experiments that did not
improve the accepted result and are retained only in the untrimmed source evidence. The comparison trims the previous
four-day baseline at exactly 3.0 days, before its noisy terminal interval, where it ends at 2,265.4 Elo. Those plotted
endpoints differ by 106.8 Elo, but they must not be used as the cross-campaign strength estimate: the historical curve
used a single-rung fit while the final curve used a three-rung bracketed fit.

The estimator-matched retrospective compares plateau windows on the three-rung estimator. It places the previous
baseline at 2,283.9 Elo and the final recipe at 2,358.0 Elo, a gain of **74.1 Elo**, with an approximately **±15 Elo
transfer/sensitivity allowance**. That allowance is not a game-level confidence interval. The defensible public
summary is therefore **about +74 Elo under a matched estimator**, not the endpoint difference or either curve's
single highest observation.

![64-search ladder Elo across five chess training campaigns](../showcase/chess-ladder-progress.svg)

The [trimmed publication input](../evidence/final-chess-20260923/ladder-elo-report-trimmed.json) is derived from the
checksum-covered source export. It records the two cutoffs and their reasons alongside descriptive series labels.

Two failed branches are excluded from that curve but remain part of the provenance:

- An early INT8 collapse produced one 1,722-Elo observation before training reverted to the last valid checkpoint.
- A from-scratch 19-block, 176-channel candidate was promoted using an invalid training-loss comparison, lost about
  270 Elo, and was reverted after eight ladder observations. The replay buffer was pruned by 4,040,112 rows so the
  failed candidate's games could not contaminate the resumed lineage.

The time rebasing removes those branches from the accepted-lineage x-axis; it does **not** pretend the discarded
compute did not happen. The delivered ladder export retains raw timestamps alongside stitched time so both views can
be reproduced.

The selected checkpoint lies at 60 hours of accepted-lineage time. At the recorded `$0.72/h` node price this is
**$43.20**. That number is a deliberately narrow training-cost measure. It excludes discarded branches, later model
growth, distillation, terminal evaluation, and any rental idle time, so it must not be reported as total project or
total rental spend.

## Terminal evaluation protocol

Each row used **100 games from 50 paired openings**, with every opening played from both colours. The opponent was
Stockfish 13 with one thread and 1,024 MiB hash at a fixed node budget from the established anchor curve. For each
model budget, the headline is the rung whose score is closest to 0.500; this minimizes extrapolation and draw-driven
distortion. Both rungs are retained below.

| Model searches | Parallel searches | Opponent nodes (anchor Elo) | W/D/L | Score | Model Elo (95% CI) | Evidence status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Policy only | -- | 1,000 (1,700) | 32/24/44 | **0.440** | **1,658 [1,608, 1,710]** | Checksum-covered |
| Policy only | -- | 2,000 (1,890) | 16/22/62 | 0.270 | 1,717 [1,638, 1,790] | Checksum-covered |
| 100 | 1 | 5,000 (2,220) | 51/28/21 | 0.650 | 2,328 [2,276, 2,384] | Checksum-covered |
| 100 | 1 | 10,000 (2,470) | 39/18/43 | **0.480** | **2,456 [2,400, 2,512]** | Checksum-covered |
| 1,000 | 1 | 20,000 (2,700) | 47/40/13 | 0.670 | 2,823 [2,774, 2,873] | Checksum-covered |
| 1,000 | 1 | 50,000 (2,960) | 21/48/31 | **0.450** | **2,925 [2,875, 2,977]** | Checksum-covered |
| 10,000 | 4 | 50,000 (2,960) | 45/40/15 | 0.650 | 3,068 [3,023, 3,120] | Checksum-covered |
| 10,000 | 4 | 100,000 (3,100) | 30/44/26 | **0.520** | **3,114 [3,065, 3,163]** | Checksum-covered |
| 100,000 | 16 | 100,000 (3,100) | 51/38/11 | 0.700 | 3,247 [3,192, 3,305] | Checksum-covered |
| 100,000 | 16 | 200,000 (3,230) | 25/56/19 | **0.530** | **3,251 [3,206, 3,297]** | Checksum-covered |

The two 100,000-search estimates agree within four Elo despite using independent opponent anchors. That agreement is
evidence that the anchor curve remains locally consistent at the top of the measured range; it is not a general
validation outside these two rungs.

### Opponent-rung disagreement

The harder opponent rung estimates a higher model rating at every budget, but the difference falls from **128 Elo**
at 100 searches to **102**, **46**, and **4 Elo** at 1,000, 10,000, and 100,000 searches. In an ideal transitive Elo
model the two rungs would agree. Draw behavior against the easier opponent, calibration error, matchup effects, and
sampling noise are possible contributors, but the project did not isolate them. Selecting the rung whose score is
closest to 0.5 reduces extrapolation; the disagreement is not material to the 100,000-search headline.

### Parallel-search trade-off

At 1,000 searches against the same 20,000-node opponent, one, four, and sixteen parallel searches scored 0.670,
0.645, and 0.610, corresponding to 2,823, 2,804, and 2,778 Elo. Thus four-way parallelism cost **19 Elo** and
sixteen-way parallelism cost **45 Elo** in this sweep.

The operational recap reports match wall times of 18.1, 3.4, and 1.3 minutes, or approximately **5.3x** speedup for
four-way and **13.9x** for sixteen-way parallelism. The archived result manifests record slightly broader aggregate
durations of 18.5, 3.8, and 1.8 minutes. Publication should preserve the named timing definition rather than blend
the two. A separate 100-search comparison found a much larger **235-Elo** penalty for sixteen-way parallelism. The
budget dependence is important, but these two budgets do not determine a universal “safe parallelism” curve; the
measured result is a strength/latency trade-off at specific operating points.

## Distilled student: separate, not part of the teacher result

Both students use the same 6-block, 64-channel convolutional architecture with a key-size-64 from-to attention head:
**470,295 parameters**, or **13.4x fewer parameters** than the teacher. They trained on the same frozen 20-million-row
replay snapshot in bfloat16 without QAT and were evaluated through TorchScript with four parallel searches.

| Training | Student searches | Opponent nodes | W/D/L | Score | Model Elo (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 36,621 steps / 7.500 epochs | 10,000 | 10,000 | 55/24/21 | 0.670 | 2,593 [2,533, 2,666] |
| 36,621 steps / 7.500 epochs | 10,000 | 20,000 | 31/33/36 | **0.475** | **2,683 [2,637, 2,731]** |
| 110,000 steps / 22.528 epochs | 10,000 | 20,000 | 35/29/36 | **0.495** | **2,697 [2,640, 2,753]** |
| 110,000 steps / 22.528 epochs | 100,000 | 20,000 | 59/28/13 | 0.730 | 2,873 [2,819, 2,935] |

Tripling the training moved the matched 10,000-search estimate by only **14 Elo**, far inside the overlapping
confidence intervals. The longer run ended with policy loss 1.8613 on the training split and 1.8815 on held-out data;
the roughly 0.020 gap had been flat since about step 60,000. This supports rapid capacity saturation for this student
and dataset, not a claim of catastrophic memorisation.

The 100,000-search student point is **unbracketed**: it beat the 20,000-node opponent decisively, while the planned
50,000-node match was deliberately skipped and recorded by a `SKIPPED` marker. It is therefore a lower anchor-based
estimate rather than a headline comparable in robustness to the teacher's two-rung 100,000-search result. Elo is an
interval scale, so the student and teacher ratings must not be compared as a percentage or ratio.

## Publication work still open

The verified result and headline ladder figure are ready for the root README. The full report and release still need:

- reconcile total self-play games, accepted positions, replay occupancy, and resume-safe training counters;
- distinguish accepted-lineage cost from discarded-work, evaluation, distillation, and total rental spend;
- generate the loss, learning-rate, throughput, replay, quantization-fidelity, and transition figures from the
  checksum-covered archives;
- finish the in-progress Hugging Face model-card, alias, checksum, and live-deployment update around the already
  hash-matched final ONNX;
- choose and publish the code and model/data licenses.

## Cross-campaign figure

The report figure compares the available 64-search ladder series over effective training time for five descriptively
labelled campaigns: the early baseline, first major architecture revision, previous four-day baseline, quantized
successor, and final recipe. Internal identifiers remain only in the provenance data. Curves use the project's established
bias-corrected 0.95 exponential moving average. The root README can reuse this SVG when the remaining publication
gate closes. The figure shows descriptive trajectories; the **+74.1 Elo** comparison above comes from the separate
matched-estimator plateau audit and must remain the quantitative cross-campaign headline.
