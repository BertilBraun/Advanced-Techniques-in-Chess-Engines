# Final chess run

> **Status: training complete; terminal teacher matrix complete; evidence capture partially pending.** The selected model
> and all ten teacher evaluation rows are now known. Three result directories were produced after the local evidence
> pull and must be re-fetched before publication. A longer distilled-student experiment remains in progress.

The complete readable recipe remains
[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). The compact evidence index is
[`documentation/evidence/final-chess-20260923`](../evidence/final-chess-20260923/README.md); it records the archive
hashes, model hashes, protocol, capture status, and machine-readable result table without putting the large archives
in Git.

## Result in one view

The reported checkpoint is a **14-block, 160-channel scaled-post-activation convolutional network** with global
context conditioning and a chess from-to attention policy head. It has **6,315,378 parameters** and was evaluated
through its pre-fold **INT8 QAT TensorRT** deployment artifact.

| Search budget per move | Headline benchmark Elo | 95% CI | Gain over previous budget |
| ---: | ---: | ---: | ---: |
| Policy only | **1,658** | [1,597, 1,717] | -- |
| 100 | **2,456** | [2,393, 2,518] | +798 |
| 1,000 | **2,925** | [2,875, 2,974] | +469 |
| 10,000 | **3,114** | [3,063, 3,166] | +189 |
| 100,000 | **3,251** | [3,206, 3,297] | +137 |

This is protocol-specific benchmark Elo calibrated against the fixed-node Stockfish 13 anchor curve. It is not a
FIDE rating and is not directly comparable with CCRL, online-server, or unrestricted contemporary-engine ratings.
The curve also mixes search parallelism, as documented below; it should not be interpreted as a controlled
single-variable scaling law.

## Selected model and evidence

The stitched multi-rung ladder was strongest in a window around generations 1020--1080. **Generation 1026** was
selected because it was the last checkpoint in that window retained with the complete training, optimizer, QAT,
ONNX, and TensorRT artifact set. A later, larger model reached parity but did not establish a stronger plateau.

| Field | Final value |
| --- | --- |
| Checkpoint | Generation 1026 |
| Architecture | 14 residual blocks, 160 channels, global pooling every second block |
| Policy head | Chess from-to attention, key size 128 |
| Parameters | 6,315,378 |
| Completed optimizer steps in checkpoint QAT state | 408,500 |
| Training presentations at configured global batch 2,048 | 836,608,000 |
| Training precision | bfloat16 with pre-fold INT8 QAT |
| Evaluation deployment | TensorRT INT8, batch 64 |
| Model SHA-256 | `c92a363b041a18d0ef93b852ac1c6d58716ae9a22b4e62d543de297c4ec5f904` |
| INT8 ONNX SHA-256 | `d634abacae3c874eac6ded89f6af861eb81b509da638b5ad710587b1a08be658` |
| TensorRT engine SHA-256 | `357652b119b4e4570127587bf0a04757ece6c01dd03abe3a0a905254249d84a5` |

The checkpoint manifest and archive-level hashes are listed in the
[evidence index](../evidence/final-chess-20260923/README.md#selected-checkpoint).

## Training trajectory and excluded work

The operator recap summarizes the stitched ladder as **216 observations over 3.0 effective days**, rising from 798
to approximately 2,380 and peaking at **2,407.6**. The later delivered JSON contains 229 multi-rung observations
through 76.33 hours, while preserving the same peak. That count and endpoint difference remains an explicit figure-
generation reconciliation item. The trajectory reached
the previous four-day baseline's 2,388.6 peak in approximately **2.5 effective days**, rather than 4.04 days.

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
| Policy only | 1 | 1,000 (1,700) | 32/24/44 | **0.440** | **1,658 [1,597, 1,717]** | Complete; awaiting re-fetch |
| Policy only | 1 | 2,000 (1,890) | 16/22/62 | 0.270 | 1,717 [1,645, 1,778] | Complete; awaiting re-fetch |
| 100 | 1 | 5,000 (2,220) | 51/28/21 | 0.650 | 2,328 [2,271, 2,391] | Captured and checksum-covered |
| 100 | 1 | 10,000 (2,470) | 39/18/43 | **0.480** | **2,456 [2,393, 2,518]** | Captured and checksum-covered |
| 1,000 | 1 | 20,000 (2,700) | 47/40/13 | 0.670 | 2,823 [2,772, 2,880] | Captured and checksum-covered |
| 1,000 | 1 | 50,000 (2,960) | 21/48/31 | **0.450** | **2,925 [2,875, 2,974]** | Captured and checksum-covered |
| 10,000 | 4 | 50,000 (2,960) | 45/40/15 | 0.650 | 3,068 [3,016, 3,124] | Captured and checksum-covered |
| 10,000 | 4 | 100,000 (3,100) | 30/44/26 | **0.520** | **3,114 [3,063, 3,166]** | Captured and checksum-covered |
| 100,000 | 16 | 100,000 (3,100) | 51/38/11 | 0.700 | 3,247 [3,195, 3,306] | Captured and checksum-covered |
| 100,000 | 16 | 200,000 (3,230) | 25/56/19 | **0.530** | **3,251 [3,206, 3,297]** | Complete; awaiting re-fetch |

The two 100,000-search estimates agree within four Elo despite using independent opponent anchors. That agreement is
evidence that the anchor curve remains locally consistent at the top of the measured range; it is not a general
validation outside these two rungs.

### Easy-rung bias

The harder opponent rung estimates a higher model rating at every budget, but the difference falls from **128 Elo**
at 100 searches to **102**, **46**, and **4 Elo** at 1,000, 10,000, and 100,000 searches. This pattern is consistent
with draw distortion against easy opponents at low model budgets. It is not material to the 100,000-search headline.

### Parallel-search trade-off

At 1,000 searches against the same 20,000-node opponent, one, four, and sixteen parallel searches scored 0.670,
0.645, and 0.610, corresponding to 2,823, 2,804, and 2,778 Elo. Thus four-way parallelism cost **19 Elo** and
sixteen-way parallelism cost **45 Elo** in this sweep.

The operational recap reports match wall times of 18.1, 3.4, and 1.3 minutes, or approximately **5.3x** speedup for
four-way and **13.9x** for sixteen-way parallelism. The archived result manifests record slightly broader aggregate
durations of 18.5, 3.8, and 1.8 minutes. Publication should preserve the named timing definition rather than blend
the two. A separate 100-search comparison found a much larger **235-Elo** penalty for sixteen-way parallelism.

## Distilled student: separate, not part of the teacher result

The completed first student is a 6-block, 64-channel convolutional model with a key-size-64 from-to attention head:
**470,295 parameters**, or **13.4x fewer** than the teacher. It trained for 36,621 steps (approximately 7.5 epochs)
on the 20-million-row replay buffer in bfloat16 without QAT and was evaluated through TorchScript. At 10,000
searches it scored 0.475 against a 20,000-node opponent, corresponding to **2,683 Elo [2,626, 2,738]**. Against the
10,000-node rung it scored 0.670, corresponding to 2,593 Elo [2,534, 2,660]. These result directories await re-fetch.

A second student run targeting 110,000 steps (approximately 23 epochs) and its queued evaluation were **still in
progress** at the evidence cutoff. Its early held-out loss crossed above training loss by step 6,000 (1.9893 versus
1.9800), so its outcome must be reported separately once complete; no strength conclusion is drawn from that partial
signal.

## Publication gate still open

Before the result moves into the root README, complete these items:

- re-fetch and checksum all evaluation result directories created after the current evidence pull;
- reconcile total self-play games, accepted positions, replay occupancy, and resume-safe training counters;
- distinguish accepted-lineage cost from discarded-work, evaluation, distillation, and total rental spend;
- reconcile the delivered ladder export with the recap's observation count and duration, then generate the
  cross-campaign ladder figure with descriptive model labels;
- generate the loss, learning-rate, throughput, replay, quantization-fidelity, and transition figures from the
  checksum-covered archives;
- freeze the longer-student result separately if it completes.

## Cross-campaign figure specification

The root README and report should share a plot of the matched 64-search ladder over effective training time for five
descriptively labelled campaigns: the early baseline, the first major architecture revision, the previous four-day
baseline, its later successor, and the final training lineage. Internal run identifiers belong only in the figure's
provenance sidecar. The final series must preserve raw time as well as stitched accepted-lineage time, show resume and
discard boundaries, and overlay smoothing without replacing raw observations.
