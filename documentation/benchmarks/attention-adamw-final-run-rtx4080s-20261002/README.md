# Final self-play run with the T1-shaped network — results

As of **2026-10-02**. The self-play run planned in
[chess-attention-final-run-plan-20260929.md](../../plan/chess-attention-final-run-plan-20260929.md), with AdamW
instead of SGD, on one 8x RTX 4080 SUPER node (200 W cap, 241 GiB RAM, Vast.ai instance 53481269, $1.60/h). It ran
52 hours of run time from 2026-09-30 07:13 UTC to 2026-10-02 12:07 UTC and was stopped by the project owner once its
ladder had been flat for about eight hours across two learning-rate schedules.

## Short answer

- **It passed the convolutional lineage.** Ladder Elo at 64 searches settled near **2,480**, against the CNN
  lineage's plateau near 2,360 and peak of 2,407.6. It reached the CNN plateau after about 28 hours of run time.
  The CNN numbers were measured on 8x RTX 4070 SUPER, so the time axis is not like-for-like; the Elo axis is.
- **It then plateaued.** From about generation 700 the ladder stayed within 2,408-2,524, and the 10x192's policy
  loss had been flat since generation ~500.
- **A learning-rate warm restart changed the loss, not the strength.** Raising the rate from 4.2e-5 to 1.2e-4 at
  generation 813 lowered the training loss from 2.871 to 2.850 over 83 generations, but the final checkpoint scored
  **0.5125 [0.463, 0.563]** (+8.7 Elo [−26, +44]) over 200 games against the checkpoint from before the restart.
- **At 100,000 searches it is about 87 Elo stronger than the CNN.** The final checkpoint scored **0.650 [0.590, 0.705]**
  (39/52/9) against Stockfish 13 at 200,000 nodes, **3,338 [3,293, 3,381]**, where the CNN's
  selected checkpoint scored 0.530 (3,251 [3,206, 3,297]) under the same protocol. The rows differ in parallel
  searches (8 against 16), serving precision (float16 against INT8) and search-value discounting (none against
  discounted); see the match section.

## Run segments

All segments share the save path `py/training_data/production/vast-chess-8gpu-final-attention-adamw`, so the replay,
the progressive state and the evaluation history carry across resumes.

| Segment | Revision | Configuration SHA-256 (prefix) | Generations | Change |
| --- | --- | --- | --- | --- |
| `vast-chess-8gpu-final-attention-adamw` | `539e51fc` | `880a48b22915` | 0-183 | 8x160 active, 10x192 candidate; AdamW 1e-3, geometric decay from generation 100 |
| `-resume-183` | `e611eb94` | `c9e92464b557` | 183-211 | after a host outage; decay from generation 0 (1e-3 → 2e-5 over 0-1000) |
| `-resume-211` | `7cc6254a` | `7eba2d564e59` | 211-270 | candidate start forced (threshold 1,000 Elo/h) |
| `-resume-270` | `61e7dc38` | `5d54e6a505a5` | 270-290 | candidate catch-up rate 5e-4 → 1e-4 geometric |
| `-resume-290` | `b822f9e3` | `95795f5cad45` | 290-666 | promotion gate 0.40 once; 10x192 promoted at generation 294 |
| `-resume-ladder-20k` | `c8379e9c` | `7c0d5e7ecce7` | 666-813 | 20,000-node rung added to the searched ladder |
| `-resume-warm-restart` | `81f2852e` | `aa6604774018` | 813-896 | rate 1.2e-4 held to 863, geometric to 2e-5 at 1113; stopped at 896 |

## Ladder trajectory

Ladder Elo at 64 searches, averaged over four hours of run time. The bracket was 3,000/5,000/10,000 nodes until
generation 666 and 5,000/10,000/20,000 after it; the 10,000-node rung is in both, so its single-rung figure is the
like-for-like column across the change.

| Run time | Bracketed fit | 10,000-node rung alone | Policy only (1 search) |
| ---: | ---: | ---: | ---: |
| 0-4 h | 1,367 | 1,380 | 1,019 |
| 4-8 h | 2,037 | 2,031 | 1,396 |
| 8-12 h | 2,209 | 2,200 | 1,604 |
| 12-16 h | 2,264 | 2,244 | 1,660 |
| 16-20 h | 2,300 | 2,299 | 1,708 |
| 20-24 h | 2,310 | 2,298 | 1,714 |
| 24-28 h | 2,342 | 2,373 | 1,722 |
| 28-32 h | 2,364 | 2,408 | 1,752 |
| 32-36 h | 2,398 | 2,438 | 1,789 |
| 36-40 h | 2,448 | 2,459 | 1,833 |
| 40-44 h | 2,460 | 2,454 | 1,859 |
| 44-48 h | 2,478 | 2,479 | 1,851 |
| 48-52 h | 2,488 | 2,469 | 1,860 |
| 52 h | 2,442 | 2,421 | 1,888 |

Under the old bracket the 3,000- and 5,000-node rungs held the fit about 40 below the 10,000-node figure; under the
new one the two agree, so about 20-30 of the step at the bracket change is the anchors, not strength.

## The plateau and the warm restart

From generation 400 the score against each Stockfish rung, 50-generation means:

| Generations | Rate | Policy loss | vs 5,000 | vs 10,000 | vs 20,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 400-449 | 1.9e-4 | 1.815 | 0.611 | 0.362 | — |
| 500-549 | 1.3e-4 | 1.797 | 0.665 | 0.414 | — |
| 600-649 | 8.7e-5 | 1.792 | 0.704 | 0.450 | — |
| 650-699 | 7.2e-5 | 1.793 | 0.746 | 0.483 | 0.259 |
| 750-799 | 4.9e-5 | 1.794 | 0.733 | 0.501 | 0.282 |
| 827-857 (warm restart, held 1.2e-4) | 1.2e-4 | — | 0.794 | 0.503 | 0.302 |

The 16 ladder points of the warm restart averaged 2,481 (2,408-2,524); the last three under the old schedule were
2,506, 2,517 and about 2,500. Training loss is measured on replay samples presented about three times each, so its
fall without a strength gain reads as a closer fit to the replay rather than a stronger network: the plateau does
not look schedule-bound.

## Matches after the stop

**Warm-restart end against its start**, `py/tools/distill_match.py`, equal nodes, 64 searches per move, one
parallel search, exploration 1.2533 (the evaluation's `auto` value at 64), 100 paired openings from
`py/reference/chess-elite-2025-11-balanced-4moves-200-v1-openings.json`, seed 20261002, TensorRT float16 on one GPU.
Generation 813's weights had been removed by checkpoint retention (only every sixth generation keeps them), so
generation 810, three generations earlier at the same rate, stands in for the end of the 20,000-node segment. Its
float ONNX was re-exported from `model_810.pt` into `/workspace/h2h-810`; the export reproduced the original artifact
byte for byte (SHA-256 `96ebb204…`).

| Student | Teacher | W/D/L (student) | Score (95%) | Elo difference (95%) |
| --- | --- | --- | --- | --- |
| generation 896 | generation 810 | 59/87/54 | 0.5125 [0.4625, 0.5625] | +8.7 [−26.1, +43.7] |

**Final checkpoint at 100,000 searches against Stockfish 13 at 200,000 nodes**, `py/tools/run_stockfish_gauntlet.py`,
8 parallel searches per game, exploration 3.0564 (`auto` at 100,000), the terminal protocol's 50 paired openings
(`chess-stockfish-8moves-v3-openings-v33.json`), 7 GPUs. The CNN's selected checkpoint was measured under the same
protocol with 16 parallel searches, INT8 serving and discounted search values.

One shard per GPU left the GPUs at 54-68% utilisation: each shard's driver held one CPU core at 100% while the node's
~123-core quota sat at a load of 7. Repeating devices (`--devices 0 0 1 1 …`) runs several drivers per GPU and is
the setting to use for the next deep match.

| Model | Model searches | Parallel | Opponent nodes (anchor) | W/D/L | Score | Model Elo (95%) |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| 10x192 attention, generation 896 | 100,000 | 8 | 200,000 (3,230) | 39/52/9 | **0.650** [0.590, 0.705] | **3,338 [3,293, 3,381]** |
| 14x160 CNN, selected checkpoint ([record](../../results/final-chess-run.md)) | 100,000 | 16 | 200,000 (3,230) | 25/56/19 | 0.530 | 3,251 [3,206, 3,297] |

Source revision `81f2852e`, configuration `aa660477…`, match seed 20460811, Stockfish 13 with one thread and
1,024 MiB hash, 95 minutes. Elo is the anchor plus the logistic difference of the score, the conversion that gives
the CNN row its 3,251. The attention row is a single rung at 0.650, further from an even score than the CNN's 0.530,
so it rests on more extrapolation along the anchor curve; a 500,000-node rung (3,350) would bracket it. As the
first player it scored 0.74, as the second 0.56. Eight-way rather than sixteen-way parallelism favours this row
somewhat; the CNN record measured sixteen-way at 45 Elo below one-way at 1,000 searches but has no measurement at
100,000.

## Cost

52 hours of run time at $1.60/h, about **$83** of node time for the AdamW run, excluding the earlier SGD attempt on
the 4070 SUPER node, the smokes and the post-stop matches.

## Evidence

Preserved with `run_control.sh preserve` on the node and copied to
`.codex-diagnostics/vast-chess-8gpu-final-attention-adamw-20261002/evidence-20261002.tar` (1.2 GB, SHA-256
`601c277996ce9a767f67f749c17396200ca390db83b9943b6cc911bcddfbe883`, with a `SHA256SUMS` inside): the complete
warm-restart archive (logs, TensorBoard, evaluations, checkpoint manifests, generation 896 weights and optimizer,
progressive models), the logs, TensorBoard and configurations of every earlier segment, both match results with
their launch and export scripts, and generation 810's weights, manifest and re-exported ONNX. The replay buffer
(about 19M samples) was not copied.
