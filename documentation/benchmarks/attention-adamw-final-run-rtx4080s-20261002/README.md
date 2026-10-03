# Final self-play run with the T1-shaped network — results

As of **2026-10-02**. The self-play run planned in
[chess-attention-final-run-plan-20260929.md](../../plan/chess-attention-final-run-plan-20260929.md), with AdamW
instead of SGD, on one 8x RTX 4080 SUPER node (200 W cap, 241 GiB RAM, Vast.ai instance 53481269, $1.60/h). It ran
56 hours of run time between 2026-09-30 07:13 UTC and 2026-10-02 17:48 UTC, to generation 950, and was stopped by
the project owner once its ladder had been flat for about twelve hours across three learning-rate regimes.

## Conclusion

The T1-shaped network moved the plateau, but it is a plateau again. Self-play with the 10x192 attention network
ended about 120 Elo above the convolutional lineage at 64 searches and about 87 above its selected checkpoint at
100,000 searches, close to the +150 that teacher distillation had measured for the architecture at equal searches.
The gain came from the architecture and the annealed AdamW schedule together; the run does not separate them.
Once the rate had decayed, three regimes (the plain decay, a 3x warm restart held for 50 generations, and its
decay) all held the ladder at 2,470-2,480 while training loss moved, and the warm restart's final checkpoint was
indistinguishable from the one before it. The limit now is not the schedule; whether it is network capacity or
the self-play loop at this size is untested. A 12x256 trained offline on the replay was considered and not run.

## The remaining gap to Lc0

On the same Stockfish 13 node calibration, Marco Meloni's tests put Lc0 at about 3,600 at 100,000 nodes. This run's
3,338 at 100,000 searches leaves a gap of roughly 250, within about 200-300 given the single-rung score (0.650) and
that Meloni's figure comes from his own pool, network and hardware. The convolutional model's 3,251 left about 350,
so this run closed about a quarter of it.

- **The gap is network quality, not search.** Search depth is already matched at 100,000. The Lc0 diagnostic ran T1,
  Lc0's 20M-parameter attention network, inside this project's unchanged search: 0.885 against 10,000 nodes at 64
  searches, about 365 above checkpoint 1026 and about 245 above this run's 10x192 (~+120). A T1-class network is
  worth about the whole gap in this search.
- **Self-play has nearly exhausted the 10x192.** The same network distilled from 47M T1-labelled positions reached
  about +150 over checkpoint 1026; self-play reached about +120. Labels from a far stronger engine bought only
  ~30 more at this size.
- **Lc0's advantage is scale, not a different algorithm.** Its main runs played hundreds of millions of self-play
  games over years on volunteer hardware and trained much larger networks; T1 itself was distilled from them. This
  run played about 1.2M games (~1,240 per generation, 950 generations). Lc0's training-target refinements, such as
  tablebase rescoring and mixing search values into value targets, plausibly account for tens of Elo, not the gap.

**What closing it would cost.** The step from the CNN ($43.20) to this run (~$89) was about one doubling of spend
for about +87 Elo. Taken at face value, the remaining ~250 is three more doublings, a run of about $180, $360 and then
$710: roughly $700-800 for the run that would close it. That is a lower bound rather than an estimate. The +87 came from a better architecture and
schedule, not from doubling compute on the same network; at fixed size this run's last twelve hours bought nothing.
Further doublings would have to pay for larger networks, which need more games to train and serve more slowly, and
returns per doubling usually shrink. The project owner judged that spend not worthwhile; the run ends here.

## Short answer

- **It passed the convolutional lineage.** Ladder Elo at 64 searches settled near **2,480**, against the CNN
  lineage's plateau near 2,360 and peak of 2,407.6. It reached the CNN plateau after about 28 hours of run time.
  The CNN numbers were measured on 8x RTX 4070 SUPER, so the time axis is not like-for-like; the Elo axis is.
- **It then plateaued.** From about generation 700 the ladder stayed within 2,408-2,524, and the 10x192's policy
  loss had been flat since generation ~500. Five more hours of decay after the warm restart (generations 896-950,
  rate 9.5e-5 → 6.8e-5) averaged 2,474 over ten ladder points.
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
| `-resume-warm-restart` | `81f2852e` | `aa6604774018` | 813-896 | rate 1.2e-4 held to 863, geometric to 2e-5 at 1113 |
| `-resume-896` | `ec481868` | `9390e467764c` | 896-950 | the same schedule resumed for five hours; stopped at 950 |

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

The 8x RTX 4080 SUPER node billed $1.60/h. Its figures below are accurate to within about an hour; the other nodes' prices were not
recorded, so they are listed by use only.

| Item | Hours | Cost |
| --- | ---: | ---: |
| AdamW run, all segments (run time, generations 0-950) | ~55.5 | **~$89** |
| of which: to the CNN plateau level (~2,360, run time ~28 h) | ~28 | ~$45 |
| of which: to the final level (~2,480, run time ~44 h) | ~44 | ~$70 |
| Post-stop matches (head-to-head and the 100,000-search gauntlet) | ~1.6 | ~$2.60 |
| 4080 SUPER node rented, 2026-09-30 ~06:30 UTC to the stop at 2026-10-02 17:48 UTC (provisioning, a ~30-minute host outage, resumes, matches) | ~59 | **~$95** |
| Earlier: SGD attention run and smokes on 8x RTX 4070 SUPER (instance 53401154, 2026-09-29 to 2026-09-30 ~06:10 UTC), an 8x RTX 3060 node and a briefly provisioned 8x RTX 3090 (53479423) | — | not recorded |

The node keeps billing $1.60/h until it is destroyed.

Against the convolutional lineage's narrow figure of $43.20 (60 accepted-lineage hours at $0.72/h to its selected
checkpoint, which excludes discarded branches and terminal evaluation), this run cost about twice as much: the 4080
SUPER node costs 2.2x as much per hour for about 1.39x a 4070 SUPER node's throughput on this network. Measured in
money rather than hours, it matched the CNN plateau for about the same spend (~$45 against $43.20) and bought the
further ~120 Elo at 64 searches for roughly another $25-45. Hardware and recipe differ between the two, so this is
a cost record, not a controlled comparison.

## Evidence

Public copies: the weights of generations 896 (with its evaluated float16 ONNX) and 950 are in the
[model repository](https://huggingface.co/BertilBraun/alphazero-chess/tree/main/production) under
`production/attention-10x192-generation-896/` and `-950/`, archived beside the deployed CNN. TensorBoard and logs of
every segment, and the run stitched into one TensorBoard run, are in the
[run dataset](https://huggingface.co/datasets/BertilBraun/alphazero-chess-runs) (`raw/attention/`,
`stitched/attention-adamw/`).

Preserved with `run_control.sh preserve` on the node and copied to
`.codex-diagnostics/vast-chess-8gpu-final-attention-adamw-20261002/evidence-20261002.tar` (1.2 GB, SHA-256
`601c277996ce9a767f67f749c17396200ca390db83b9943b6cc911bcddfbe883`, with a `SHA256SUMS` inside): the complete
warm-restart archive (logs, TensorBoard, evaluations, checkpoint manifests, generation 896 weights and optimizer,
progressive models), the logs, TensorBoard and configurations of every earlier segment, both match results with
their launch and export scripts, and generation 810's weights, manifest and re-exported ONNX. The last segment (`-resume-896`, generations 896-950)
is in `evidence-resume-896.tar` beside it (131 MB, SHA-256
`c5546b8c6b7e52bc5c54e53c58db285c23e7352a2fba3ca010a79ea7109799b9`): its logs, TensorBoard, configuration, new
evaluations and generation 950's weights. The replay buffer (about 19.7M samples) was not copied, so no segment can
be resumed with its replay.
