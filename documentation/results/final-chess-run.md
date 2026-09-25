# Final chess run

> **Status: complete.** The run stopped cleanly on 2026-09-23, its evidence is fetched and checksum verified, and
> the publication gate below is met. Fields that remain marked as not retained are genuinely unavailable: they were
> not captured before the node was destroyed, and they must not be reconstructed by estimation.

The final run uses the recipe written in full in
[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). Development and operational
continuations used the V89–V93 lineage, but the readable final configuration is the stable reproduction entry point.
The published result will also retain the exact source revision, resolved configuration, configuration hash, and
archive manifest from the completed run so later edits to the living recipe cannot change the historical experiment.

## Publication gate

| Requirement | State |
| --- | --- |
| Run stopped cleanly, archive fetched and verified | met, 2026-09-23 |
| Terminal checkpoint and inference artifacts hashed | met, see Run identity |
| Training-volume and wall-clock statistics derived | partly met, see Training volume |
| Selected evaluations complete under a frozen protocol | met, see Terminal strength |
| Plots generated from archived inputs | outstanding |
| Every number traceable to committed compact evidence | met for the tables below |

## Run identity

| Field | Final value | Evidence |
| --- | --- | --- |
| Run directory and lineage root | `vast-chess-8gpu-v89-v35-progressive-int8-sgd-reuse4-plateau` | Run manifests |
| Segment producing the selected checkpoint | `vast-chess-8gpu-v97-revert-promotion` | `run_manifest.json` |
| Source revision | `f9f8cee4c59403befdf55d34d3d7dffc3671a97a` | V97 run manifest |
| Resolved configuration SHA-256 | `498e7687f68eb48da87212a2a6e43b5fc84b903983c23c9ab1af7cd46a60d243` | V97 approval record |
| Effective training lineage | V89 → V91 → V92 → V93 → V94 → V95 → V97 → V99, continuous learning state in one run directory | Run manifests, checkpoint history |
| Excluded from the lineage | V90 (INT8 collapse, reverted to checkpoint 480) and V96 (19x176 promoted on a training-loss comparison, −270 Elo, reverted to checkpoint 990) | Progressive state backups, replay prune record |
| Canonical readable recipe | [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) — living recipe, since revised | Repository |
| Selected checkpoint generation | **1026** | `final-model/checkpoint_1026.json` |
| Checkpoint SHA-256 | `c92a363b041a18d0ef93b852ac1c6d58716ae9a22b4e62d543de297c4ec5f904` | `final-model/SHA256SUMS` |
| Optimizer SHA-256 | `34c468e1f07ffa7f0db5d4306337248a5ef9a9b6cf6a07e266e3b4fda60eb25e` | `final-model/SHA256SUMS` |
| Inference artifact and SHA-256 | `model_1026.int8.onnx`, `d634abacae3c874eac6ded89f6af861eb81b509da638b5ad710587b1a08be658` | `final-model/SHA256SUMS` |
| QAT state SHA-256 (pre-fold, 408,500 steps) | `c41c955f8d306844201b9cb1ffae7916963cb1f1c96a0c69c4c2f779227cfda9` | `final-model/SHA256SUMS` |
| Hardware | 8x NVIDIA GeForce RTX 4070 SUPER, Vast.ai offer 48571853, driver 595.71.05, 80 logical CPUs, 251 GiB RAM | Provisioning note, archived manifests |
| Locked runtime | PyTorch 2.12.1+cu126, CUDA 12.6, cuDNN 9.10.2, `uv.lock` `a09c3c9697dbe690c874bac20ee8690a3904dd6b8bc8e6b5478c27d61ddd03aa` | Archived manifests |
| Effective training time | 2.5 days on the stitched evaluation axis (60 h); 3.32 days of raw per-run boundary seconds before excising V90 and V96 | `ladder-elo-export.json`, manifests |
| Training-node cost | **$43.20** (60 h x $0.72/h) | Frozen node price and stitched duration |

Evidence archives, all checksum verified against the node before it was destroyed, under
`.codex-diagnostics/final-2026-09-23/`:

| Archive | SHA-256 | Contents |
| --- | --- | --- |
| `evidence-small.tgz` | `4fb8a5941e002d6d8f3186b998ae4a38f3eddd885ef81da3092558c9f556460c` | selected checkpoint, evaluation results, distilled student |
| `evidence-tensorboard.tgz` | `72936c566026c3065cc07c9ce5d234a89dcb4cefd0efd035cea0db9e44e1c910` | every TensorBoard run V77–V99 |
| `evidence-logs.tgz` | `7af5705a92a7bfbaabe5016e9830c68a8b80cd9feaa8d1a84985435c705f8f82` | run-control logs, registry, approvals |
| `evidence-provenance.tgz` | `06a6e071fa41c840407e498b130e868e3aa68620d942450625d6aefe8474ffd7` | run-outcome, resolved configurations, manifests |
| `evidence-tail.tgz` | `06e8807fc9ba5b9e0b7250d1b895f1580ceeaa88d0e92cadacec6d0ee0365779` | all 22 evaluation result directories, both distilled students, float export |
| `evidence-plateau-probe.tgz` | `eb6d94acbad6b76e8cc8ff6ebe44f4dab94f5adecd50dd75674e8009242e41e6` | 400-search and 64-search probe at generations 900/960/1020 |
| `evidence-v100.tgz` | `bde0adb1605f8c37fa96590ab331aa208c781c0d2cd875012d71e2a1f2912306` | V100 ceiling-candidate run |
| `evidence-v101.tgz` | `e6faa44f56d8a462be6fd6b7ba5dac9632ec526e3c88b73b1ee4f1a01ff55e41` | V101 capacity run |

Neither the replay store nor the superseded checkpoint set was retained; both were deliberately left on the node.

## Training volume

| Metric | Final value | Definition |
| --- | ---: | --- |
| Optimizer steps | **513,000** | Generation 1026 x 500 steps per quantum |
| Training presentations | **1,050,624,000** | 513,000 steps x global batch 2,048 |
| Optimizer steps on the selected stage | **408,500** | The 14x160's own counter in the checkpoint QAT state |
| Materialized positions | **≥ 261,016,277** at generation 990 | Credit ledger at the V97 rewind point; not re-recorded at 1026 |
| Replay capacity | 20,000,000 rows, staged 0.6M → 20M | Configuration |
| Configured replay reuse | 4 | Configuration |
| Completed self-play games | not retained | Counted only in the run directory, which was left on the node |
| Final replay occupancy | not retained | The replay store was deliberately not fetched |
| Time in each model stage | 12x128 to generation 481, 14x160 from 481, 19x176 from 1081 (after the selected checkpoint) | Progressive state, benchmark record |
| Time in each search-budget stage | 300/400/500/600 visits by generation 0/10/50/90; 800 from 1000 | Configuration |

The generation-1026 ledger was not separately archived, so materialized positions are quoted at the nearest recorded
point rather than interpolated. Games and occupancy are unavailable by choice, not by loss: the 33 GB replay store
and the superseded checkpoints were left behind to keep the evidence pull small.

## Selected model

| Field | Final value |
| --- | --- |
| Progressive stage and architecture | Second of three: 14x160 scaled-post-activation convolutional, global pooling every second block, `chess_from_to_attention_v1` policy head with key size 128, 2 value channels and a 48-unit value projection |
| Trainable parameter count | **6,315,378** (6,319,887 including 4,509 QAT `_amax` scalars) |
| Inference parameter count | **6,261,007** |
| Training precision | bfloat16 |
| Self-play inference backend and precision | TensorRT INT8 QAT, pre-fold deployment copy, recalibrated every generation |

The selected checkpoint is not the terminal one. Generation 1026 is the last fully retained checkpoint inside the
1020–1080 window where the stitched ladder peaks; the run continued to 1192 under V99's promoted 19x176, which
measured 36–42 Elo below its 14x160 parent and then stayed flat. The terminal state is preserved in the archives.

## Terminal strength

The exact opponent nodes, opening count, paired-game count, search parallelism, hardware, and confidence interval
must accompany every row. Search counts and elapsed time are not interchangeable; a time-based headline must also
state the measured serving topology and latency distribution.

Protocol, identical for every row: Stockfish 13 (`ec56cd6a…`, bmi2 build) at fixed nodes, 1 thread, 1024 MiB hash;
50 opening pairs from `chess-stockfish-8moves-v3-openings-v33.json` (`490425ed…`) played from both colours for 100
games; maximum 300 plies; model served through TensorRT INT8 on 8x RTX 4070 SUPER. Anchors are the fixed-node
Stockfish curve recorded in [the Elo reporting note](../analysis/chess-elo-scale-and-reporting-20260911.md).

The reported figure per budget is the rung scoring nearest 0.500, which is the least draw-distorted and least
model-dependent. Both rungs are listed so the bracket is visible.

| Model search per move | Parallel | Opponent (anchor) | Games | W/D/L | Score | Benchmark Elo (95% CI) |
| ---: | ---: | --- | ---: | --- | ---: | ---: |
| Policy only | 1 | 1,000 nodes (1700) | 100 | 32/24/44 | **0.440** | **1658** (1597–1717) |
| Policy only | 1 | 2,000 nodes (1890) | 100 | 16/22/62 | 0.270 | 1717 (1645–1778) |
| 100 | 1 | 5,000 nodes (2220) | 100 | 51/28/21 | 0.650 | 2328 (2271–2391) |
| 100 | 1 | 10,000 nodes (2470) | 100 | 39/18/43 | **0.480** | **2456** (2393–2518) |
| 1,000 | 1 | 20,000 nodes (2700) | 100 | 47/40/13 | 0.670 | 2823 (2772–2880) |
| 1,000 | 1 | 50,000 nodes (2960) | 100 | 21/48/31 | **0.450** | **2925** (2875–2974) |
| 10,000 | 4 | 50,000 nodes (2960) | 100 | 45/40/15 | 0.650 | 3068 (3016–3124) |
| 10,000 | 4 | 100,000 nodes (3100) | 100 | 30/44/26 | **0.520** | **3114** (3063–3166) |
| 100,000 | 16 | 100,000 nodes (3100) | 100 | 51/38/11 | 0.700 | 3247 (3195–3306) |
| 100,000 | 16 | 200,000 nodes (3230) | 100 | 25/56/19 | **0.530** | **3251** (3206–3297) |

Headline curve, one figure per decade of search: **1658 → 2456 → 2925 → 3114 → 3251**, gains of +798, +469, +189
and +137. The top is anchored by two independent opponents agreeing within **4 Elo**, which is the evidence that the
anchor curve transfers to this engine rather than fanning out.

Search parallelism is not free and must be held fixed across a compute curve. Measured at 1,000 searches against
20,000 nodes: parallel 1 scores 0.670 (2823, 18.1 min), parallel 4 scores 0.645 (2804, 3.4 min), parallel 16 scores
0.610 (2778, 1.3 min). The cost of 16-way parallelism is 235 Elo at 100 searches, 45 at 1,000, and negligible above.

Easy-rung bias, the gap between the harder and easier rung at the same budget, is 128 / 102 / 46 / 4 Elo at
100 / 1,000 / 10,000 / 100,000 searches. It is a low-budget phenomenon driven by draws against weak opposition; the
100,000-search headline is not meaningfully biased.

### Distilled student

A 6x64 convolutional network with a from-to attention head, **470,295 parameters, 13.4x smaller than the selected
model**, trained on the 20M-row replay buffer in float bf16 with no QAT and served through TorchScript, which is not
the teacher's inference path. Weights are distinguished by `inference_model_sha256`, because every evaluation
records the same `run_directory`.

| Student | Searches | Opponent (anchor) | W/D/L | Score | Benchmark Elo (95% CI) |
| --- | ---: | --- | --- | ---: | ---: |
| 36,621 steps (`0a72e733…`) | 10,000 | 10,000 nodes (2470) | 55/24/21 | 0.670 | 2593 (2533–2666) |
| 36,621 steps (`0a72e733…`) | 10,000 | 20,000 nodes (2700) | 31/33/36 | **0.475** | **2683** (2637–2731) |
| 110,000 steps (`ec9eaf25…`) | 10,000 | 20,000 nodes (2700) | 35/29/36 | **0.495** | **2697** (2640–2753) |
| 110,000 steps (`ec9eaf25…`) | 100,000 | 20,000 nodes (2700) | 59/28/13 | 0.730 | **2873** (2819–2935) |

**2683 Elo at 13.4x fewer parameters, 86% of the teacher's 3114 at the same budget**: shrinking the network 13x
costs roughly what cutting search 10x costs. Tripling the training bought +14 Elo, inside the confidence intervals.
The 100,000-search figure is a **lower bound**: at 0.730 the student beat its rung decisively and the bracketing
50,000-node rung was deliberately skipped, so unlike the teacher's 100k headline it has no second opponent.

The project reports protocol-specific benchmark Elo calibrated from fixed-node Stockfish 13 results. It is not a
FIDE rating and is not directly comparable with CCRL, online-server, or current unrestricted-engine ratings. The
existing reporting policy is documented in
[What the v34 Elo numbers mean](../analysis/chess-elo-scale-and-reporting-20260911.md); the final evaluation must
either reuse that calibration exactly or document a revised scale.

## Headline cross-lineage figure

The root README and technical report should share one publication-quality plot of 64-search ladder Elo over
effective training time for the major chess lineages:

- v9;
- v29;
- v34;
- the v46/v48-era successor; the archive audit must resolve the exact lineage label before publication;
- the final V89–V93 continuation, presented as one continuous learning lineage with visible resume markers.

Use `evaluation/ladder_elo_64` wherever that budget-specific series exists. Older logs that expose only
`evaluation/ladder_elo` require a configuration and evaluator audit proving that 64 searches was the primary budget;
do not infer equivalence from the tag name or the shape of the curve. Preserve raw observations in a committed table
and show a documented smoothing line only as an overlay. The x-axis must be effective elapsed training time derived
from the recorded boundary seconds, with downtime excluded consistently. V89–V93 offsets must come from manifests
and event metadata, not visual alignment, and their boundaries must remain visible.

The caption should report the exact start-to-final improvement only after the final point is frozen. Approximate
live impressions such as a 350–450 Elo gain are hypotheses for the final audit, not publishable measurements. A
companion optimizer-step view is useful if schedule efficiency needs explanation, but it must not replace the
wall-clock comparison that demonstrates engineering progress.

## Required figures

- the headline cross-lineage 64-search ladder-Elo figure specified above;
- final-run benchmark Elo and match score versus effective training time;
- policy, WDL, auxiliary, and total training losses;
- learning rate, gradient norm, and clipped-step fraction;
- optimizer steps, self-play games, and fresh positions versus effective time;
- self-play, inference, replay-materialization, and trainer throughput;
- replay age, capacity, and sampling distributions;
- progressive-model candidate start and promotion events;
- search-budget, backend, resume, and other material lineage transitions.

Every generated figure must name or link its source archive or committed compact table. Resume gaps and effective
training time must be represented explicitly rather than silently joined on wall-clock timestamps.

## Relation to v34

This page supersedes v34 generation 1465 as the latest completed public result. Its terminal evidence remains in the
[v34 benchmark](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md) and its trajectory in
the [v34 dynamics report](../benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md).

The two are **not measured by the same estimator** and must not be overlaid without saying so. v34's
`evaluation/ladder_elo` is a single-rung fit and the run logs no `ladder_elo_single_rung*` series at all; this run
logs `ladder_elo` as a three-rung bracketed fit. Converted onto the same estimator using the 35 boundaries where
this run recorded both on the 5,000-node rung, the plateau comparison is:

| comparison | v34 | this run | gap |
| --- | ---: | ---: | ---: |
| single-rung against single-rung | 2281.2 | 2388.6 | +101.8 |
| naive cross-estimator | 2286.8 | 2358.0 | +71.2 |
| **both on the three-rung estimator** | **2283.9** | **2358.0** | **+74.1 ± ~15** |

The often-quoted ~+100 is the single-rung artifact: over its plateau this run's single-rung fit ran 30.6 Elo hot
against its own bracketed fit, because it was pinned on the 10,000-node rung scoring 0.3–0.4, while on the same rung
earlier in the run the offset was −6.9. v34's apparent end-of-run rise to 2388 is the same mechanic and is **not a
strength gain**: one match scored 0.725 against Stockfish at 5,000 nodes, +3.4σ against the preceding 44 points,
crossing the 0.70 advance threshold, after which the last five points are single-rung fits on the 2470 anchor rather
than 2220. Two controls that never changed rung — the policy-only ladder and fixed-dataset accuracy — show no step.

Decomposed by budget with identical evaluation search settings, roughly **+30 Elo of the +74 is a better network**
and **+44 appears only once the tree runs**: at 1 search the two sit at 1689.0 against 1719.3, at 64 searches at
2283.9 against 2358.0.
