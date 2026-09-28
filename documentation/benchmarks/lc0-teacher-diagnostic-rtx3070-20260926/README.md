# Lc0 teacher diagnostic — does the self-play plateau come from the network or the harness?

**Branch `worktree-lc0-teacher`, not for merge.** Plan and runbook:
[`lc0-teacher-diagnostic-20260926.md`](../../plan/lc0-teacher-diagnostic-20260926.md).

## Phase A — the Lc0 evaluator in this project's unchanged search

Every arm uses the identical gauntlet: V97 search at generation 1026, `--exploration-constant 1.0`, paired
openings, float32 TorchScript, Stockfish 13 (`ec56cd6a…`, the archived protocol's binary). Only the network
differs. Scores are for the model; intervals are the gauntlet's 95% bounds.

| Condition vs Stockfish 13 | Lc0 teacher | Checkpoint 1026 | Gap |
|---|---|---|---:|
| Policy only, SF 2,000 nodes, 100 games | **0.855** (79/13/8) [0.795, 0.91] | 0.395 (28/23/49) [0.305, 0.48] | ≈ +380 Elo |
| 64 searches, SF 10,000 nodes, 100 games | **0.885** (80/17/3) [0.84, 0.93] | 0.485 (32/33/35) [0.405, 0.565] | ≈ +365 Elo |
| 1,000 searches, SF 50,000 nodes, 40 games | **0.8625** (30/9/1) [0.80, 0.925] | 0.5625 (10/25/5) [0.4625, 0.6625] | ≈ +275 Elo |

Against an opponent checkpoint 1026 plays dead even at 64 searches, the Lc0 evaluator in the same search scores
88.5%. **The search exploits a stronger evaluator; the plateau is in the learned function, not the harness.**
The gap is almost as large without any search (policy only) as with it.

The 1,000-search condition was cut to the first 20 opening pairs for time; it is the same prefix for every arm.

### Stockfish 18 runs (not the specified protocol)

`engines/stockfish` on the node is Stockfish 18; the first matches ran against it before this was caught. They
remain valid matched comparisons against a much stronger opponent and point the same way:

| Condition vs Stockfish 18 | Lc0 teacher | Checkpoint 1026 |
|---|---|---|
| Policy only, 2,000 nodes | 0.27 (7/40/53) | 0.08 (0/16/84) |
| 64 searches, 10,000 nodes | 0.40 (16/48/36)\* | 0.09 (0/18/82) |

\* The teacher was registered as generation 1 in this run, which resolved the search's maximum game plies to
150 against the control's 250; 42 of its games ran past 140 plies. Every later run registers it at 1026.

### Relation to the archived protocol

Checkpoint 1026 policy-only against SF13 at 2,000 nodes scored 0.395 here against 0.270 in the archived terminal
evaluation. The intervals only just exclude each other; the serving path differs (float32 TorchScript here).
The in-session matched comparison above is the result; archived numbers are context.

## Phase B — a first distillation pilot does not transfer the teacher's strength

**Data.** 2,000,000 positions from 23k search-free games of the teacher: up to eight random opening plies,
temperature 1.3 falling to 0.1 across ply 80, argmax after. Each record stores the project's own 52-plane
input with the teacher's plain policy (top 64 legal moves) and WDL; no search targets, no terminal outcome,
no auxiliary targets. The teacher was queried on Lc0 planes built from the real move history.

**Training.** Checkpoint 1026 loaded strictly (205 tensors; 72 auxiliary-head and QAT tensors dropped),
6,261,007 parameters, float, AdamW at 2e-4 with 100 warm-up steps, batch 1,024, **3,000 steps** (3.07M samples,
about 1.6 passes over the 1.96M training rows), policy and WDL losses, 40,000 held-out rows. Cut from a planned
6,000 steps to fit the reporting window.

| Step | Held-out policy | Gap above the teacher's entropy | Held-out WDL |
|---:|---:|---:|---:|
| floor | 2.1239 | — | 0.3066 |
| 500 | 2.2899 | 0.1661 | 0.3546 |
| 1,000 | 2.2775 | 0.1536 | 0.3426 |
| 1,500 | 2.2648 | 0.1409 | 0.3410 |
| 2,500 | 2.2463 | 0.1224 | 0.3342 |
| 3,000 | 2.2450 | 0.1211 | 0.3336 |

Checkpoint 1026's gap before any fine-tuning, measured at the start of the follow-up run, is **0.2364**: the pilot closed about half of it and its raw policy gained nothing measurable.

| vs Stockfish 13 | Student | Checkpoint 1026 | Teacher |
|---|---|---|---|
| Policy only, 2,000 nodes | 0.405 (24/33/43) [0.33, 0.48] | 0.395 [0.305, 0.48] | 0.855 |
| 64 searches, 10,000 nodes | **0.38** (22/32/46) [0.31, 0.45] | 0.485 [0.405, 0.565] | 0.885 |
| 1,000 searches, 50,000 nodes, 40 games | **0.35** (8/12/20) [0.2625, 0.45] | 0.5625 [0.4625, 0.6625] | 0.8625 |

The raw policy did not get measurably stronger, and searched play got weaker: about 75 Elo at 64 searches
(intervals touch) and about 150 Elo at 1,000 searches (intervals separate). A loss that grows with search depth
points at the value function rather than the policy. The pilot does not show that this network cannot absorb the teacher: it shows that 3,000 steps of joint
policy-and-value fine-tuning does not. Consistent with the plan's warning, the value head was retrained on the
teacher's WDL, a different quantity from the discounted outcomes the search and its FPU were tuned against, which
fits a searched loss with an unchanged raw policy.

### Follow-up: policy-only distillation with the value held to checkpoint 1026

To remove the value-head explanation, a second run replaced every batch's WDL target with a frozen copy of
checkpoint 1026's own WDL prediction (`--value-anchor-checkpoint`), so the policy learned from the teacher while
the value stayed where the search was tuned. Same data and optimiser, **10,000 steps** (10.2M samples, about 5
passes), 47 minutes.

Checkpoint 1026 started **0.2364** nats above the teacher's entropy on held-out positions; the run ended at
**0.1169**, closing 51% of the gap (the pilot closed 49% in 3,000 steps). Held-out improvement stopped near step
8,000 and training loss finished 0.10 below held-out, so further passes over this data would mostly memorise it.

| vs Stockfish 13 | Anchored student | Pilot student | Checkpoint 1026 | Teacher |
|---|---|---|---|---|
| Policy only, 2,000 nodes | **0.325** (17/31/52) [0.245, 0.405] | 0.405 | 0.395 | 0.855 |
| 64 searches, 10,000 nodes | **0.42** (27/30/43) [0.335, 0.515] | 0.38 | 0.485 | 0.885 |

Holding the value recovered part of the pilot's searched loss, but neither student is stronger than checkpoint
1026, and neither raw policy is. **Halving the cross-entropy gap to the teacher produced no measurable gain in
playing strength.**

### Top-move agreement with the teacher

`measure_teacher_agreement.py` replayed the four 64-search Stockfish 13 matches (checkpoint 1026, both students
and the teacher) and scored each network's raw policy against the teacher's at the 21,563 positions where the
evaluated model was to move, with the teacher queried on real game history. It did the same on 20,000 held-out
dataset rows using their stored teacher policy.

| Network | Position set | Top-move agreement | Probability on teacher's move | KL from teacher |
|---|---|---:|---:|---:|
| Checkpoint 1026 | match | 0.595 | 0.345 | 0.268 |
| Pilot student | match | 0.633 | 0.319 | 0.179 |
| Anchored student | match | **0.640** | 0.327 | 0.178 |
| Checkpoint 1026 | dataset | 0.602 | 0.329 | 0.236 |
| Pilot student | dataset | 0.680 | 0.310 | 0.122 |
| Anchored student | dataset | **0.687** | 0.317 | 0.118 |

By phase on match positions, agreement barely varies (1026: 0.593 opening, 0.604 middlegame, 0.586 late;
anchored: 0.632, 0.635, 0.648).

- **Distillation moved the policy, but little of the way.** Top-move agreement on the positions games actually
  reach rose from 0.595 to 0.640: 4.5 of the 40.5 points separating checkpoint 1026 from the teacher.
- **Distribution shift halves the gain.** The same students gained 8.5 points on dataset rows but 4.5 on match
  positions, and their KL is half again as large on match positions (0.178 against 0.118). Checkpoint 1026 itself
  shows no such difference (0.595 against 0.602).
- **The students became less decisive, not more.** Probability on the teacher's top move fell from 0.345 to 0.327
  while agreement rose: the teacher's raw policy is broad (2.12 nats of entropy), and imitating it flattens the
  prior this search was tuned against.
- **Agreement is a weak proxy for strength, so these small gains cannot be converted to Elo.** An earlier
  version of this note read 4.5 agreement points as roughly 40 Elo on a straight line. The from-scratch student
  below disproves that reading: 8 points below checkpoint 1026 at match positions, it plays about 450 Elo weaker.
  What decides strength is how bad a move is when a network disagrees with the teacher, which top-move agreement
  does not measure.

### From-scratch student

Checkpoint 1026's exact architecture (`--architecture-checkpoint`), fresh weights, the same 2M rows and
held-out tail, policy and WDL from the teacher, AdamW at 2e-3 with 500 warm-up steps. Stopped at step 11,000 of
a planned 20,000 with held-out improvement slowing and training loss 0.10 below held-out; the step-10,000
checkpoint was evaluated.

| Step | 0 | 2,500 | 5,000 | 7,500 | 10,000 |
|---|---:|---:|---:|---:|---:|
| Held-out policy gap above the teacher | 1.1900 | 0.3240 | 0.2697 | 0.2479 | **0.2347** |

At step 10,000 the scratch student imitates the teacher on held-out rows **exactly as well as checkpoint 1026
does** (0.2347 against 0.2364) and plays far worse:

| | Scratch (step 10,000) | Checkpoint 1026 |
|---|---|---|
| Policy only vs SF13 2,000 nodes | **0.045** (1/7/92) [0.015, 0.08] | 0.395 |
| 64 searches vs SF13 10,000 nodes | **0.065** (3/7/90) [0.025, 0.11] | 0.485 |
| Top-move agreement, match positions | 0.514 | 0.595 |
| Top-move agreement, dataset rows | 0.570 | 0.602 |

Equal imitation of the teacher on its own search-free games, and hundreds of Elo apart in real games. The
scratch student loses 5.6 agreement points between dataset rows and match positions; checkpoint 1026 loses 0.7.
Checkpoint 1026 learned on the positions its own games reach; the scratch student saw only the teacher's
search-free games, and those do not contain the positions that decide real games.

## Reading

- **Ruled out: the search and the harness.** A stronger evaluator in the unchanged search, served through the
  same pipeline and verified against real Lc0, plays about 365 Elo above checkpoint 1026 at 64 searches and about
  275 above it at 1,000.
- **Ruled out: joint or policy-only fine-tuning of checkpoint 1026 on 2M search-free teacher positions, at up to
  10,000 steps, as a way to close that gap.** Held-out imitation improved steadily and play did not.
- **Measured: the teacher's search-free games are the wrong distribution to learn from on their own.** A
  from-scratch student that matches checkpoint 1026's held-out imitation of the teacher plays about 450 Elo
  weaker; fine-tuned students gain agreement on dataset rows but lose half of it at match positions.
- **Consequence for a larger dataset:** more of the same search-free generation alone is unlikely to train a
  model that beats checkpoint 1026. Teacher labels on positions from real searched games (this engine's, or the
  teacher's own) are the likely missing ingredient.
- **Update, 47M positions with Stockfish-played games (sections below):** fine-tuning checkpoint 1026 on them
  gives the first student at or above 1026 at 64 searches (0.57 against 0.485, not yet significant), but it
  falls behind at 1,000 searches (0.425 against 0.5625); from scratch the same data stops about 90 Elo short.
- **Open:**
  1. *Per-move regret.* Top-move agreement cannot tell a harmless alternative from a blunder. Scoring each
     network by the teacher's evaluation of its chosen move against the teacher's best move measures the thing
     that decides strength.
  2. *Representation.* The teacher is a 20M-parameter attention network; the student is a 6.3M convolutional one
     that encodes eight recent moves rather than eight board states. Not separable until the data distribution
     is fixed.
  3. *Prior sharpness.* Imitating a broad teacher flattens the student's prior; the search's exploration
     constant and FPU were tuned for checkpoint 1026's sharper one and were not re-tuned for the students.

## Data generation for a larger training set

The first 2M rows were labelled by the float32 teacher at a fixed batch of 64, about 2,300 positions a second
with the GPU to itself. A float16 trace (`--precision float16`, traced in half precision so the constants the ONNX
conversion creates are half too) at a fixed batch of 512 (`--fixed-batch 512`) is 5.7 times as fast under the same
load and chooses the float32 teacher's top move on 99.86% of 5,000 match positions (prior difference mean 0.0005,
worst 0.004; WDL worst 0.006). The float32 trace at 512 is bit-identical to the gated one; the batch size, not
the precision, carried most of the gain, since at 64 rows the converted graph spends its time launching kernels.

Four builders on disjoint 5M-position part ranges ran at about 9,300 positions a second together. Parts are
labelled by the float16@512 teacher and cannot be merged with the original 2M, which `distill_merge_datasets.py`
correctly refuses because the teacher hash differs.

## From-scratch student on 47M positions

**Data.** 47,000,000 positions in 11 parts, all labelled by the float16@512 teacher: 25M from the teacher's own
search-free games (parts 00, 01, 10, 20, 30) and 22M from games whose moves Stockfish 13 chose at 1,000 nodes,
sampled from its MultiPV 4 by expected score at temperature 0.05 (parts 40, 50, 60 at 5M; 70, 80 at 2.5M; 90 at
2M). Merged with the Stockfish-mode parts last, so the 470,000 held-out rows are Stockfish-mode positions, where
the teacher's entropy is 2.139 nats; held-out gaps here are not comparable with the earlier runs' teacher-mode
held-out sets. Merged file: 54.9 GB.

**Training.** Checkpoint 1026's architecture with fresh weights, AdamW at 2e-3 (cosine to 0, 500 warm-up
steps), batch 1,024, 30,000 steps (30.7M samples, two thirds of one pass). A first attempt ran at 16% GPU:
random rows from a memory-mapped file larger than the node's memory triggered the kernel's readahead, reading
about 100 times the data used; `MADV_RANDOM` on the map fixed it (85% GPU, about 4.4 steps a second).

| Step | 64 searches vs SF13 10k | Held-out gap | Train / held-out policy |
|---:|---|---:|---|
| 5,000 | 0.08 (3/10/87) [0.04, 0.13] | 0.2465 | 2.3649 / 2.3860 |
| 10,000 | 0.12 (6/12/82) [0.07, 0.17] | 0.2010 | 2.3115 / 2.3405 |
| 15,000 | 0.20 (13/14/73) [0.14, 0.26] | 0.1837 | 2.2842 / 2.3232 |
| 20,000 | 0.225 (15/15/70) [0.15, 0.305] | 0.1626 | 2.2700 / 2.3020 |
| 25,000 | 0.295 (15/29/56) [0.225, 0.37] | — | — |
| 30,000 | **0.275** (15/25/60) [0.215, 0.335] | 0.1527 | 2.2580 / 2.2922 |

Policy only against SF13 at 2,000 nodes, final checkpoint: **0.16** (6/20/74) [0.105, 0.22].

| vs Stockfish 13 | 47M student, 30k steps | 2M scratch student | Checkpoint 1026 |
|---|---|---|---|
| Policy only, 2,000 nodes | 0.16 | 0.045 | 0.395 |
| 64 searches, 10,000 nodes | 0.275 | 0.065 | 0.485 |

The larger, partly Stockfish-played dataset lifted the from-scratch student three to four times over at both
budgets, with training and held-out loss still within 0.034 of each other: it is limited by training, not by
data. It remains below checkpoint 1026. Steps 25,000 and 30,000 are statistically equal, as the learning rate
approached zero.

### Continuation to 130,000 steps

The step-30,000 weights continued under AdamW at 1e-3 (500 warm-up steps), cosine to a 1e-4 floor over a
planned 115,000 steps, same held-out rows. It was stopped after its step-100,000 match (130,000 steps in all,
133M samples, 2.9 passes), when the learning rate was 1.37e-4, to free the GPU for the fine-tune below.

| Total step | 64 searches vs SF13 10k | Held-out gap | Train / held-out policy | Learning rate |
|---:|---|---:|---|---:|
| 40,000 | 0.27 (15/24/61) [0.21, 0.335] | 0.1577 | 2.2603 / 2.2971 | 9.83e-4 |
| 50,000 | 0.235 (13/21/66) [0.175, 0.30] | 0.1492 | 2.2509 / 2.2887 | 9.34e-4 |
| 60,000 | 0.31 (18/26/56) [0.245, 0.38] | 0.1415 | 2.2442 / 2.2810 | 8.57e-4 |
| 70,000 | 0.34 (19/30/51) [0.265, 0.415] | 0.1345 | 2.2366 / 2.2740 | 7.57e-4 |
| 80,000 | 0.30 (20/20/60) [0.23, 0.37] | 0.1313 | 2.2304 / 2.2707 | 6.42e-4 |
| 90,000 | 0.345 (19/31/50) [0.27, 0.425] | 0.1258 | 2.2282 / 2.2653 | 5.19e-4 |
| 100,000 | 0.26 (14/24/62) [0.195, 0.33] | 0.1229 | 2.2216 / 2.2623 | 3.99e-4 |
| 110,000 | 0.37 (24/26/50) [0.295, 0.45] | 0.1198 | 2.2207 / 2.2592 | 2.90e-4 |
| 120,000 | 0.365 (21/31/48) [0.295, 0.435] | 0.1174 | 2.2166 / 2.2568 | 2.01e-4 |
| 130,000 | **0.36** (24/24/52) [0.285, 0.435] | 0.1159 | 2.2150 / 2.2554 | 1.37e-4 |

Policy only against SF13 at 2,000 nodes, step 130,000: **0.315** (19/25/56) [0.245, 0.385].

More training lifted the from-scratch student from 0.275 to about 0.36 at 64 searches (about 70 Elo) and from
0.16 to 0.315 policy-only, and its held-out gap from 0.153 to 0.116, with training and held-out loss 0.040
apart. It levelled off below checkpoint 1026 (0.485 and 0.395), about 90 Elo short at 64 searches.

## Fine-tuning checkpoint 1026 on the 47M positions

**Training.** Checkpoint 1026 loaded strictly (205 tensors; the 72 auxiliary-head and QAT tensors dropped),
the same targets as the from-scratch student (the teacher's policy and WDL, no auxiliary targets, no outcome),
the same 47M dataset and held-out rows. AdamW at 5e-4, 500 warm-up steps, cosine to a 5e-5 floor, batch
1,024, 60,000 steps (61.4M samples, 1.3 passes), 4 hours. Only the starting weights differ from the
from-scratch run.

| Step | 64 searches vs SF13 10k | Held-out gap | Train / held-out policy | Learning rate |
|---:|---|---:|---|---:|
| 0 (checkpoint 1026) | 0.485 [0.405, 0.565] | 0.2260 | — | — |
| 10,000 | 0.345 (20/29/51) [0.265, 0.43] | 0.1288 | 2.2363 / 2.2682 | 4.70e-4 |
| 20,000 | 0.405 (26/29/45) [0.325, 0.49] | 0.1133 | 2.2157 / 2.2528 | 3.88e-4 |
| 30,000 | 0.435 (26/35/39) [0.36, 0.505] | 0.1036 | 2.2039 / 2.2430 | 2.75e-4 |
| 40,000 | 0.505 (35/31/34) [0.43, 0.585] | 0.0943 | 2.1943 / 2.2338 | 1.63e-4 |
| 50,000 | 0.55 (42/26/32) [0.46, 0.64] | 0.0896 | 2.1880 / 2.2291 | 8.01e-5 |
| 60,000 | **0.57** (44/26/30) [0.485, 0.655] | 0.0879 | 2.1878 / 2.2274 | 5.00e-5 |

| vs Stockfish 13 | Fine-tuned 1026 | Checkpoint 1026 | From-scratch student, 130k | Teacher |
|---|---|---|---|---|
| Policy only, 2,000 nodes | **0.445** (31/27/42) [0.365, 0.525] | 0.395 [0.305, 0.48] | 0.315 | 0.855 |
| 64 searches, 10,000 nodes | **0.57** (44/26/30) [0.485, 0.655] | 0.485 [0.405, 0.565] | 0.36 | 0.885 |
| 1,000 searches, 50,000 nodes, 40 games | **0.425** (6/22/12) [0.3375, 0.513] | 0.5625 [0.4625, 0.6625] | — | 0.8625 |

- **The first student at or above checkpoint 1026.** At 64 searches the final checkpoint scores 0.57, about
  60 Elo above 1026's 0.485; steps 50,000 and 60,000 together score 0.56 over 200 games. One 100-game match
  does not separate them (the final interval's lower end is 1026's score), so this is a likely gain, not a
  demonstrated one. Policy only it scores 0.445 against 0.395, also inside the intervals.
- **Strength first fell, then rose as the learning rate fell.** Step 10,000 lost about 100 Elo (0.345) while
  the loss gap had already closed most of the way (0.226 to 0.129), the pattern the pilot showed; every later
  checkpoint was stronger than the one before. Strength followed the annealing, not the loss alone.
- **Starting weights matter more than steps.** From 1026 the student reached a held-out gap of 0.088 in
  60,000 steps; from scratch it reached 0.116 in 130,000 and stayed about 90 Elo below 1026.
- **Deeper search erodes the gain.** At 1,000 searches the fine-tuned student scores 0.425 against 1026's
  0.5625, about 95 Elo lower on 40 games (intervals overlap): ahead at 64 searches, behind at 1,000, the
  depth-dependent pattern of the pilot (0.35 there), milder. A deeper search leans harder on the value head,
  which was retrained on the teacher's WDL rather than the discounted outcomes the search and its FPU were
  tuned against. The same fine-tune with the value held to checkpoint 1026 (`--value-anchor-checkpoint`) on
  this dataset is the run that separates a value mismatch from a weaker policy.

Evidence: `.codex-diagnostics/lc0-teacher-diagnostic-20260926/evidence-finetune.tgz`
(`71ae73c3c751c287c6044a2a7fcbda7c0953c55a23ddccce12701fb1f3301127`): 19 match result files, both training
logs, the node scripts and logs. Weights of the fine-tuned student (`model_1026.pt` `3f7bf420…`) and of the
from-scratch student at 130,000 steps: `student-weights-20260927.tgz`
(`43a3c88c5f3d83e64450637ab68282cb46392ef7691e940eb0eaf0cd08867f46`).


## Architecture: a student built like T1 against the convolutional student

The student network's construction, not its size, turned out to be the lever. A student built like Lc0's T1
(`lc0_attention`, `7c4a89ae`) and trained from scratch on the same 47M positions, with the same held-out rows and
exactly the 14x160 convolutional student's schedule, beats that student at every checkpoint and passes checkpoint
1026 by about 150 Elo at 64 searches.

**Network.** T1's construction with only the sizes changed: each square's 52 input planes concatenated with a
64-dimensional one-hot square code, a linear embedding with Mish and learned per-square multiplicative and
additive gates; ten post-norm encoders (embedding 192, 6 heads of 32) with residual branches scaled by
(2N)^-1/4, a squared-ReLU feed-forward of 768 and a generated attention bias (smolgen 32/192/192, one template
bank shared by all layers) in every layer; the from-to policy head with Mish at width 192; a value head of 32 per
square into 128. 11,908,643 parameters, 366M MAC per position against the convolutional student's 6,261,007 and
396M; most of the extra parameters are smolgen's position-level layers, as in T1. At T1's own size (10x256,
8 heads) the same code has 20,114,979 parameters against T1's 20,204,372. Lc0's own square-code table is not
copied (GPL); any full-rank code gives the embedding the same freedom.

**Training.** AdamW at 2e-3, cosine to zero over 30,000 steps, then 1e-3 cosine to a 1e-4 floor over a planned
115,000 steps; batch 1,024 as two accumulated micro-batches of 512 (the whole batch needs about 8.1 GB).
2.95-3.0 steps a second against the convolutional student's 4.1-4.4. Stopped after step 110,000 of the planned
130,000, once three consecutive checkpoints agreed.

| Step | Held-out gap | 64 searches vs SF13 10k | Convolutional student, same step |
|---:|---:|---|---|
| 5,000 | 0.2151 | 0.055 (1/9/90) | 0.2465, 0.08 |
| 10,000 | 0.1660 | 0.195 (9/21/70) | 0.2010, 0.12 |
| 15,000 | 0.1363 | 0.29 (17/24/59) | 0.1837, 0.20 |
| 20,000 | 0.1110 | 0.495 (32/35/33) | 0.1626, 0.225 |
| 25,000 | 0.0952 | 0.47 (28/38/34) | —, 0.295 |
| 30,000 | 0.0910 | 0.49 (34/30/36) | 0.1527, 0.275 |
| 40,000 | 0.1037 | 0.38 (22/32/46) | 0.1577, 0.27 |
| 50,000 | 0.0982 | 0.545 (33/43/24) | 0.1492, 0.235 |
| 60,000 | 0.0906 | 0.53 (40/26/34) | 0.1415, 0.31 |
| 70,000 | 0.0810 | 0.58 (38/40/22) [0.495, 0.66] | 0.1345, 0.34 |
| 80,000 | 0.0746 | 0.565 (40/33/27) [0.495, 0.635] | 0.1313, 0.30 |
| 90,000 | 0.0690 | **0.685** (57/23/20) [0.61, 0.76] | 0.1258, 0.345 |
| 100,000 | 0.0660 | **0.695** (55/29/16) [0.61, 0.78] | 0.1229, 0.26 |
| 110,000 | **0.0623** | **0.68** (51/34/15) [0.62, 0.745] | 0.1198, 0.37 |

Policy only against SF13 at 2,000 nodes, step 30,000: **0.36** (20/32/48) [0.29, 0.43], against 0.16 for the
convolutional student and 0.395 for checkpoint 1026.

- **The architecture is worth about 200 Elo at equal compute per position.** From step 70,000 the T1-shaped
  student scores 0.57-0.70 at 64 searches where the convolutional student scores 0.26-0.37.
- **It passes checkpoint 1026 from scratch, without self-play.** Steps 90,000-110,000 score 0.68-0.695, each
  interval above 1026's 0.485 [0.405, 0.565]: about +150 Elo. It matched 1026 by step 20,000.
- **It fits the teacher better than any student before it.** Held-out gap 0.0623 against 0.0879 for 1026
  fine-tuned and 0.0806 for the grown 19x176, and still falling when stopped; training and held-out loss stayed
  0.037-0.042 apart.
- **It remains far below the teacher** (0.885 at 64 searches), and T1 is itself distilled from much larger Lc0
  networks, so this does not show that self-play at this size would reach it.

### Early-learning ablation of the convolutional student (SGD)

Before the T1-shaped student, eight 14x160-class variants were trained from scratch with production's optimizer
(SGD, Nesterov 0.9, weight decay 1e-4, clip 1.0; 0.05 falling linearly to 0.005 at batch 1,024) for 30,000
steps. At that length every variant is far from converged (the baseline plays 0.11), so this measures learning
speed only.

| Variant | Parameters | Held-out gap, step 19,000 | Final gap | 64 searches |
|---|---:|---:|---:|---|
| Baseline (scaled post-activation, ReLU6, pooling, from-to policy, value 2/48) | 6.26M | 0.2462 | 0.2123 | 0.11 |
| Value head 32/128 | 6.52M | 0.2477 | 0.2116 | 0.11 |
| 10x192 (wide, shallow) | 6.45M | 0.2473 | 0.2092 | 0.115 |
| Plain post-activation (AlphaZero block) | 6.26M | 0.2492 | 0.2113 | 0.10 |
| Scaled pre-activation | 6.26M | 0.2531 | 0.2141 | 0.09 |
| Global pooling off | 6.60M | 0.2558 | — | — |
| 20x128 (deep, narrow) | 5.72M | 0.2565 | — | — |
| Dense policy head | 6.69M | 0.3183 | 0.2665 | 0.03 |

Only the policy head mattered: the production from-to head is clearly better. Pooling-off and 20x128 were lost at
step ~19,000 when their trainers' source was overwritten mid-run (TorchScript export re-reads it). The ablation
used SGD and the T1-shaped student AdamW, so their numbers do not compare with each other.

### Inference cost

The same 3070, fixed batch 320, float16. Self-play: `tools/run_self_play_search_benchmark.sh` with V97's live
settings (4 processes, 512 games each, one inference worker, 2 outstanding batches, 800 visits), 60 s measured
after warm-up. Forward-only: 200 timed batches after 20 warm-up batches.

| | 14x160 CNN (checkpoint 1026) | 10x192 T1-shaped | T1-shaped / CNN |
|---|---:|---:|---:|
| Self-play, TorchScript float16 | 28,514 searches/s (GPU 96%) | 20,306 searches/s (GPU 95%) | 0.71 |
| Forward only, TorchScript float16 | 23,539 positions/s | 22,094 | 0.94 |
| Forward only, TensorRT float16 | 60,230 positions/s | 32,888 | **0.55** |

TensorRT speeds the convolutional network 2.56x over TorchScript but the attention network only 1.49x, so at
equal arithmetic the T1-shaped network serves at a bit over half the CNN's rate. Production serves the CNN in
INT8, which widens the gap further and has no attention counterpart; that was not measured. The native
TensorRT self-play arms could not run: this node's native extension was built without TensorRT, so the TensorRT
rows are forward-only. Self-play was GPU-bound in both TorchScript arms.

At equal time, about 1.8x fewer searches would cost roughly 50-120 Elo by the report's 190-470 Elo per decade in
this range, against about +150 Elo at equal searches; only a self-play run settles whether the architecture
wins at equal time.

Evidence: `evidence-lc0arch-and-throughput.tgz`
(`971a26486d3099f98ffd2a348b9eb8855acdc793761ab2eef15d7723608ac4be`): training logs, all matches, the step-110,000
weights, the benchmark results and forward timings; `evidence-ablation-and-lc0arch-part1.tgz`
(`96c218a6b0ded5ef1a1be4e2160802d03709dd3336c73867380b345311a2cd8c`): the ablation and the step-30,000 and
step-100,000 weights; `evidence-grown-final.tgz`
(`4253f76ad4647679b63ff1660cd5fae6b3100d8bf5c592301c5f5056277ac914`): the grown 19x176's final weights and matches.

## What this cost

One RTX 3070 node, roughly 3 hours including provisioning, the lc0 build, the gate, two distillation runs and
fifteen matches, plus the agreement check. Evidence: `.codex-diagnostics/lc0-teacher-diagnostic-20260926/evidence-final.tgz`
(`6d82ffb0c25478820bea4c66ecee764bf95c1690718eb565fdd499cdfc60a48f`), 15 match result files, training logs and the
node scripts. The dataset (2.1 GB) and the student weights were not fetched.

## Provenance

| | |
|---|---|
| Node | Vast.ai 83.233.222.244:26204, 1x RTX 3070 8 GiB (CC 8.6), driver 570.211.01; cgroup grant 13.44 CPUs, 42.5 GiB |
| Runtime | PyTorch 2.12.1+cu126, CUDA 12.6, cuDNN 9.10.02 |
| Source revision (matches) | `5a82fd53` and later on `worktree-lc0-teacher` |
| Match configuration | `py/configs/production/vast-chess-8gpu-v97-revert-promotion.yaml` (the selected checkpoint's own recipe) |
| Serving | TorchScript, float32, batch 64, one inference worker, for **every** arm (`GAUNTLET_INFERENCE_PRECISION=float32`, `GAUNTLET_TORCHSCRIPT_INFERENCE=1`) |
| Search | this project's MCTS, `--parallel-searches 1`, `--exploration-constant 1.0`, otherwise the V97 evaluation search, unchanged between arms |
| Opponent | Stockfish 13 from `install_evaluation_engines.sh`, fixed nodes, the project's frozen protocol |
| Openings | `chess-stockfish-8moves-v3` selection (`selection_sha256 ec0fe734…`), paired colours, prefix order; project-layout manifest `f8eceed9…`, Lc0-layout manifest `1904685c…`, identical move sequences (50 of 50) |

The historical opening manifest (`490425ed…`) was not archived. The rebuilt one carries the same 50 openings
from the same selection; its file hash differs because it embeds the builder's source revision.

### Teacher

| | |
|---|---|
| Network | `t1-256x10-distilled-swa-2432500` (lczero.org best-networks list), attention body, 10 encoders x 256, 8 heads, DFF 1024, attention policy, WDL value, moves-left head |
| Parameters | **20,204,372** (ONNX initialisers), 3.2x checkpoint 1026's 6,261,007 |
| Input format | `INPUT_CLASSICAL_112_PLANE` (`lc0 describenet`) |
| Weights sha256 | `bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406` |
| Lc0 used for export and the gate | v0.31.2 from source, OpenBLAS backend |
| Wrapped TorchScript sha256 | `6d1e701e4cb540f9ebcde85e283438d0f34f551e4b5c60315423f858bf66f88f` |
| Policy map sha256 | `94f57edad4bbb9008ec430e1175067f622db84c6e62297e7db334850952ff238` — all 1,880 actions mapped, all 1,858 Lc0 indices covered |

It is the only network on the best-networks list in the requested 10-20M range; every other listed network is 140 MB or larger.

### Baseline

Checkpoint 1026, the selected final-run model: `model_1026.pt` `c92a363b…`, float TorchScript `model_1026.jit.pt`
`1cb9fe4b…`, both matching the archived manifests.

## Fidelity gate — the teacher in this engine is the real Lc0 network

`verify_lc0_teacher_fidelity.py` drove the wrapped teacher and real Lc0 through the same 50 opening lines as
`position startpos moves …`, so both see genuine move history, and compared Lc0's root priors (policy softmax
temperature pinned to 1) and WDL:

| Check | Result | Tolerance |
|---|---:|---:|
| Worst per-move prior difference | 0.00066 | 0.002 |
| Top-move disagreements | **0 of 50** | 0 |
| Worst WDL difference | 0.0026 | 0.005 |
| Batched versus single-position | 1.35e-6 | 1e-4 |

This asserts the plane order, the history planes, the policy permutation (including Lc0's king-takes-rook
castling and bare knight promotions) and the WDL conversion at once. The teacher cannot run in bfloat16 (its
converted graph keeps float32 constants), which is why every arm is served in float32.

Four failures the gate caught before it passed, each of which would have produced a plausible but degraded
teacher: rule-50 was divided by 99 (Lc0 feeds the raw ply count), the trace baked in a CPU device, the trace
baked in its sample batch size, and the knight-promotion index is shared with a plain move.
