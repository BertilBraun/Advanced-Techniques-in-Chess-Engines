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
