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

PHASE_B_NEXT

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

It is the most recent network on the list in the requested 10-20M range; every newer network is 140 MB or more.

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
