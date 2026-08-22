# Why the post-rework chess runs stopped learning — analysis and recommendation

Date: 2026-08-20. Scope: the successful four-day run (`vast-chess-8gpu-1d-r3/r4`, source d39d5c85…d9888436, frozen under `.codex-diagnostics/chess-baseline-four-day-freeze-20260817`) versus the current `master` (994a8cfa) and the 2–4 h runs launched from it on 18–19 August. Evidence used: the frozen TensorBoard event files, run log, configs and Stockfish ladders of the four-day run; the complete `git diff d9888436..HEAD` for `py/`, `cpp/`, `deployment/`; the new production/validation configs; the 18 August pilot, node-comparison and production-launch artifacts; the 19 August overfit benchmark; and the literature. The TensorBoard logs of the failed short runs were not preserved, so statements about those runs rest on the code, the configs and the few numbers recorded in `logs/` and `documentation/benchmarks/`.

Plots referenced below are in `documentation/images/four-day-analysis/`.

## 1. Summary

The four-day run was not bootstrapped. It started from random initialisation (first evaluation at generation 2: fixed-dataset top-1 0.066, Stockfish level-0 score 0.065, policy loss 4.8 ≈ uniform), the r4 continuation started with an empty replay buffer, and there is no replay- or weight-import code path in that tree. Hypothesis C can be dropped.

I found no bug in the data path that would explain a stall: the 76-plane policy index mapping, black-side flipping, mirror augmentation, legal-move masking, value-target orientation (including Syzygy adjudication and all four new auxiliary targets), JIT export, FIFO replay, credit arithmetic and replay ratio are all consistent between the old and new trees and were checked with executable tests (2.1 M legal moves round-tripped, export-vs-training outputs equal to 1e-5, FIFO verified against a reference model). Hypothesis B in its strong form ("targets are wrong") is very unlikely.

What did change, and what the evidence points to, is the optimisation regime of the new network stack (hypothesis A, with a small-B flavour: a design defect rather than a logic bug):

1. The new `chess_76_plane_direct_v2` policy head is a bare 1×1 convolution from the trunk with no hidden layer, normalisation or non-linearity, and the generic init loop applies `kaiming_normal_(nonlinearity='relu')` to it and to every `nn.Linear` in the transformer (QKV, output and feed-forward projections). The attention trunk also has no final LayerNorm before the heads. Measured at initialisation: policy-logit standard deviation 8.4–11.9 for the attention models and 4.6 for the 12×144 CNN, versus 1.0 for the old dense head. The network starts with near-one-hot random priors (legal-masked entropy 0.04–2.2 nats instead of ≈3.4). The repository's own 15×192 trainer benchmark on 18 August diverged within 100 optimizer steps at the production learning rate (policy loss 18 → 313, gradient norm 54 → 715, on both the 3090 and the 4070 S node). The same AdamW 5e-3 / no-warmup / clip-0.5 recipe that was benign for a BN-ResNet with a 1.0-std head is known to be unstable for pre-LN transformers.
2. Self-play now runs with `parallel_searches: 4` (virtual loss) instead of 1, at 200–400 visits, with forced playouts and policy-target pruning on top. The literature says this is mildly harmful to neutral at high visits; at low visits with a sharp random prior it flattens and biases the early targets.
3. The bounded-ingestion coordinator (commit 6b00d893) pauses the 16 "pause during training" workers whenever the inbox holds ≥ 100 games and resumes them only when the inbox is empty and training cannot start. In practice that leaves 8 of 24 self-play workers active most of the time, and the inbox is drained in lexicographic worker order rather than arrival order. This is a throughput and freshness problem, not a learning bug, but it removes most of the headroom the 3090 node was supposed to give and makes wall-clock comparisons with the old run unfair.
4. `cnn-reference` is not the replica it was meant to be. Through `extends` it inherits the new head and its init, the six auxiliary targets, `parallel_searches: 4`, the 3-workers-per-GPU topology with two-thirds paused, `compilation: default`, the new evaluation dataset and the bounded ingestion. The only things it restores are the CNN trunk (at 144 instead of 112 channels) and the visit schedule.

Recommendation in one paragraph: do not launch another multi-day run on the current stack. Fix the head/initialisation/warm-up issue and the ingestion pause logic, then run two short, instrumented A/B runs (2 h each, 8 GPUs) against the four-day yardstick in section 2: (a) the old r3 configuration byte-for-byte on the current code, which tests the code, and (b) the same with only the 76-plane head, which tests the head. Only if (a) reproduces the old 2 h numbers (fixed-dataset top-1 ≈ 0.25, Stockfish level-0 ≈ 0.4, level-1 ≈ 0.2, prev-20m ≥ 0.7) is the platform trustworthy again; only if (b) matches (a) should the new head be used, and only then should attention and progressive sizing be re-tested, one change at a time. If you want a second long run for the write-up, the most defensible one is the old recipe with the fixes that are pure wins (bounded ingestion fixed, 3090s, 12×144 or 15×192 CNN, visits → 1000 earlier), not the attention/progressive stack. A >3000 ladder-Elo result from roughly the same compute is not something the literature supports; see section 5.

## 2. What the four-day run actually did

All numbers are from the frozen event files under `tensorboard-immutable/composed-r3-r4` (74,815 scalar rows, 169 tags). t₀ = 2026-08-13 11:52 UTC. A generation is 500 optimizer steps at batch 2048 with replay ratio 8, i.e. 128k new positions admitted per generation, each presented eight times on average.

### 2.1 Timeline and interjections

| wall h | gen | event |
|---:|---:|---|
| 0 | 0 | r3 start, random init, LR 0.005, 200/50 visits, 8×3060 |
| 0.7 / 1.6 / 3.0 / 7.3 | 10 / 30 / 50 / 90 | visits 300 / 400 / 500 / 600 |
| 8.7 | 100 | LR 0.005 → 0.0035, replay cap reaches 2.0 M |
| 15.6 → 23.4 | 150 | r3 stopped; 7.9 h idle; r4 resumed from checkpoint 150 with an empty replay, LR 0.004, replay ratio 10, value discount 0.9985, max plies 200 |
| 26.2 | 175 | restart: replay cap 2.5 M → 1.5 M |
| 26.9 / 36.1 | 180 / 250 | visits 700 / 800 |
| 43.3 | 298 | restart: value discount 0.996 from gen 300, LR drop moved to gen 350 |
| 50.8 | 350 | LR 0.003 |
| 79.4 → 80.7 | 546 | restart with d39d5c85 (replay ratio 8, LR 0.002 and visits 1000 at gen 550); first quantum crashed on a telemetry assertion, 1.4 h supervisor crash loop |
| 81.1 | 550 | LR 0.002, visits 1000 |
| 96.6 | 624 | stopped; 88.0 h effective training, 704 GPU-hours |

The run was self-play-bound from hour 3 onward: the trainer waited 60–80 % of every generation for credits (`credit/wait_seconds` ≈ 400 s of a 520 s generation at 800 visits, 625 s of 756 s at 1000 visits). Roughly 80 M positions were admitted in total.

### 2.2 The early-progress yardstick

This is the table a new 2–4 h run has to be measured against. Evaluation: every 20 min, 100 games per opponent (50 paired openings), 64 visits per move, Stockfish 13 skill levels 0–4 and fixed 1000 nodes; fixed dataset = Stockfish self-play positions labelled at 10k nodes.

| wall h | gen | fixed top-1 | fixed CE | SF L0 | SF L1 | SF L2 | SF L3 | SF L4 | SF 1000n | prev-20m | prev-60m | policy loss | visits |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.4 | 2 | 0.066 | 3.76 | 0.07 | 0.02 | 0.02 | 0.03 | 0.01 | 0.02 | – | – | 4.8 | 200/50 |
| 1 | 19 | 0.17 | 3.36 | 0.19 | 0.07 | 0.04 | 0.03 | 0.02 | 0.01 | 0.81 | 0.93 | 2.85 | 300/75 |
| 2 | 36 | 0.25 | 3.11 | 0.39 | 0.18 | 0.13 | 0.07 | 0.06 | 0.02 | 0.74 | 0.91 | 2.60 | 400/100 |
| 3 | 50 | 0.30 | 3.06 | 0.62 | 0.39 | 0.29 | 0.17 | 0.14 | 0.02 | 0.66 | 0.83 | 2.42 | 500/125 |
| 4 | 61 | 0.33 | 3.02 | 0.78 | 0.60 | 0.40 | 0.31 | 0.24 | 0.02 | 0.59 | 0.74 | 2.29 | 500/125 |
| 6 | 79 | 0.35 | 3.00 | 0.94 | 0.84 | 0.69 | 0.56 | 0.47 | 0.05 | 0.56 | 0.65 | 2.04 | 500/125 |
| 12 | 124 | 0.38 | 3.04 | 0.97 | 0.94 | 0.84 | 0.76 | 0.68 | 0.13 | 0.51 | 0.51 | 1.77 | 600/150 |
| 24 | 156 | 0.37 | 3.09 | 0.97 | 0.93 | 0.87 | 0.77 | 0.72 | 0.12 | 0.50 | 0.45 | 1.91 | 600/150 |
| 48 | 330 | 0.41 | 3.02 | 0.99 | 0.99 | 0.96 | 0.88 | 0.83 | 0.23 | 0.51 | 0.49 | 1.66 | 800/150 |
| 72 | 495 | 0.42 | 3.00 | 1.00 | 0.98 | 0.97 | 0.91 | 0.88 | 0.27 | 0.49 | 0.47 | 1.61 | 800/150 |
| 96 | 621 | 0.42 | 3.02 | 1.00 | 0.98 | 0.95 | 0.92 | 0.90 | 0.32 | 0.50 | 0.48 | 1.56 | 1000/150 |

(`plot_first8h.png`, `plot_stockfish_scores_full.png`.) The first hour is the most diagnostic window: fixed top-1 must move from ≈0.07 to ≈0.17 by generation ~19, and the prev-20m score must sit at 0.7–0.9, meaning every new checkpoint clearly beats its 20-minute-older self. A run that after 3 h and ≥ 40 generations is below 0.2 top-1 and 0.3 against level 0 is off the trajectory, independent of hardware. The older v13 run needed ≈17 h to reach what this run reached in 4–5 h (`plot_v13_vs_4day_early.png`), so the four-day run's early speed was unusually good, not merely normal.

Note that the fixed-dataset CE/top-1 of the new era (`…direct-policy-evaluation-v2.bin`, legal-masked softmax) is not numerically comparable with v1 (dense softmax over 1880 actions; uniform baseline 7.5 vs ≈3.5 nats). Match scores are comparable.

### 2.3 Where it stagnated and why

Twelve-hour linear slopes of the evaluation series:

| window | top-1 / day | SF L3 / day | SF L4 / day | SF 1000n / day |
|---|---:|---:|---:|---:|
| 0–12 h | +0.41 | +1.78 | +1.62 | +0.31 |
| 24–36 h | +0.01 | +0.18 | +0.17 | +0.13 |
| 36–48 h | +0.04 | +0.06 | +0.05 | +0.03 |
| 48–60 h | 0.00 | +0.09 | +0.05 | +0.06 |
| 60–72 h | 0.00 | +0.05 | +0.04 | −0.01 |
| 72–84 h | −0.02 | +0.03 | +0.02 | +0.04 |
| 84–96 h | −0.02 | 0.00 | 0.00 | −0.06 |

Fixed-dataset CE reached ≈3.00 at hour 5–6 and never went below 2.98 again; the later top-1 gain (0.35 → 0.43) came from a sharper, not a better-calibrated, policy. Training-side KL (policy loss minus MCTS target entropy) is flat at 0.45 from hour 12; the continued decline of the policy loss is explained entirely by falling target entropy as visits rise. The previous-checkpoint gates are at 0.50 ± 0.05 from hour 10–12 onward and carry no information after that. Stockfish levels 0–2 saturate (≥ 0.95) by 48 h. Levels 3–4 and top-1 are flat from ≈60 h (gen ≈ 450). The only late movement is on the hardest rung: fixed-1000-nodes 0.27 → 0.31 after the visits 800 → 1000 step at 81 h (≈1.5 SE with 100-game evaluations; suggestive, not conclusive).

Events that visibly cost something: the gen-150 restart with an empty replay (top-1 0.39 → 0.36, level-4 0.72 → 0.55–0.60, ≈3 h to recover) and the 1.4 h crash loop at gen 547. The cap cut at gen 175 and the value-discount change at gen 298 show no effect. The structural reading is that a 12×112 trunk with a flat CE from hour 5 and a flat KL from hour 12 is capacity- and target-noise-limited, and the lever that moved the hardest rung late was better targets (more visits), not more data or more epochs. Between 60 h and 96 h the run added ≈20 M positions and 87k optimizer steps for at most +0.02 top-1, +0.01–0.02 on levels 3–4 and +0.04 on the 1000-node rung.

### 2.4 Three-day versus four-day checkpoint

Recomputed from `reports/all-rungs.csv` (8 rungs × 10 games per mode per checkpoint) with a bootstrap that resamples rungs as well as games (the report's intervals ignore rung-level sampling and are too narrow; its t1s interval [+63, +124] is really [−92, +377]):

| mode | pooled score 3d | pooled score 4d | Δ Elo (4d − 3d) [95 %] |
|---|---:|---:|---|
| s64 | 0.231 | 0.212 | −34 [−241, +221] |
| s1000 | 0.575 | 0.538 | −61 [−272, +148] |
| s10000 | 0.744 | 0.694 | −86 [−291, +96] |
| t1s | 0.612 | 0.669 | +94 [−92, +377] |
| t5s | 0.675 | 0.719 | +74 [−111, +238] |
| all 400 games | 0.5675 | 0.5663 | – |

Fixed-node modes lean toward the three-day checkpoint, timed modes toward the four-day one, and the pooled difference is −0.001 (SE ≈ 0.03). The 200-game confirmatory matches exist only for the four-day checkpoint (s10000 vs SF13 @ 20k nodes: 96/75/29 = 0.668 → ≈2820 ladder Elo; t1s 44/109/47 = 0.49 → ≈2695). Presenting generation 546 as the result is defensible, provided the text says both checkpoints come from a single run (gen 546 at 79.4 h wall, ≈71.5 h effective training because of the 7.9 h idle gap), that the ladder numbers are exploratory 10-game rungs, and ideally after repeating the 200-game confirmatory on gen 546. Cutting the TensorBoard export at gen 546 is fine as long as the run log and freeze metadata are not edited to hide that training continued.

## 3. What changed since the four-day run

### 3.1 Configuration

Resolution rule (`py/src/experiment/configuration.py:73-90`): `extends` merges dictionaries recursively, lists replace wholesale, a dictionary whose `kind` changes replaces. Almost every field is required by the pydantic models, so there are few hidden defaults; the ones that matter (`materialization_processes`, `sdpa_backend`, `max_grad_norm`, `duplicate_multiplicity_weight_cap`, `auxiliary_targets`) did not silently change between trees.

| parameter | old run, gens 0–60 (later) | `vast-chess-8gpu-optimal` | overnight run a9d6ea5f/6b00d893 | `cnn-reference` | first-4 h impact |
|---|---|---|---|---|---|
| hardware | 8×3060, 64 CPU | 8×3090, 80 CPU | same | same | should help |
| network | CNN 12×112 + global pooling, dense 1880-way head (conv→BN→ReLU→FC), 3.1 M params | attention 8×128 (1.1 M) → 10×160 at 0.75 d → 15×192 at 2 d, each from random init | attention 6×96 (0.47 M) → 10×160 → 15×192 | CNN 12×144, 4.3 M | major, see §4 |
| policy head | dense; init logit std 1.0 | `chess_76_plane_direct_v2`: bare 1×1 conv, 4864 outputs; init std 8.4–11.9 | same | same, std 4.6 | major |
| policy loss | CE over all 1880 actions | CE masked to legal moves | same | same | correct practice; changes CE scale |
| optimizer | AdamW 5e-3 (→3.5e-3 at 100), no warm-up, clip 0.5, bf16 | identical (→4e-3 at 100) | identical | identical | aggressive for a transformer |
| compilation | disabled | default | default | default | speed only, unless a compile bug |
| replay ratio | 8 (10 briefly in r4) | 8 | 10 | 8 | 128k vs 102k positions per gen |
| replay capacity | linear 300k → 2 M over gens 0–100 | staged 300k/600k@25/1 M@50/1.5 M@100/… | same | same | marginally fresher |
| backpressure quanta | 5 | 2 | 5 | 2 | irrelevant with bounded ingestion (§4.3) |
| self-play processes / paused during training | 2 per GPU, 1 of 2 | 3 per GPU, 2 of 3 | 2 per GPU, 1 of 2 | 3 per GPU, 2 of 3 | throughput |
| `parallel_searches` (self-play) | 1 | 4 | 4 | 4 | target quality at low visits |
| full-search visits | 200/300@10/400@30/500@50/600@90 (…1000 at 550) | fixed, same early, 800@180 … 1600@500 | adaptive, min = max = 400 for gens 0–29, 400–500, then 400–800 with a learned gate | 200/…/600@90/700/800/1000 | overnight: 2× visits per position early |
| fast visits / full-search probability | 50→150 / linear 1.0→0.25 over 70 gens | 50/100/150 / staged 1.0, 0.5@30, 0.35@50, 0.25@70, 0.2@250 | same | old values | minor |
| greedy / max plies / force-fast after ply | 60 / 150→160@50 / 200 | 60 / 150→180@50 / 110→140@50→160@100 | same | old values | optimal: no primary rows after ply 110 early |
| Syzygy adjudication at cap | none | wdl345 | wdl345 | wdl345 | negligible early |
| root value blend | 0→0.15 over gens 50–110 | 0→0.10 | same | same | none early |
| auxiliary targets (weights) | next_policy 0.1, remaining_length 0.1 (scale 400) | next_policy 0.05, remaining_length 0.025 (scale 200), future_search_value 0.025, irreversible_progress 0.0125, legal_moves 0.0125, search_correction 0.05 | 0.1/0.05/0.05/0.025/0.025/0.1 | inherits optimal's six | two extra 4864-way heads on the trunk |
| evaluation dataset | v1 | v2 (legal-masked) | v2 | v2 | metric not comparable |

Everything that differs between the old run and `cnn-reference`, which was supposed to be the replica: the head and its init, 112 → 144 channels, `parallel_searches` 1 → 4, the six auxiliary targets, compilation, the 3-per-GPU topology with two-thirds paused, backpressure 2, the staged replay capacity, `minimum_remaining_plies`, Syzygy, the v2 dataset, 3090s, and the bounded-ingestion coordinator. A "replica" that reproduces none of the old run's numbers therefore does not yet tell you whether the code or the design is at fault.

### 3.2 Throughput, old versus new

Old: 128k positions per generation, generations of 170–500 s early, so ≈250–750 primary positions/s across 8 GPUs; ≈160k network evaluations/s in total, ≈20k per 3060 with the 3.1 M CNN at batch 62–63; a training quantum of ≈85 s (≈12k samples/s, eager); trainer waiting 60–80 % of each generation.

New, measured: the 18 August node comparison (`logs/node-comparison-20260818/rtx3090/self-play-summary.json`) shows 487k searches/s for 16 workers with a 0.47 M attention net at 8 visits, i.e. ≈61k evaluations/s per 3090, three times the old per-GPU rate for a six-times-cheaper network. The trainer benchmark (`training-final-15x192.json`) shows 8.6k samples/s for the compiled 15×192 model, i.e. ≈119 s per quantum, slower than the old 85 s. The stopped `vast-chess-8gpu-optimal` run reached generation 94 in at most ≈11 h (the old run was at gen 94 after ≈8 h, and at a 0.9+ score against level 0). So the new runs are trainer-bound or balanced, not data-starved, and they reached a step count at which the old run was already strong. Whatever is wrong is per-sample, not per-hour.

## 4. Findings, ranked

Severity and certainty are mine. "Verified" means checked by executing code or by reading the exact lines in both trees.

### 4.1 Policy head and transformer initialisation, with the old learning rate — degrading to possibly stalling; high certainty on the measurement, medium on the impact

`py/src/training/network.py:212-218` applies `kaiming_normal_(nonlinearity='relu')` to every `Conv2d` and every `Linear` in the model. `network.py:368-375` builds `chess_76_plane_direct_v2` as `Conv2d(hidden → 76, 1×1, bias) + Flatten` with nothing between the residual stream and the logits. `AttentionOutput` (`network.py:551-560`) is a pure reshape, so the attention trunk feeds an un-normalised pre-LN residual stream (RMS 5.7 at 6 layers, 9.3 at 15 layers) straight into that head. Reproduced on CPU with `init_probe.py`:

| model | params | feature RMS | policy-logit std | max |logit| | masked CE at init |
|---|---:|---:|---:|---:|---:|
| old CNN 12×112, dense head | 3.1 M | – | 1.0 | – | ≈3.7 (≈ ln 1880 + 0.3) |
| attention 6×96 (overnight run) | 0.47 M | 5.7 | 8.5 | 31 | ≈20 |
| attention 8×128 (`optimal`) | 1.08 M | 5.9 | 8.4 | 34 | ≈20 |
| attention 15×192 | 4.49 M | 9.3 | 11.9 | 43 | ≈33 |
| CNN 12×144, 76-plane head (`cnn-reference`) | 4.31 M | 3.2 | 4.6 | 29 | ≈6 |

Consequences. First, the search at generations 0–10 runs with near-one-hot random priors. In a toy PUCT simulation (30 legal moves, 200 visits, c = 1.5, Dirichlet 0.25/0.3, FPU −0.2, noisy leaf values) a std-1.0 prior lets the search find the truly best move 43 % of the time with a root-visit entropy of 1.6 nats; a std-8.5 prior gives 16 % and 0.9 nats, and std 4.6 gives 21 %. The policy target therefore mostly echoes the random prior, which the policy then learns, which is the self-reinforcing failure mode Gumbel-AlphaZero describes for low-visit classic AlphaZero. This is a toy, but the direction is not in doubt.

Second, the trainer side. The four-day run's gradient norm was 0.57 at generation 1 and ≈0.2 afterwards with clip 0.5. The new stack's own 100-step benchmark on 18 August (`logs/node-comparison-20260818/rtx3090/training-final-15x192.json` and the 4070 S twin) went from policy loss 18.4 / gradient norm 53.9 after 50 steps to policy loss 313 / gradient norm 715 after 100 steps at LR 0.005, bf16, compiled. A loss of 313 nats over legal-masked logits means logit spreads in the hundreds; that is divergence, on a degenerate two-row replay admittedly, but with exactly the production optimizer. The 19 August overfit benchmark at generation 94 (256-row batch, 225–250 steps) shows that the 6×96 and 8×128 attention models can fit targets, so the wiring is fine; it does not show that the online loop is stable during the first generations, which is where the old run did most of its learning.

How to verify on your machine in minutes: log `training/gradient_norm` and `training/policy_loss` for the first five quanta of any new-stack run and compare with 0.57 / 4.8 → 2.9; print `policy_logits.std()` of `model_0.pt` on one replay batch (expect ≫ 1 now, ≈ 1 after the fix).

Fix candidates, all standard: zero- or 0.01-scaled init of the policy projection (AlphaZero/Lc0 "policy map" heads also put a conv+BN+ReLU hidden layer before the plane projection); N(0, 0.02) or Xavier init for transformer `Linear` layers with the usual 1/√(2·depth) scaling on residual projections; a final LayerNorm before the heads; a linear LR warm-up over the first ≈1000–2000 steps (KataGo trained the first 5 M samples at a third of its base LR "to reduce early instability"); and for the attention models a lower base LR (1e-3 to 2e-3 with AdamW is the conventional range).

### 4.2 `parallel_searches: 4` in self-play — degrading; medium certainty

`SearchTree.hpp:128-150, 284-330` is unchanged and correct (reservations count as visits and as in-flight losses, results are collected only with `in_flight == 0`), so this is a design choice, not a bug. With 200–400 visits, four in-flight leaves and a full-loss virtual loss, up to three phantom losses distort every selection, the first visits are forced apart, and forced playouts flatten the pruned 60-entry target further. The project's own plan document rejected 8-way for the same reason. Expected effect on its own: slower early learning, not a stall. It compounds with 4.1 because a one-hot prior plus virtual loss gives the search the least information per visit.

### 4.3 Bounded ingestion pauses two-thirds of self-play and drains the inbox out of order — throughput and freshness; high certainty on the logic, medium on the magnitude

`py/src/training/coordinator.py:149-180` (`_ingest_toward_next_quantum`) ingests only `samples_needed_for_quantum` per quantum, so credits never exceed about one quantum and `self_play_backpressure_quanta` never fires. When the inbox holds ≥ `INGESTION_PAUSE_BACKLOG_GAMES = 100` games it pauses `node_ids_to_pause_during_training` (16 of 24 workers in `optimal`/`cnn-reference`) and resumes them only when the inbox is exactly empty *and* a quantum cannot start (`coordinator.py:176-180, 252-260`). After every quantum the inbox always holds hundreds of games (eight workers × 512 games kept playing during training), so the 16 are re-paused immediately. Net effect: 8 of 24 workers do nearly all the self-play, and the same 16 are the ones paused during training. The old coordinator ingested everything on every loop (`old-d9888436/py/src/training/coordinator.py:130-145`), kept the inbox near-empty and let backpressure regulate. In addition, `replay/manager.py:270-273` ingests `sorted(glob('*.json'))`, which orders by `worker-{id}-process-{uuid}-game-N`: worker 0's games always go first, so if producers out-run the trainer the high-numbered workers' games become arbitrarily stale and the replay is biased toward a subset of workers; each game JSON is 100–200 KB, so a backlog can also eat the 10 GiB disk margin. Verify with `ls completed-games/inbox | wc -l` over time and `credit/available_presentations`. Fix: ingest everything that arrived (or sort by mtime and cap by samples, not by games), and let backpressure — not an inbox count — pause workers.

### 4.4 `cnn-reference` inherits the new stack — design; high certainty

See §3.1. As written it cannot serve as a code-regression test. A true replica config is given in §6.

### 4.5 Late positions no longer trained on — design change; high certainty, low impact early

`force_fast_search_after_ply` is 110 at generation 0 (150-ply cap), 140 from gen 50 and 160 from gen 100 in `optimal`, versus 200 (effectively never in the first 110 generations) in r3. Fast-search observations produce no primary rows (`materialization.py:107-108`), so endgames and all Syzygy-adjudicated positions only influence training through the result propagated to earlier plies. Intentional per the plan, but it is a real distribution shift relative to the four-day run.

### 4.6 Smaller items

Progressive promotion compares `total_loss` including auxiliary terms (`session.py:225`, `progressive.py:282-283`), and a promoted candidate starts from random init with its own generation counter so the LR schedule restarts at generation 0 for it (`session.py:193-196`); the switch criterion is training loss on data generated by the *smaller* model, which is the same caveat KataGo's scheme has. None of this matters in the first 0.75 days. The `legal_moves` auxiliary head has its own parameters and does not interact with the masked primary policy. Auxiliary weights (0.0125–0.05 each, 0.175–0.35 total) are in the normal range; at initialisation they contribute ≤ 20 % of the total loss. The fixed-dataset metric change (v1 → v2) flatters rather than hides progress.

### 4.7 Verified correct

Policy index mapping (`ChessPolicyEncoding.cpp`: id = plane·64 + from-square; 56 queen-ray, 8 knight, 12 promotion planes; rank flip for black) round-trips 2.1 M legal moves from 400 random games against python-chess; `mirrorActionId` is an involution that maps legal sets exactly; the C++ unit test covers the same. Input bit *i* lands on tensor index *i* (`PackedPlane.hpp:70-89`), and `conv[b,p,r,c] == flat[b, p·64 + r·8 + c]` for both the CNN and the attention token order. Training loss is CE over legal-masked logits (`objective.py:37-52, 151-157`), C++ softmaxes raw logits over legal moves only (`InferencePipeline.hpp:203-248`), evaluation takes the argmax over legal ids; no double softmax; masking gives zero gradient to illegal logits in fp32 and bf16 alike. JIT export (`persistence.py:184-196`) matches the training model to 1e-5 for CNN and attention and correlates 0.99995 with fp32 in bf16. `final_wdl` is always side-to-move at the final position (natural, adjudicated, resigned and Syzygy cases; `chess/contract.py:108-116`, `syzygy.py:27-37`) and `materialization.py:170` flips per player exactly as before; `_validate_result` re-probes Syzygy so a worker/materializer disagreement raises instead of mislabelling. `future_search_value` (offset 4 → same player, sign +1), `irreversible_progress`, `search_correction` and `legal_moves` targets are laid out in the same order in `targets.py:191-229` and `objective.py:100-127` and survive a store round trip. Replay FIFO with capacity changes matches a reference deque over 40 randomised trials; sampling is uniform without replacement; un-ingested games are never deleted; credit arithmetic is unchanged (1 optimizer step per 256 admitted positions at replay ratio 8, each sample presented 8×, identical to r3). Input encoding (`ChessEncoding.cpp`) is byte-identical to the four-day tree. 154 Python tests across the affected modules pass with a stub for the native extension; the C++ tests and an end-to-end materialisation run were not executed here.

## 5. Literature perspective

Small attention networks from scratch in chess. Lc0's transformer gains (BT3/BT4) are for 10–100 M-parameter networks trained on hundreds of millions of positions with smolgen and positional tricks; there is no published evidence that a 0.5–4.5 M-parameter vanilla pre-LN encoder with learned row/column embeddings is as sample-efficient from scratch as a BN-ResNet of the same size, and the project's own fixed-batch test found the 1.1 M attention model 21 % slower per step than a matched CNN. On optimisation, pre-LN removes the hard need for warm-up but large Adam learning rates still destabilise transformers (Xiong et al. 2020; Liu et al. 2020); 5e-3 without warm-up and with ReLU-gain Kaiming init on every projection is outside common practice.

Policy heads. AlphaZero reported that the flat 1858-way head trained "slightly slower" than the 8×8×73 plane head, and Lc0's plane ("policy map") head gained +30 to +90 Elo at equal steps — but both put a hidden conv + BN + ReLU before the plane projection. The direction of the change is right; the implementation lacks the hidden layer and the small-init convention.

Virtual loss at low visits. AlphaZero used 8 parallel simulations at 800 visits; re-implementations report virtual loss as a mild exploration bonus with no measured benefit; Gumbel AlphaZero (Danihelka et al. 2022) shows classic AlphaZero can fail to improve its policy at low simulation counts when the root is not covered. Neutral-to-mild at 800 visits; harmful in the 200-visit, sharp-prior regime.

Progressive growth with re-initialised larger models is exactly KataGo's scheme (b6c96 → b10c128 → b15c192 → b20c256, switching when the candidate's loss catches up), so it is sound in principle; the known costs are the loss-based switch criterion measured on the small model's data distribution and the multiplied trainer cost. Auxiliary targets: KataGo's ownership and score heads and Lc0's WDL and moves-left heads use small weights like the ones here; the new `legal_moves`, `irreversible_progress` and `search_correction` targets have no published precedent and should count as unvalidated.

Replay ratio. KataGo's guideline is about four presentations per generated row, warning that more risks over-fitting; this project uses eight (ten at times), as did the successful run, so it is not a regression — but it is a lever to test if the platform turns out trainer-bound on the 3090s.

Compute versus Elo. Jones (2021) finds ≈500 Elo per 10× training compute in the power-law regime for AlphaZero-style training on board games, and a 15× test-time trade-off per 10× training compute. The four-day run used ≈700 RTX-3060 GPU-hours (≈0.08 V100-years; KataGo's first superhuman run used ≈1.4 V100-years even with all its efficiencies) and reached ≈2520 ladder Elo at 1000 visits and ≈2800 at 10,000. By the Jones slope, +200 to +500 Elo at fixed search needs 2.5–10× the compute, or algorithmic gains of that size on top of a recipe that already contains KataGo's main efficiencies. A >3000 result on the same budget would itself be a research contribution; it is not a reasonable expectation for the next run, and "just scale the model and train longer" contradicts the project's own stated premise.

The Stockfish-13 node ladder. Node-to-Elo is strongly non-linear (for Stockfish 8 on CCRL: ≈200+ Elo per halving below 100 nodes, ≈70 per doubling in the thousands, ≈14 per doubling above a million), Stockfish 13 NNUE is several hundred Elo stronger than SF8 at equal nodes, and MCTS engines scale with nodes differently from alpha-beta. The ladder is a good *relative* yardstick; the absolute "2800" should be labelled ladder-Elo with the calibration stated, not CCRL Elo.

## 6. Recommended plan

Superseded by `chess-recovery-plan-20260820.md` after discussion on 2026-08-20; kept as the original recommendation.

Phase 0 — fix before any GPU is rented (a day of work, all low-risk):

1. Policy head: add the hidden conv/linear + normalisation + non-linearity before the 76-plane projection (or at minimum scale the projection's init to ≈0.01), and give the attention trunk a final LayerNorm before all heads. Replace the blanket Kaiming-ReLU init on `nn.Linear` with a transformer-appropriate init. Add a 1000–2000-step linear LR warm-up, and expose it in the config. Add an assertion-style startup check that `policy_logits.std()` on a real batch is below ≈1.5 at generation 0; it would have caught this.
2. Coordinator: ingest all arrived games each loop (or order by mtime and cap by samples), and tie the worker pause to credit backpressure rather than an inbox count. Log inbox size and `available_presentations` to TensorBoard.
3. Make `cnn-reference` a true replica: `kind: convolutional, num_layers: 12, hidden_size: 112`, the old dense policy head kind (keep the code path alive), `parallel_searches: 1`, the two old auxiliary targets at the old weights, `compilation: disabled`, 2 workers per GPU with 1 paused, backpressure 5, the r3 replay/visit/ply schedules, the v1 evaluation dataset. Diff its resolved configuration against `training_args.txt` from the freeze until only `hardware`, paths and run names differ.

Phase 1 — two-hour A/B runs on 8 GPUs, compared against the §2.2 table at 1 h and 2 h (≈1 % of the four-day budget each):

- A: the true replica on current code. Pass criterion: fixed top-1 ≥ 0.22 and level-0 ≥ 0.3 at 2 h, prev-20m ≥ 0.7 in the first hour. If A fails, the platform has a regression that is not in the code I could execute (native build, compile path, runtime image, inference engine changes); bisect with the old `d9888436` build on the same node before touching anything else.
- B: A plus only the fixed 76-plane head. Pass criterion: within noise of A. This isolates the head.
- Then, one at a time and only on top of a passing B: `parallel_searches: 4`; the four new auxiliary targets; the 8×128 attention model at a warm-up and a lower LR; progressive sizing. Anything that does not match A's first two hours is rejected or fixed, not carried forward.

Phase 2 — the second long run, if you still want one for the write-up. The evidence says the old recipe's ceiling was set by model capacity and target quality, not by data, and that the run was self-play-bound. The changes that address that with the least risk are: the 3090 node with the ingestion fix (≈2–3× the self-play rate per GPU), a 12×144 or 15×192 CNN (which the faster self-play can now afford), the (fixed) 76-plane head, visits reaching 1000 by about generation 250 rather than 550, and the LR decay pulled forward to match the earlier plateau (the 48–60 h window gained most right after the 0.004 → 0.003 drop). Keep replay ratio 8 and the two old auxiliary targets. Expect, on Jones's scaling, on the order of +100 to +200 ladder Elo at fixed search from roughly 2× the effective compute, not +300. Run it only after Phase 1 has passed; a third failed multi-day run would cost more credibility than a well-documented 2800.

On publication: use the generation-546 checkpoint if you prefer, with the caveats in §2.4 (single run, 71.5 h effective training, exploratory 10-game rungs, 200-game confirmatory on gen 624; ideally repeat the confirmatory on gen 546). Report ladder Elo with the calibration procedure, include the 3-day-vs-4-day comparison as evidence of the plateau rather than hiding it, and state the 704 GPU-hours. That is an honest and still impressive result; the stagnation analysis in §2.3 is itself worth a paragraph because it motivates exactly the changes proposed above.

## 7. Artefacts

In `documentation/images/four-day-analysis/`: `plot_first8h.png`, `plot_stockfish_scores_full.png`,
`yardstick_wall_h.csv` (the per-wall-hour table of §2.2 with training-side columns), `threeday_vs_fourday.csv`
(§2.4), `init_probe.py` (initial logit scale; run from `py/` with a stub for `AlphaZeroCpp`), and
`flip-harness/` (`harness.cpp`, `run_checks.py`, `head_check.py` — the compiled colour-symmetry verification of the
policy planes, see the recovery plan). The full scalar export, the 20-minute evaluation bins and the remaining plots
were working files and are not kept in the repository; they can be regenerated from the frozen event files.
