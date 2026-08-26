# Offline search and self-play option evaluation — plan of 2026-08-26

Branch `search-evaluations`, worktree `C:/Projects/AZ-search-eval`, node `50.120.65.61:41841` (single RTX 3060).
Frozen checkpoint: `vast-chess-4day-production-v9` generation 162, rung `chess-cnn-12x128-dense4`.
Nothing here is started without an explicit instruction; this document is the design and the sizing, not a launch.

Two families, deliberately measured by different instruments.

- **Family A — strength-relevant.** First-play urgency, exploration constant, search value discount, virtual loss
  jointly with parallel searches, and visit count change *which move the search picks*. Measured as playing
  strength of the frozen checkpoint against Stockfish 13 at a fixed node count. No training.
- **Family B — fidelity-type data shaping.** Adaptive versus fixed full-search budget and visit schedules claim
  *equally good policy targets for less compute*. Playing strength cannot separate them, so they are measured as
  the divergence of each rule's policy target from a 10,000-visit reference target, reported against the visits
  each rule spent.

Exploration-type shaping (forced playouts, Dirichlet noise, temperature, `greedy_after_ply`, restart-state starts,
resignation, ply cap) is out of scope: it trades target fidelity for exploration by design, so a fidelity metric
misreads it and only end-to-end runs can judge it.

## 0. Gating finding — the evaluation search could not express the Family A knobs

This had to be resolved before anything could be measured, and it is the one change to shared code.

`ChessImplementation.create_evaluation_search` took the resolved self-play parameters and then hard-overrode
`first_play_urgency` to `ZeroFirstPlayUrgencyParameters()` unconditionally, while `virtual_loss_weight` and
`value_discount_per_ply` were silently inherited from the self-play configuration with no way to set them.
`EvaluationSearchConfiguration` carried only `searches_per_move`, `parallel_searches`, `exploration_constant` and
`inference`, and `run_stockfish_gauntlet` hard-coded `exploration_constant=1.0`. So of the five Family A knobs, the
existing ladder workflow could vary exactly one — the visit count — and silently ran every evaluation at zero FPU
and cpuct 1.0 even though self-play at generation 162 runs reduced-parent FPU 0.2 and cpuct 1.5.

The native side was already capable: `TreeSearchParameters` carries all five and they are bound to Python. The gap
was Python-side only, so no C++ change was needed.

What changed (commit `0408bfb7`):

- `EvaluationTreeSearchOverrides` (first-play urgency, virtual loss weight, search value discount per ply) in
  `src/evaluation/configuration.py`.
- `resolved_evaluation_parameters` in `src/games/implementation.py`, now shared by chess and Go, replacing the
  duplicated `replace(...)` block in both.
- The overrides travel as an **argument to `create_evaluation_search`, not as a field on
  `EvaluationSearchConfiguration`**. This is deliberate: adding the field changed
  `experiment_configuration_sha256` for every experiment, which would invalidate every approval file's recorded
  configuration SHA. With the argument form the hash is byte-identical to `master`; the pinned regression test
  passes unchanged.
- `run_stockfish_gauntlet` exposes `--exploration-constant`, `--first-play-urgency`,
  `--first-play-urgency-reduction`, `--virtual-loss-weight`, `--search-value-discount-per-ply`, records them in
  `model_search_budget` in its result JSON, and applies them through the existing `candidate_selector` hook.

**Not expressible, stated plainly.** The *timed* budget path (`--model-move-time-seconds`) runs the native
`GameAnalysis` engine, which hard-codes zero FPU, discount 1.0 and forced-playout coefficient 0. Overrides are
rejected on that path rather than silently ignored. Family A therefore uses fixed-visit budgets throughout, which
is what the prior standalone ladders did anyway.

## 1. Family A — playing strength

### 1.1 Design

Two stages, because a rung far from 50% wastes most of its games.

1. **Rung calibration.** `tools.run_stockfish_ladder` with `--probe-games 10` over Stockfish 13 fixed nodes
   `1000 2100 3500 6500 11000 20000`, at the baseline arm's search settings. This range was corrected by
   measurement: a 20-game probe at 300 nodes scored **0.925**, so the 50% rung sits far above the v9 evaluation
   rungs (30-1000 nodes), which were chosen for a much earlier generation. The probe finds the rung rather than
   assuming it, and costs about 200 s.
2. **Arm matrix.** `tools.run_search_arm_matrix` runs every arm against the **single** rung closest to 50%, with
   every arm on the same opening sample, the same colour assignment and the same match seed. Arms run as
   concurrent processes on device 0.

Every arm plays the same openings with the same colours, so the comparison against the baseline arm is **paired
per opening**: the per-pair score difference removes opening difficulty, and that difference is what is
bootstrapped (20,000 resamples over opening pairs). The tool reports each arm's absolute score with its own
interval *and* the paired difference against the baseline with `excludes_zero`.

### 1.2 Arms

Baseline is the current **self-play** search at generation 162 — cpuct 1.5, reduced-parent FPU 0.2, virtual loss
1.0, search value discount 0.99, 600 visits, 4 parallel searches — because the decision being informed is how to
configure a future run's self-play search. Note that this is *not* what evaluation runs today (zero FPU,
cpuct 1.0), so the `fpu-zero` and `cpuct-1.0` arms also answer whether the evaluation search has been quietly
handicapping itself.

| Axis | Arms | Rationale |
|---|---|---|
| First-play urgency | `zero`, `parent-value`, baseline `reduced-0.2`, `reduced-0.4` | The three kinds plus one reduction step; the current split between self-play and evaluation makes this the highest-value axis |
| Exploration constant | 1.0, 1.25, baseline 1.5, 2.0, 3.0 | Brackets both the evaluation value (1.0) and the self-play value (1.5) |
| Search value discount | 0.98, baseline 0.99, 1.0 | 1.0 is "off"; the recovery analysis twice found no conversion effect, so this arm tests whether it costs anything |
| Visits | 200, 400, baseline 600, 1000, 1600 | Strength-per-compute; each arm's wall-clock is recorded so the trade is visible |
| Virtual loss × parallel searches | parallel 1 (virtual loss inert); parallel 4 × {0.25, 0.5, baseline 1.0}; parallel 8 × {0.5, 1.0}; parallel 16 × {0.5, 1.0} | Gridded jointly, never alone — with one descent in flight virtual loss does nothing |

21 arms including the baseline. Virtual-loss and parallel arms must be read together with the recorded duration:
raising `parallel_searches` buys wall-clock and usually costs quality per visit.

### 1.3 Size and power — what this design can and cannot detect

This is the part to be blunt about, and the measured throughput (§3) improves the answer.

At a **100-game** arm the reference figure is a 95% interval of about ±0.045 on the score, so a difference between
two arms measured independently carries roughly ±0.064 — about **±45 Elo** near 50%. That is too coarse for the
FPU and cpuct axes. The measured rate allows more than that, so the design is sized at **100 opening pairs =
200 games per arm**, which scales the interval by 1/sqrt(2):

| Games per arm | 95% interval, one arm | On an unpaired difference | Near 50% | Est. wall-clock |
|---|---|---|---|---|
| 100 | ±0.045 | ±0.064 | ~±45 Elo | 2.2 h |
| **200 (chosen default)** | ±0.032 | ±0.045 | ~±31 Elo | **4.4 h** |
| 300 | ±0.026 | ±0.037 | ~±26 Elo | 6.6 h |
| 400 (manifest maximum) | ±0.023 | ±0.032 | ~±22 Elo | 8.8 h |

Pairing on openings removes the opening-difficulty component and tightens this further, but *how much* is an
empirical property of each arm pair and is not knowable before the run — the achieved paired interval is reported
per arm. The smoke test showed the degenerate end: two identically configured arms produced a paired difference of
exactly 0.000 with interval [0.000, 0.000], because the search is deterministic and the games were identical.

Stated without hedging:

- **Detectable at 200 games:** differences of roughly **30 Elo and above** unpaired, somewhat smaller where the
  pairing correlates well. The visit-count arms will clear this comfortably.
- **Not detectable:** differences below roughly 20 Elo. A null result on the FPU or cpuct axes means "no effect
  larger than about 30 Elo", **not** "no effect", and must not be written up as "setting X does not matter".
- The design therefore **answers** "is any of these settings badly wrong, how much does visit count buy, and does
  raising parallelism cost quality?" and only **narrows** "which FPU and which cpuct is best".

If an axis comes back flat and still matters, the efficient instrument is direct **arm-versus-arm** matches rather
than each arm against Stockfish: it removes the anchor and roughly halves the variance of the contrast. That is a
different tool, is not built here, and is the follow-up worth authorising if the ladder result is flat.

### 1.4 Cost, from measurement

Measured on the node while sharing the GPU with two other tenants (§3), and then corrected once calibration
identified the real rung — this matters, because game length depends on it.

- One 600-visit arm against Stockfish at **300** nodes (score 0.925, games end fast): 3.94 s/game.
- One 600-visit arm against Stockfish at **3,500** nodes — the actual 50% rung: **7.3 s/game**, 1.85x slower,
  because balanced games run much longer.
- Six concurrent arms: 2.05x aggregate speedup over one arm, so the GPU is close to saturated at concurrency 6.

At the 3,500-node rung that is 0.281 games/s in aggregate. Weighting the 21 arms by visit count gives 22.3
baseline-arm equivalents (17 arms at 600 visits, plus the 200/400/1000/1600-visit arms at 0.33/0.67/1.67/2.67x),
so the matrix costs `22.3 x games_per_arm / 0.281` seconds — the wall-clock column in §1.3.

**Default: 100 opening pairs (200 games) per arm, about 4.4 h**, plus 50 min for Family B and about 25 min for
calibration — roughly **5.5 h total**, which leaves margin inside one night for the GPU contention that is
already present. Raising `OPENING_PAIRS` to 150 buys ~26 Elo resolution instead of ~31 Elo for about two more
hours; it is a single environment variable on `run-night.sh`.

## 2. Family B — policy-target fidelity per unit of compute

### 2.1 Method, and why it is cheap

Naively, a grid of 35 stopping rules over 3,000 positions is 35 searches per position. It is not: the C++ search
emits a `SearchCheckpoint` every `observation_interval` visits carrying the policy target, root value, top-visit
share, top-two margin and leader. So the study runs **one** search per position, with the adaptive limiter
neutered (`minimum_visits = maximum_visits = reference`, thresholds at 1.0) so it always runs to 10,000 visits,
and then **replays every stopping rule offline against that one checkpoint trace**. The whole grid costs one
reference pass. This reuses the mechanism `tools.calibrate_adaptive_search` already established; the new tool
generalises it from the learned-gate thresholds to arbitrary stopping rules and adds the compute axis.

- `tools.sample_chess_search_positions` harvests the position set by rolling out the frozen model from the 200
  book openings and sampling uniformly from every position seen, so the ply distribution matches what training
  actually sees. Positions with a single legal move are excluded — their target is degenerate.
- `tools.measure_policy_target_fidelity` runs the reference pass and the replay, and reports per rule: mean and
  median stop visits, fraction hitting the maximum, policy-leader (top-1) agreement, most-visited agreement,
  KL(reference ‖ candidate), total variation, and root-value error.

### 2.2 The headline comparison

The fixed rules form a compute–fidelity frontier. For each adaptive rule the tool interpolates that frontier at
the adaptive rule's own mean visit count and reports both directions:

- `kullback_leibler_advantage` — how much better (or worse) the adaptive target is than a fixed budget that spent
  the same visits. Positive means adaptive genuinely improves targets at equal compute.
- `visit_saving` — how many visits a fixed budget would need to reach the adaptive rule's fidelity. Positive means
  adaptive genuinely reduces the visits needed.

Both are `null` rather than extrapolated when the adaptive rule falls outside the measured frontier. This is
exactly the headline question: adaptive either buys fidelity at equal compute, or buys compute at equal fidelity,
or does neither.

### 2.3 Grid — 35 rules

`py/configs/evaluation/chess-search-stopping-grid-v1.json`.

- **14 fixed rules**, 100 to 10,000 visits, covering every stage of the v9 visit schedule (200/300/400/500/600) and
  beyond. `fixed-10000` equals the reference budget, so its divergence must come back exactly 0.0 and its top-1
  agreement exactly 1.0 — a self-consistency assertion that runs on every execution. It did in the smoke.
- **21 adaptive rules**, anchored on the historical chess adaptive budget (`vast-chess-comp2-adaptive`, resolved at
  generation 162: minimum 200, maximum 600, interval 100, window 200, tolerance 0.04, top share 0.7→0.5, top-two
  margin 0.45→0.15, relaxation 1200) and varied **one factor at a time** — minimum visits, root-value tolerance,
  top-share schedule, top-two-margin schedule, interval/window, relaxation, maximum visits — plus four corner
  combinations (aggressive, conservative, and two taller variants) to expose interaction. A full cross of seven
  axes would be 128 cells and would not be more informative one factor at a time is: replay is free, so the limit
  here is interpretability, not compute.
- The **learned search-correction gate is disabled** throughout. It depends on a per-node prediction that is only
  exposed as an end-of-search scalar, so replaying it faithfully is not possible; it already has its own
  calibration tool.

### 2.4 Validated assumption, and its one real limit

The replay is only sound if a search stopped at visit *V* is exactly the prefix of a longer search at visit *V*.
That was tested, not assumed, by `tools.validate_adaptive_replay`: it runs a genuine native adaptive search with a
rule's parameters and compares its `final_visits` against the replay's prediction, position by position.

| Configuration | Rules checked | Positions | Exact agreement |
|---|---|---|---|
| `parallel_searches = 1` | 5 | 200 each | **1.000, mean difference 0.0 visits** |
| `parallel_searches = 4` | 3 | 40 each | 0.900–1.000, mean difference 0–25 visits |

The native search is bit-repeatable across runs, so the parallel-4 gap is not noise: with several descents in
flight the trajectory depends on the visit budget itself, so a 10,000-visit trace is not a strict prefix of a
600-visit run. Disagreements go in both directions, are one to a few observation intervals wide, and bias mean
visits by ≲3%.

**Consequence:** the Family B run uses `--parallel-searches 1`, where the method is exact. Production self-play at
generation 162 uses 4, so the fidelity numbers describe serial-descent search; the table above is the measured
cost of that difference and belongs in the write-up.

### 2.5 Other stated limits

- Dirichlet root noise is off (epsilon 0). Real self-play targets are computed with noise; the comparison is
  cleaner without it and noise is out of scope by construction. Results describe target *shape* absent root noise.
- Forced playouts stay at the production coefficient 1.5, because they shape the policy target deterministically
  and pruning is part of what is being measured.
- Searches start from a fresh root. Real self-play retains 60% of the parent's visits
  (`retained_root_visit_fraction`), so real searches begin partly warm. Fresh-root fidelity is therefore a
  conservative reading of every rule.
- KL floors the candidate probability at 1e-6, so an action the candidate never visited carries a large but finite
  penalty. Total variation is reported alongside and needs no floor.

## 3. Measured rates and sizing

All on the node, shared RTX 3060, filled in from §4 of the benchmark README.

All measured on the node **while two other tenants shared the GPU** (an AlphaZero distillation job at 426 MiB and
a Qwen/TTS job at 10.1 GiB), so these rates are contended and pessimistic rather than clean-machine figures.

| Measurement | Rate |
|---|---|
| Rollout sampling, 200 roots, parallel 4, batch 256 | 13,667 simulations/s |
| Reference pass, 40 roots, parallel 4, batch 128 | 8,542 simulations/s |
| Reference pass, 200 roots, parallel 1, batch 256 | 9,967 simulations/s |
| Reference pass, 64 roots, parallel 1, batch 128 | 9,973 simulations/s |
| Family A, one arm, 600 visits, 20 games | 3.94 s/game |
| Family A, six concurrent arms, 600 visits | 0.50 games/s aggregate |

Reference-pass throughput is insensitive to chunk size across 64-200 roots, and the fidelity tool holds only
**290 MiB** of device memory (the search tree lives in host RAM), so it coexists with other GPU tenants without
memory pressure. Chunk 64 is the recommendation.

**Family B full pass:** 3,000 positions x 10,000 visits = 30M simulations at 9,973 simulations/s, about
**50 minutes**.
**Family A calibration:** six rungs x 10 games, about **25 minutes** measured.
**Family A full matrix:** about **4.4 h** at 200 games per arm, concurrency 6.
**Night total:** about **5.5 h**, Family B first because it is short and its result does not depend on the rung.

### 3.1 Calibration already measured

The calibration stage was run during staging, so the rung is known in advance (the night script re-runs it anyway,
which is a cheap consistency check):

| Stockfish 13 nodes | Score (10 games) | W/D/L | s/game |
|---|---|---|---|
| 1,000 | 0.750 | 7/1/2 | 6.1 |
| 2,100 | 0.850 | 8/1/1 | 7.7 |
| **3,500** | **0.500** | **3/4/3** | **7.3** |
| 6,500 | 0.050 | 0/1/9 | 4.9 |
| 11,000 | 0.100 | 0/2/8 | 6.0 |
| 20,000 | 0.150 | 1/1/8 | 6.5 |

**3,500 nodes is the rung.** The curve is steep — 0.85 at 2,100 down to 0.05 at 6,500 — so the arm matrix sits on
a sensitive part of the scale, which is what maximises the information per game. Ten games per rung is coarse
(±0.3), so the 6,500/11,000/20,000 non-monotonicity is noise, not a real reversal. `ladder_elo_fit` came back
`null` because these node counts are not all in `STOCKFISH_FIXED_NODES_ANCHOR_ELO`; only the rung choice was
needed here.

## 4. What a result would change

- **Family A, visit arms:** if 1,600 visits beats 600 by a detectable margin at acceptable wall-clock, the next
  run's `full_search_budget.visits` schedule rises. If 200 matches 600, the schedule falls and self-play gets
  cheaper.
- **Family A, FPU:** if `zero` is detectably worse than `reduced-0.2`, the evaluation search — which runs zero
  today — has been understating every checkpoint, and `create_evaluation_search` should inherit the self-play FPU.
  That is a correctness fix to the measurement apparatus, not a tuning choice.
- **Family A, cpuct:** the same argument for the evaluation search's hard-coded 1.0 against self-play's 1.5.
- **Family A, virtual loss × parallel:** if quality at parallel 8 or 16 is indistinguishable from parallel 4, the
  next run raises parallelism for throughput; if it degrades, parallelism stays where it is.
- **Family B:** if the adaptive rules show a positive `visit_saving` of a useful size at equal fidelity, adaptive
  returns to the configuration with those parameters. If `kullback_leibler_advantage` sits at zero and
  `visit_saving` is small, the fixed schedule stays and the adaptive machinery's many parameters stop being a
  live question. Either way the fixed frontier itself sets the visit schedule for the next run, which is a
  result the study delivers regardless of how adaptive fares.

## 5. Explicitly not answered

- Nothing here measures a *training* effect. Both families evaluate a frozen checkpoint; a setting that produces
  better targets or better play in isolation can still train worse.
- Family A cannot resolve small strength differences (§1.3).
- Family B measures targets, not the games those targets come from: it says nothing about whether a cheaper
  search visits *worse positions*, only whether it labels the same position comparably.
