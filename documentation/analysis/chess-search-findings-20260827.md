# Chess search and self-play option findings — 2026-08-27

Distilled decision-relevant findings from the offline study of 2026-08-26/27. The evidence record, with every
raw number, provenance hash and the measurement corrections made along the way, is
[`benchmarks/chess-search-evaluation-rtx3060-20260826/README.md`](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md).
The follow-up work is [`plan/chess-search-followup-plan-20260827.md`](../plan/chess-search-followup-plan-20260827.md).

Frozen checkpoint `vast-chess-4day-production-v9` generation 162 (`model_162.jit.pt` sha256 `a61e8502…`), single
RTX 3060, Stockfish 13 at 3,500 nodes. Every strength arm: 200 games, paired per opening, 20,000-sample
bootstrap. Nothing here measures a **training** outcome; both families evaluate a frozen checkpoint.

## 1. Act on these

### 1.1 The evaluation search has been handicapped since it was written

`create_evaluation_search` forces `first_play_urgency` to zero unconditionally. Self-play at generation 162 runs
reduced-parent FPU 0.2. Measured, that costs **−0.107 score, interval [−0.205, −0.007], about −76 Elo.**

Every ladder number this repository has recorded, for every checkpoint, was produced by a search handicapped by
roughly that much. This is a defect in the measurement apparatus, not a tuning preference, and it makes historical
ladder Elo systematically pessimistic.

The same function also fixed cpuct: evaluation runs 1.0 (`vast-chess-4day-production-v9.yaml:213`), self-play runs
1.5 (line 335). That mismatch is **benign** — cpuct 1.0 against 1.5 measured +0.020 with the interval spanning
zero — but it should be unified for consistency while the FPU fix is made.

### 1.2 Visits dominate every other search parameter

Score against Stockfish 3,500 by full-search budget: 0.318 / 0.537 / 0.580 / 0.662 / 0.748 at 200 / 400 / 600 /
1,000 / 1,600 visits, about **81 Elo per doubling** over 400–1,600 (per-arm interval ±45 Elo, so read the trend,
not the individual steps). 200 visits is catastrophic at −189 Elo, so the early generations of the v9 schedule
search far below the useful range.

Family B agrees independently: policy-target fidelity is still improving steeply at 10,000 visits, and at the
production 600 visits the target names the same best move as its own 10,000-visit reference only **72.4%** of the
time. Two different instruments pointing the same way is the strongest signal in the study.

### 1.3 Fast searches produce no training samples, which reframes the whole allocation question

`replay/materialization.py:112` skips every observation with `full_search` false. **A fast-search position is
never a training sample.** It is played only to advance the game. So the 25% full-search probability is not a
quality dial on the training set — it decides *which positions the network learns from at all*, and the other 75%
of search compute buys nothing but a plausible move.

Two consequences, and they pull in opposite directions.

**Per-position budget scaling on the full searches is the clean win.** Holding the selection random and varying
only how long each full search runs, the oracle bound of §2 applies directly: at a mean of 600 visits a perfect
allocator reaches what a flat budget needs 2,623 visits for, **4.4× effective compute**. Nothing about which
positions become targets changes, so there is no distributional hazard.

**Selecting *which* positions become targets by contestedness is not safe on its own.** Ranking positions by how
much the target moves between 200 and 600 visits and taking the top quartile gives targets that are markedly less
reliable:

| Quartile selected for the full search | KL of its 600-visit target |
|---|---|
| Random — what runs today | 0.2971 |
| Most contested 25% | **0.4694** |
| Least contested 25% | 0.3994 |

The contested quartile needs about **1,600 visits** to reach the target accuracy a random quartile already has at
600. Contested positions carry the most learning signal *and* the least trustworthy labels at a fixed budget, and
reallocating budget within that quartile barely helps (0.4530 against 0.4694 flat) because it is homogeneously
hard — the heterogeneity the oracle exploits lives mostly in *easy* positions, which selection has already
removed.

Note that both extreme quartiles score worse than random, because low benefit mixes two different populations:
already-decided positions, where both budgets agree and the target is good, and diffuse positions where search
does not converge at either budget. Benefit alone does not separate them.

So selection and budget are not independent choices. Selecting hard positions without also lengthening their
searches trades target accuracy for learning signal at an unmeasured exchange rate; the defensible order is to get
budget scaling working first, and treat selection as a second step that must come with a budget increase.

## 2. Adaptive search: the mechanism is sound, the criteria are not

Adaptive stopping does reduce visits as designed — 513.9 against a 600 cap. But those visits buy only what a flat
**466**-visit budget buys, and all 21 rules in the grid share the sign. About 40% of that loss is Jensen convexity
that any non-uniform allocation incurs regardless of how well it chooses; the rest is genuine anti-selection.

The reason no parameter setting rescued it: **the signal it reads is inverted-U, not monotone.**

| Decile of top-visit share at 200 visits | Mean benefit of 200 → 600 | Against population |
|---|---|---|
| 1 (0.025–0.093) | 0.0253 | 0.16× |
| 5 (0.335–0.408) | 0.2617 | 1.65× |
| 9 (0.696–0.842) | 0.2725 | **1.71×** |
| 10 (0.843–1.000) | 0.0483 | 0.30× |

Diffuse positions gain little, already-decided positions gain little, genuinely contested positions gain most. The
production schedule stops at a top-visit share of 0.7 decaying to 0.5 — the band where the rule **forgoes more
benefit than picking positions at random** (0.1808 and 0.1846 against 0.1589). Only at ≥0.85 does it beat random,
and that stops just 13% of positions.

**A monotone threshold on a non-monotone signal cannot work at any setting.** That is why the seven-parameter grid
moved nothing.

Meanwhile the ceiling for per-position budgeting is large: a perfect allocator reaches with 600 visits what a flat
budget needs **2,623** for, **4.4× effective compute**. Not an artefact of the finite reference — at a mean of 521
visits only 0.1% of positions receive the reference budget, and removing it from the allocator's menu barely moves
the result.

A per-position head for this already exists: `search_correction_target`
(`cpp/src/search/SearchExecutor.hpp:161`) is trained as a scalar auxiliary target and read back at the root by
`SearchCorrectionGate`. It predicts `max(policy_correction, value_correction)` — how far the search moved the
answer from the network's prior — which is a *learning-value* signal, not the *marginal-return* signal budget
allocation needs. A tactic the network missed scores high on it but converges in a few dozen visits; two near-equal
moves score low on it but keep moving for hundreds. See the follow-up plan WP-S2b for the proposed label.

**Conclusion: retire the current adaptive budget, keep the idea.** The gap between "what the rule achieves" and
"what perfect allocation achieves" is the entire case for a learned difficulty head.

## 3. Parallelism, virtual loss, and the fast/full tail

Two earlier readings of this axis were wrong and both are recorded in the benchmark README. What follows is the
corrected picture.

`schedulableTask` hands each tree one in-flight descent per pass before advancing
(`cpp/src/search/SearchExecutor.hpp:507`), so effective concurrency is
`min(parallel_searches, inference_capacity / active_trees)`. **When trees outnumber capacity, the parameter is
inert.**

### 3.1 Where it binds, parallelism costs strength

Measured at batch 1024 against ~100 trees so it genuinely binds, 600 visits, 200 games per arm:

| parallel_searches | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Score | 0.613 | 0.585 | 0.562 | 0.550 |
| Elo against parallel 1 | — | −20 | −36 | −45 |

No single contrast clears 95% at 200 games, but the ordering is monotone across four arms and agrees with an
independent observation of 100–200 Elo lost at `parallel_searches` 64 in single-game evaluation. Virtual loss
changes nothing at parallel 8 (0.550–0.560 across weights 0.25–1.0), so **leave it at 1.0**.

### 3.2 But in real self-play, parallelism buys throughput — because of the fast/full tail

Self-play submits all games in one call. At generation 162, 25% run 600-visit full searches and 75% run 150-visit
fast searches, and the fast-search staggering in `initialFastSearchAdmissionCount` does **not** engage: its
ratio-based value of 96 is overridden by the capacity-based value of 384, so all 512 games start together. The
fast searches finish after 150 visits and leave 128 full-search trees to carry the remaining 450 — about 43% of
the simulations at a quarter of the trees.

Measured with the real mix (`tools.benchmark_self_play_search`, production inference settings, 40 s per cell):

| Games | parallel 1 | parallel 2 | parallel 4 | parallel 8 |
|---|---|---|---|---|
| **512** | 22,553 (batch 86) | 24,475 (165) | **27,013 (268)** | 28,690 (315) |
| 320 | 15,135 | 22,772 | 24,794 | 27,032 |
| 160 | 12,653 | 14,861 | 22,257 | 24,910 |
| 80 | 10,516 | 13,084 | 14,441 | 21,959 |
| 40 | 7,275 | 11,047 | 14,352 | 16,677 |

At 512 games with `parallel_searches: 1` the average inference batch is only **86.1** against a batch size of 320
— the tail starves. Parallel 4 lifts it to 267.9 and buys **+20% throughput**; parallel 8 gives +27%. Under
uniform 600-visit searches the same comparison is worth only +3.6%.

**So `parallel_searches: 4` is not wasted. It exists for the fast/full tail and is earning its keep.** The
unwelcome corollary is that the tail is exactly where the *full* searches live, so about 75% of every full search
— the searches that produce training targets — runs at parallel 4 and pays the quality cost of §3.1.

### 3.3 The games/latency trade, from the real mix

Seconds per move per game, at a mean of 262.5 visits per move:

| Configuration | Searches/s | s/move/game | Against today |
|---|---|---|---|
| **512 × 4 (today)** | 27,013 | 4.98 | — |
| 320 × 4 | 24,794 | 3.39 | 1.47× fresher, −8% throughput |
| 320 × 2 | 22,772 | 3.69 | 1.35× fresher, −16% throughput, less quality cost |
| 160 × 4 | 22,257 | 1.89 | 2.6× fresher, −18% throughput |
| 80 × 8 | 21,959 | 0.96 | 5.2× fresher, −19% throughput, more quality cost |

Note that **320 × 4 dominates 320 × 2** on both throughput and latency at equal quality cost, because the tail
rewards parallel headroom. Throughput, game latency and search quality form a three-way trade with no free corner.

## 4. Everything else measured

| Parameter | Result |
|---|---|
| cpuct 1.0 / 1.25 / 2.0 against 1.5 | all within noise; **3.0 is −95 Elo and clearly harmful** |
| Search value discount 1.0 (off) / 0.98 against 0.99 | +0.060 / −0.048, both intervals spanning zero — no evidence either way |
| FPU `parent_value` and `reduced_0.4` against `reduced_0.2` | within noise |
| Virtual loss 0.25 / 0.5 / 1.0 | no material difference where it binds |

Six arms returned nulls. At 200 games these **bound** effects at roughly ±30 Elo; they do not show the settings
are irrelevant.

## 5. What this study cannot say

- **No training effect is measured.** A setting that improves targets or play in isolation can still train worse.
- The parallelism Elo costs are point estimates, not significant at 200 games. Resolving 45 Elo needs ~800 games
  per arm.
- Family B ran at `parallel_searches` 1 from fresh roots; production uses 4 with 60% root retention, and the right
  metric for the parallelism cost is target fidelity rather than Elo. That comparison has not been run.
- The learned search-correction gate was disabled throughout and is untested.
- The oracle bound is perfect hindsight and unreachable; how much a learned head recovers is unknown.

## 6. Method lessons worth keeping

Three measurements in this study were wrong before they were right, and each failed differently:

1. **Per-arm wall-clock from a concurrently-scheduled matrix is not a throughput measurement.** A draining thread
   pool makes the last arms look fast. Hold concurrency fixed, or run sequentially.
2. **A parameter can be silently inert.** Nineteen parallelism arms measured one configuration because the batch
   scheduler never let the parameter bind. Scores clustering implausibly tightly across a wide parameter range is
   the tell; always verify a knob changes behaviour before interpreting its absence of effect.
3. **A microbenchmark is not the workload.** Uniform 600-visit searches said parallelism was worth 3.6%; the real
   fast/full mix says 20%. Measure the mix that production runs.
