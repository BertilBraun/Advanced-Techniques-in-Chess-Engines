# Learned early stopping for self-play search (DS-MCTS adaptation)

**Status:** plan, revised 2026-09-01 after user review. Not implemented; no code changed. Supersedes
the predicted-curve budget allocator (`search-budget-curve-20260830.md`). The predecessor's negative
result is committed (`77bcfe96`, `documentation/analysis/adaptive-search-budget-negative-result-20260901.md`),
so the old subsystem is deleted outright in this work (section 9) — git and the writeup are the record.
User decisions already taken: `Nmax = 2B`; no online spend controller; CPU TorchScript predictor;
audits ride live self-play; the 3→2-tensor inference-contract break is accepted; Stage O2 (~1 GPU-hour)
is authorized — v20 is stopped checkpoint-safe and the node (38.49.42.120:53893) is free for it.
Expected scale: about a day of development and iteration, after which either adaptive stopping works
or the whole adaptive-search line is retired and the final run uses v13/v17-era configurations.

**Base paper:** Lan, Tsai, Wu, Wu, Hsieh, *Learning to Stop: Dynamic Simulation Monte-Carlo Tree Search*
(DS-MCTS), AAAI-21, arXiv 2012.07910. Their stated future work — applying the method to generating
self-play records during AlphaZero training — is this project.

---

## 1. Why the predecessor failed, and what this design changes structurally

The predicted-curve allocator worked as measured by its own proxy (+10.7% KL reduction vs flat at spend
0.967, ~1.22x effective search compute, sensible allocation) and still lost 60–100 Elo at matched
wall-clock across five runs (v16–v20). The leading explanation: `KL(pi_deep || pi_at_budget_k)` is a
proxy that cannot see training-data cost. A cheapened position's policy target is closer to the
network's own prior; the run trains its next generation on softer targets exactly where the allocator
judged search "unprofitable". KataGo (Wu 2019) reaches the same conclusion independently and therefore
never uses fast searches as policy targets.

Four structural changes follow. Each one removes a mechanism that was implicated in the failure or in
one of the three production-killing defects (config pin, inverted isotonic projection, stale dual):

1. **Hard quality constraint instead of a hard spend constraint.** The Lagrangian dual forced mean
   spend to 1.0 every generation, so poor predictions still forced cheap searches somewhere. Here the
   constraint is on target quality (false-stop rate held under a ceiling by a self-tuning calibrator
   on the resignation pattern, section 6.1); the compute saving is whatever falls out. There is no
   dual variable and no state that can go stale for 231 generations: threshold selection is a fresh
   evidence evaluation each generation, tightening immediately and relaxing only step-bounded.
2. **The decision is conditioned on the live tree, not made before the search.** Every previous
   ablation showed the recoverable signal lived in observables (`top_visit_share`, `policy_entropy`,
   `baseline_visits`, `generation`), none in the head's own output. Live tree state at a checkpoint is
   a strictly stronger observable set, and it is what DS-MCTS's MCTS-UN uses.
3. **Same-tree labels are now the *right* measurement, not a bias.** The criterion asks "will *this*
   search's distribution move further before reaching B?" — the stopped production search is literally
   the same tree that would have continued. The old scheme's fresh-root re-search deployment gap
   (`search-budget-curve-20260830.md`, allocation section, last paragraph) disappears.
4. **No trunk head, no auxiliary gradient, no replay target column.** The eight-wide curve head, its
   masked Huber loss, dedicated batches and the `AUXILIARY_VALUE` replay plumbing are deleted. The
   predictor is a small external MLP fit offline on audit records (the corrector precedent, which is
   the one component that measurably worked: 2×64 MLP, TorchScript, CPU-evaluated natively).

### 1.1 The thesis: verifiability, not magnitude

The retired system cheapened positions where a *predicted* curve looked flat — prediction error was
large (best corrector: 32.9% of oracle out-of-fold), nothing verified the prediction on the affected
position, and the damage (prior-like targets) was invisible to its own proxy. This design inverts
that: a position is stopped only when its own tree's distribution has been observed to have
stopped moving over the last checkpoint interval (a measured necessary-condition filter, section
5.2 — not a guarantee: a monotone drift passes every local check while accumulating total drift,
which is exactly why the predictor and labels exist) *and* a predictor certifies it will stay within
eps of the full-budget distribution — a property that is directly verified on every audit position,
with an asymmetric threshold driving false stops toward zero. A true stop saves a bounded quantity
(some fraction of one search); a false stop injects a degraded policy target whose cost compounds
through the next generation's training — v20 is the measured proof that enough of those outweigh a
1.22x compute equivalent. The whole design is shaped by that asymmetry (sections 6 and 7): quality is
the constraint, spend is the free variable.

### 1.2 Measured constraints from v20 (6,351 labelled positions, generations 260–268)

These numbers, measured against the 8x deep reference, bound what this plan can deliver and are the
reason its claims are modest:

- **DS-MCTS's cheap-and-easy mass does not exist here.** Their method leans on >62% of Go states
  being settled after ~1 simulation under an *argmax* criterion. Under a *distribution* criterion at
  tight eps, only 4.6–11.5% of our positions are within eps 0.01 of the deep policy at budgets
  0.125x–1.0x (21.5% at 0.125x even at eps 0.10). Their hit rates do not transfer; plan numbers
  below assume ours.
- **The oracle ceiling is known.** A perfect stopping rule (stop at the cheapest grid budget whose
  KL < eps, else run to 2x) yields, at eps 0.10, mean spend 0.956 and +26.7% KL gain vs flat —
  **1.64x effective compute at unit spend**. The retired system achieved 1.22x live *and still lost
  60–100 Elo*. So the total headroom over the thing just rejected is ~1.35x before any prediction
  error. Nobody should expect this mechanism to win on magnitude; if it wins, it wins because the
  1.64x is delivered without the target-degradation channel that nullified the 1.22x.
- **Extension beyond baseline is where the value is** (user decision, confirmed): pure saving is
  nearly worthless (eps 0.20 saves 36% of compute for +6.4% gain), while the gain at eps ≤ 0.10 comes
  from spending up to 2x on unconverged positions. This matches the paper: their fixed-Nmax stopping
  regime is break-even (~49% win rate at 715–801 avg simulations); the headline 61.4% comes from
  raising Nmax to 4000 while averaging 1584. The search therefore runs to a **fixed cap of 2x
  baseline** unless stopped earlier — stopping is the mechanism, extension is the payoff.

**Value proposition, corrected twice by measurement (2026-09-01).** The O2 noise floor moved the
objective from "better targets at equal compute" to "equal-quality targets at lower compute", and
the trained-predictor holdout then bounded what is realizable: the oracle's 0.49 spend at eps =
floor is not reachable at acceptable false-stop rates — the group-split holdout frontier is
**spend ≈ 0.75 at beta ≈ 0.10** with written targets flat-equivalent (mean 0.056 vs flat's own
0.049 gap to the reference; p90 0.093 ≤ floor median), i.e. **~1.3x search throughput**, with
Amdahl on top of that before it becomes generation cadence. Guard-only stopping (no predictor)
was measured and fails badly (16–41% of stops exceed eps); the predictor earns its place. The
retire-only-on-Elo rule from the v13–v18 findings still applies to the final decision, but the
entry gate is Stage O: the realizable rule must clear the throughput bar of section 10 before any
production time is spent.

---

## 2. Design overview

Per self-play position with baseline (new) visits `B` and a fixed cap `Nmax = 2B`:

- The search always runs. At five **checkpoints** `c_i = starting_visits + round(m_i * B)`,
  `m_i ∈ {1/3, 1/2, 2/3, 1.0, 1.5}` (a subset of the existing `BUDGET_CURVE_MULTIPLES` grid spanning
  both sides of baseline), the native executor first applies a hard **observational guard** —
  stopping is only considered if the measured movement since the previous checkpoint,
  `KL(pi_{c_i} || pi_{c_{i-1}})`, is already below eps — and then evaluates a TorchScript stop
  predictor on scalar tree features. If the guard holds and the predicted uncertainty
  `u < thr[i]`, the search stops and the current policy-target distribution becomes the training
  target; otherwise it continues. Reaching `starting_visits + 2B` ends the search unconditionally.
  Mean spend is **not** controlled online: there is no dual, no spend ledger, no per-generation
  controller. Spend floats, is measured per generation, and `eps` / `Nmax` are adjusted **between
  runs** only. (Three of the four run-killing defects in this subsystem's history came from online
  spend control; the fixed-cap design removes the mechanism class entirely. The expected operating
  point — eps 0.10 — sits at mean spend ≈0.96 by the oracle table, so unit spend is the design
  center, not a constraint to enforce.)

**Why exactly this checkpoint set — settled by measurement, not taste.** Oracle stopping on the same
6,351 v20 positions at eps 0.10, restricted to each candidate subset (each checkpoint costs one
drain/sync in the executor, section 5.3):

| checkpoint subset | mean spend | gain vs flat | effective compute |
|---|---|---|---|
| full 8-point grid (0.125…2.0) | 0.956 | +26.7% | 1.64x |
| **plan: 1/3, 1/2, 2/3, 1, 1.5 (+2 cap)** | 1.014 | +28.8% | **1.60x** |
| plan + 1/5 restored | 0.974 | +27.5% | 1.63x |
| coarse: 1/2, 1, 1.5, 2 | 1.100 | +31.2% | 1.54x |
| minimal: 1/2, 1, 2 | 1.148 | +32.2% | 1.50x |

Dropping 1/8 and 1/5 is not ambition: those two cheap checkpoints together are worth ~0.04x effective
compute — under a distribution criterion almost nothing has converged that early (section 1.2) — and
each would buy that sliver at the price of a per-move drain. Even the three-checkpoint minimal set
retains 1.50x, so the five-point plan set sits comfortably on the flat part of the curve. Stage O2
re-checks this table at the true 2B label reference; the subset only changes if that re-derivation
disagrees materially.
- A `sample_fraction` of positions are **audit positions**: the stop rule is evaluated and its verdict
  *recorded* but not obeyed; the search runs to the full `2B` cap with checkpoint policies captured
  (`SearchCheckpointDetail::Policies`). These records provide ground-truth labels, predictor training
  data, threshold calibration data, and a continuous unbiased measurement of what stopping is doing to
  target quality. Audit cost is the position's own full cap rather than a deep multiple — at
  `sample_fraction` 0.01–0.02 that is ≤ +2% of self-play visits, versus the retired system's 8x deep
  searches at ~12% GPU cost.
- No State-UN (constraint 1), no deep 8x reference search, no separate GPU labelling jobs, no network
  head, no dual, no corrector-on-top-of-head, no isotonic projection.

The entire deep-label subsystem (fresh-root re-search of replay positions at 8B on claimed GPUs)
is replaced by in-situ shadow measurement. That is the single largest simplification: the ~4.6%
label-generation cadence penalty and the 12%-GPU-cost deep searches go away entirely.

---

## 3. Label definition (the intended novel contribution)

### 3.1 Formula

Let `pi(.|s, n)` be the **policy-target distribution** at root visit count `n` — the
forced-playout-pruned `policyTargetVisits` normalization, i.e. exactly what would be written to replay
(`SearchExecutor.hpp:385-415`), not the raw visit counts. Let `V(s, n)` be the root value at `n`.
Let `N = starting_visits + 2B` be the full capped search, and `c_1 < … < c_K < N` the checkpoints.
The reference is the cap distribution `pi(.|s,N)`, not an 8x deep search: it is what an unstopped
production search would actually produce, so labels certify equivalence to the realizable target.

For each checkpoint index `i`, the position is **uncertain at `c_i`** — label `U_i = 1`, meaning
"keep searching" — iff

```text
U_i = 1  ⟺  KL(pi(.|s,N) || pi(.|s,c_i)) ≥ eps_pi  ∨  |V(s,c_i) − V(s,N)| ≥ eps_v
```

(Instantaneous exceedance — revised 2026-09-01 from the O2 measurement. The earlier draft used
DS-MCTS's future-max clause, `max over j ≥ i`; for a *stopping* objective it is the wrong event:
the written target of a stop is the current distribution, later wandering that returns never
enters it, and the continuation's endpoint is exactly what `KL(pi_N || pi_ci)` already measures.
O2 measured the two labels indistinguishable for the realizable rule (holdout spend 1.361 vs
1.365), and the instantaneous form makes beta the literal corrupted-target rate:
`false stop ≡ written target ≥ eps from the reference`. The future-max flag stays in the audit
records as a diagnostic.)

with `KL` computed exactly as `src/search_budget/targets.py::policy_kl` (floor 1e-6, zero-mass terms
of the reference skipped). One audit position yields `K` labelled examples (one per checkpoint), all
from the same search.

Design notes, in order of importance:

- **Distribution convergence, not argmax stability (constraint 3).** DS-MCTS's
  `R(s,Nmax) − R(s,n) ≥ eps` certifies that the *chosen action* stays near-best. AlphaZero self-play
  trains on the full visit distribution and *samples* moves from it under temperature, so a position
  can be DS-certain while its distribution is far from `pi(.|s,N)` (e.g. two near-equal moves whose
  relative mass is still shifting: the argmax may be stable-ish while the 60/40 vs 45/55 split — which
  is the training target — is not). `KL(pi_N || pi_{c_j})` with the *full-search* distribution as the
  reference weights exactly the mass the training loss weights.
- **Why instantaneous, not the future-max clause.** "Close now" *is* "safe to stop now" for this
  objective: stopping writes the current distribution and plays from it — there is no future in
  the stopped branch, and divergence-then-return between checkpoints affects neither the written
  target nor the `pi_N` endpoint the KL already compares against. The future clause protects
  DS-MCTS's play-time criterion (the action chosen after *continuing*), not ours, and it inflates
  measured false stops with harmless cases.
- **Should argmax stability be an additional term?** Recommendation: **no term, but a tracked
  diagnostic.** Self-play move *selection* samples from the distribution, so a small-KL argmax swap
  changes the played game no more than temperature sampling already does, and the training target is
  distribution-valued. The one consumer that argmaxes is evaluation/arena play, which this plan leaves
  flat. Record `argmax(pi_{c_i}) ≠ argmax(pi_N)` per audit checkpoint in the analysis record; if the
  offline study shows stopped-eligible positions with argmax swaps at material rates (>2–3%), promote
  it to a conjunctive term. Building it in unconditionally would re-import DS-MCTS's play-time
  criterion that constraint 3 rejects.
- **Why the value term.** Since Syzygy removal (`02293f38`), a ply-capped game takes its value from a
  forced full search, and `root_value` enters replay for every position. A distribution can converge
  while the value estimate is still moving (long forced lines discovered late). `V(s,c_j)` is already
  in every checkpoint (`SearchCheckpoint.root_value`), so the term is free. `eps_v` should be generous
  (see 3.2); its purpose is to catch gross value drift, not to double the criterion.

### 3.2 Choosing eps

The v20 oracle table (measured at the 8x reference; Stage O2 re-derives it at the 2B reference, where
KLs shrink somewhat) makes the trade-off concrete rather than principled-but-vague:

| eps | mean spend | gain vs flat | effective compute |
|-----|-----------|--------------|-------------------|
| 0.05 | 1.279 | +35.6% | 1.43x |
| 0.10 | 0.956 | +26.7% | **1.64x** |
| 0.20 | 0.642 | +6.4% | 1.72x |

**Superseded 2026-09-01 by the O2 measurement — eps is anchored to the noise floor, not to
spend.** The O2 cross-seed floor (two capped searches differing only in noise seed) has median
0.103; pinning spend at 1.0 would force eps ≈ 0.02, demanding targets eight times more
reproducible than the target's own seed noise and discarding the entire saving, while at eps ≈
the floor the oracle spends 0.49 for a written-target perturbation below reseeding the same
search. Therefore: `eps_pi = clamp(noise_floor_multiple × median live paired-audit floor,
[eps_pi_minimum, eps_pi_maximum])`, with the floor measured continuously from a
`paired_audit_fraction` of audit positions searched twice with independent noise (fresh-root
pairs — an approximation in the warm frame, re-checked at the warmup gate). **Spend is an output**,
reported per generation and bounded from above only by the circuit breaker (section 6.2) — spend
falling toward 0.5 is the win, not a fault. A per-position eps (predicting each position's own
floor) was evaluated on O2 and rejected: at matched spend it is equal-or-worse on written KL
(0.061 vs 0.056) and argmax swaps (10.2% vs 9.9%), and excess swaps concentrate in *high*-floor
murky positions that per-position eps loosens further, not in the sharp positions it was meant to
protect (swap rate by predicted-floor quartile: 6.3/10.2/13.5/9.5%).

The paragraph below is retained for the record of the earlier (superseded) design:

**eps is an output of Stage O2, not an input of this plan.** The 0.10 row is the *provisional*
working point — it is the unit-spend point of the oracle table and the reason the design centers on
mean spend ≈1.0 — but reading a launch value off an oracle table at the wrong (8x) reference would be
arbitrary, and it ignores a systematic effect: **a real predictor held to a false-stop ceiling stops
strictly fewer positions than the oracle, so realized spend will come in above the oracle's 0.956,
and the launch eps will likely need to be *higher* than 0.10** to re-center spend at 1.0. That must
be an O2 finding, not a generation-30 surprise. Stage O2 therefore must produce, before eps is fixed:

1. the oracle table re-derived at the true `pi_2B` reference (the 8x frame overstates KLs);
2. the paired-seed noise floor: on ~2,000–5,000 positions, run the capped search twice with
   different noise seeds; `eps_pi` must sit at or above the median seed-to-seed
   `KL(pi_N^(a) || pi_N^(b))` — an eps below the target's own stochasticity is meaningless.
   `eps_v` comes from the p90 of the paired `|V^(a) − V^(b)|`;
3. the eligible-mass curve vs eps (what fraction of positions can stop, at which checkpoints);
4. the **realized** spend / effective-compute curve for a trained predictor behind the guard, held to
   the false-stop ceiling, at several eps values — the curve the launch eps is actually read from.

**Can eps be tuned during a run? Yes — and it is the run's one adaptive quantity** (user
decision, superseding this plan's earlier "no"). The earlier argument assumed a window of baked
labels; in fact the audit records store the raw per-checkpoint KLs and value gaps, so any eps can
relabel the whole window for free and nothing mixes. Section 6 specifies the per-generation
one-shot solve that pins realized mean spend at 1.0, clamped to `[eps_pi_minimum, eps_pi_maximum]`
— the clamp, not the solve, carries the quality guarantee, so O2's job shifts from choosing eps to
choosing the **clamp and beta**. `eps_v` and the movement guard epsilon stay fixed per run.

### 3.3 Ground truth generation and training-data collection

No separate labelling job. Audit positions are selected inside the self-play worker before each
search, deterministically by hashing the stable position identity (reuse the seeded-hash selection in
`src/search_budget/sampling.py::select_generation_sample`, called per position instead of per
generation batch). For an audit position the worker issues the search with
`policy_checkpoint_visits = {c_1..c_K}`, `checkpoint_detail = Policies`, stop rule in shadow mode.
The completed-game path already carries `checkpoints` through
`SelfPlaySearchResult` (`SelfPlay.hpp:98`); the Python side appends one audit record per position to
an append-only per-generation numpy file (the `analysis_log.py` pattern, new dtype: identity, ply,
`B`, per-checkpoint policy-KL-to-final, value-gap-to-final, all predictor input features
per checkpoint, predictor output `u_i`, shadow verdicts, argmax-swap flags, final visits). Failures
log and never affect self-play — same contract as today's analysis log.

**External anchor (review defect C).** The reference `pi_2B` is produced by the model being
trained, at visit counts (600–1200 new visits) that straddle the ~800-playout threshold below which
search barely leaves the prior. If the model collapses toward its prior on a class of positions,
the reference collapses with it: KLs shrink, labels certify stopping as safe, and realized
false-stop telemetry reads clean — the v20 failure mode made invisible to its own instruments. The
deleted 8x machinery was the only measurement outside that loop, so a vestigial anchor replaces it:
for ~5% of audit positions, the self-play worker additionally runs a fresh-root, noise-free search
at `4B` on the same position after the audit search completes, recording only
`KL(pi_4B || pi_capped_target)`. Cost ≈ 0.05 × audit_fraction × 2 extra search multiples ≈ +0.2%
visits. The tracked alarm is the *absolute* level and trend of that KL among stop-eligible
positions — an outside-the-loop drift detector, not a label source.

Because audits ride live searches, labels come from the production distribution (with tree reuse,
noise, current model) — no train/deploy gap. `sample_fraction` ~0.01–0.02 gives 5,000–15,000 audit
positions per generation at v20 game rates, i.e. 20,000–60,000 labelled checkpoint examples per
generation; a 10-generation trailing window is ample for a 2×64 MLP.

**Window segmentation at `baseline_visits` schedule steps (review defect G).** When the staged
schedule raises `B` (300→…→800→1000), the reference `pi_2B` deepens and every KL in the labels
shrinks systematically, so a window spanning a step mixes two label distributions and a fixed eps
is effectively loosened — the same mixing section 3.2 forbids for eps itself. The trailing window
therefore never crosses a `baseline_visits` schedule step: at a step it resets and the predictor,
the eps solve and the thresholds recalibrate on post-step data only (the warmup evidence minimum
applies again, during which the policy is `closed`). The eps solve re-derives eps on post-step data
automatically; the clamp `[eps_pi_minimum, eps_pi_maximum]` stays per-run, and a step whose solved
eps saturates the clamp is flagged in the generation report for the between-runs judgment.

Cost of auditing: checkpoint capture with `Policies` detail is memory and serialization only (the
policy-target vector K times for ~1–2% of positions); the schedulable-task drain at checkpoints
applies to audit and stop-eligible positions alike (section 5, overhead budget).

---

## 4. The predictor

### 4.1 Where it lives: external TorchScript MLP, not a trunk head

Rejected: a head on the trunk. A trunk head is computed once, at root expansion, from the root
position; it cannot see checkpoint-time tree state, which is the entire point (constraint 4's note).
Re-running the trunk at each checkpoint with tree-feature input planes would cost 4 extra batched GPU
evaluations per position (~+0.7% of visits — small) but, worse, would put a synchronous
inference round-trip inside the checkpoint drain, and would need a second model/refresh path through
`InferencePipeline`. Meanwhile the direct evidence from four production runs is that the trunk
contributes nothing recoverable to this kind of prediction while scalar observables carry all of it.

Chosen: a small MLP (2 hidden layers × 64, sigmoid output, ~16 inputs) exported with
`torch.jit.script`, loaded and evaluated on CPU by the native side — the exact seam that already
exists as `SearchBudgetCurveCorrector` (`cpp/src/search/SearchBudgetCorrector.hpp/.cpp`: pimpl,
probe-forward validation at load, CPU eval). Per-checkpoint cost is single-digit microseconds against
a millisecond-scale search; if it ever matters the MLP is trivially hand-evaluable without libtorch.

### 4.2 Input features (fixed order; a binding contract like the corrector's)

Per checkpoint `c_i`, all computable from the root node and the previous checkpoint:

| # | feature | rationale |
|---|---------|-----------|
| 1 | top share of current policy-target distribution | strongest single signal in every ablation |
| 2 | entropy of current policy-target distribution | ditto |
| 3 | top1−top2 share gap | separates "one clear move" from "two contenders" |
| 4 | `KL(pi_{c_i} || raw prior)` | how far search has moved off the prior — the KataGo deviation measure |
| 5 | `KL(pi_{c_i} || pi_{c_{i-1}})` | recent movement: the finite-difference version of the very quantity being predicted (DS-MCTS temporal channels, collapsed to a scalar). For `i = 1` the previous distribution is a **zeroth checkpoint `c_0` captured at `starting_visits`** — the retained warm-root distribution, or the raw prior only on a genuinely fresh root. (Review defect D: comparing against the prior on a warm root fails the guard for exactly the converged-but-prior-disagreeing positions — the most valuable true stops — and biases early stops toward prior-shaped targets.) |
| 6 | top share of the *latest-segment* distribution `(n_i·N_{c_i} − n_{i-1}·N_{c_{i-1}})/(n_i−n_{i-1})` over **raw cumulative visit counts** retained on the task | DS-MCTS channels 4–6, scalarized: is the recent search still choosing the leader. Review defect F: forced-playout-pruned `policyTargetVisits` are not cumulative (a move pruned at `c_i` but present at `c_{i-1}` yields negative mass), so the task retains the raw per-child visit counts of the previous checkpoint — never the pruned normalization — for this feature. |
| 7 | `V(s,c_i)` | value context |
| 8 | `V(s,c_i) − V(s,c_{i-1})` | value trend |
| 9 | `V(s,c_i) −` network root value | how much search has corrected the static eval |
| 10 | raw-prior top share | reuse `rootPriorFeatures` (`SearchExecutor.hpp:433-454`) |
| 11 | raw-prior entropy | ditto |
| 12 | legal-move count | normalizes 2 and 4 |
| 13 | ply | proven feature |
| 14 | `baseline_visits` (B) | proven feature |
| 15 | model generation | proven feature; standardization folded into the export as in the corrector |
| 16 | checkpoint multiple `m_i` | see below |
| 17 | root warmth `starting_visits / B` | review defect D: the vector carries `m_i`, `B`, ply and generation but no measure of how warm the reused root is, and warmth changes what a checkpoint at `m_i·B` *new* visits means; scale-freeness was already given up by including `m_i` |
| 18 | support count of the checkpoint distribution | strongest single predictor of the position's noise floor in the O2 pair study (Spearman +0.49); free at a checkpoint |
| 19 | top-3 share of the checkpoint distribution | floor-study addition; with 1 and 3 it summarizes the head of the distribution (raw top-k vectors were considered and rejected: scalars already saturate the recoverable signal, per the v17–v19 ablations and the O2 per-position study) |

On including `m_i`: DS-MCTS deliberately excludes `n` and `Nmax` so one model trained at a single
checkpoint generalizes to others and to larger `Nmax`. We do not need that generalization — labels
exist at *every* checkpoint (each audit search yields all K examples), the checkpoint set is fixed by
configuration, and per-checkpoint thresholds already break scale-freeness. Including `m_i` lets one
model specialize per checkpoint without training K models. Features 1–11 are scale-free ratios, so
the model still transfers across the staged `baseline_visits` schedule (300→600+).

Class imbalance and per-checkpoint reweighting are handled as in the paper (subsample the easy
majority per checkpoint); decided by the Stage-O data, not up front.

### 4.3 Training, validation, export

Fit Python-side per generation on the trailing audit window (default 10 generations), BCE loss, Adam
1e-3, batch 4096, ≤30 epochs — the `fit_curve_corrector` /`export_corrector` machinery in
`src/search_budget/corrector.py` carries over nearly line-for-line (different feature vector, BCE
head, sigmoid clamp instead of correction clamp), with one deliberate departure: **the holdout is
split by game identity, not by row stride** (review defect E). One audit search yields K sibling
examples with deterministically nested labels (`U_i` is monotone in `i`), and positions of one game
are correlated; a stride split puts siblings on both sides and inflates the holdout gate, the
publish decision and the O2 curves. Fail-closed rejection rules, extending the corrector's:

- any non-finite parameter or buffer → reject;
- holdout BCE not better than predicting the window base rate → reject;
- **operational check:** at the thresholds the section-6 solve would publish, the holdout
  false-stop rate must be ≤ beta and the implied visit saving ≥ a configured floor (a predictor
  that saves nothing must not be published just because its BCE improved);
- on rejection the previous published predictor (or "never stop") stays referenced and the rejection
  is logged — identical contract to the corrector's.

Export: `torch.jit.script` → `stop-predictor-generation-XXXXXXXX.jit.pt` under a
`search-stopping/` run directory, referenced by path + sha256 in the published policy; probe-forward
validation on load on both sides (existing pattern).

---

## 5. The stopping algorithm in the native search

### 5.1 Types (rework of `cpp/src/search/SearchTypes.hpp`)

```cpp
struct SearchStopPolicy {
    std::vector<double> checkpoint_multiples;      // strictly increasing, in (0, cap_multiple)
    std::vector<double> thresholds;                // same length; u < thr[i] stops at checkpoint i
    double movement_guard_epsilon;                 // observed KL(pi_ci || pi_c(i-1)) must be below this
    double cap_multiple;                           // fixed Nmax as a multiple of B; 2.0 at launch
    std::shared_ptr<const SearchStopPredictor> predictor;  // null only with apply_learned=false
    bool apply_learned;                            // false = CLOSED: flat search to B, no cap
};

struct StoppableSearchLimit {                      // replaces PredictedSearchBudgetLimit
    std::uint32_t baseline_visits;                 // B (additional visits, as today)
    SearchStopPolicy policy;
    std::uint64_t model_generation;
    bool shadow_only;                              // audit positions: record verdicts, never stop
};
```

Validation in constructors mirrors `SearchBudgetPolicy`'s (finite, ordered). **There is exactly one
closed state and one name for it** (review defect B — under a 2x cap, "never stop" is ambiguous
between two states whose spends differ 2x, the inverted-projection defect class):

- **`closed` = `apply_learned = false`**: a flat search to `B`, no checkpoints, no cap, bit-identical
  to the baseline configuration. This is the *only* fail-closed state; every structural failure,
  the warmup period, and the calibrator's global fallback all land here. It never spends 2x.
- **`attenuated at checkpoint i`** = `apply_learned = true` with `thresholds[i] = 0`: checkpoint `i`
  stops nothing but the cap and other checkpoints stay active. This is a *degraded open* state, is
  never called "closed", and is only reachable while at least one checkpoint still has a safe
  threshold — the calibrator publishes `closed` outright when none does, because an open policy
  that stops nothing anywhere is a ~2x compute burn, not a safe default.

Audit searches are the one exception: they run to the cap by explicit request even under a `closed`
policy, because labels and the warmup re-derivation (section 10) need them — at the audit fraction
that is ≤ +2% visits, not 2x.

### 5.2 Executor changes (`cpp/src/search/SearchExecutor.hpp`)

The load-bearing reusable machinery is the checkpoint seam, and it already does the hard part:

- `schedulableTask` (lines 585-593) never schedules a leaf past the next checkpoint, so the tree
  arrives at each checkpoint *exactly* with `in_flight == 0`;
- `updateCheckpointsAndStop` (lines 417-429) fires when `root.visits()` equals the checkpoint.

Insert the stop decision there: when a checkpoint at index `i` is captured and the task's limit is a
`StoppableSearchLimit` with `apply_learned`, compute the feature vector (extend
`rootSelectionFeatures`, lines 460-495, with the checkpoint-delta features; the task retains the
previous checkpoint's distribution, which it already stores in `task.checkpoints`), and:

1. **Observational guard first:** if the measured movement `KL(pi_{c_i} || pi_{c_{i-1}})` is not
   below `movement_guard_epsilon`, the position cannot stop at `c_i` — no predictor call. For
   `i = 1` the previous distribution is the zeroth checkpoint at `starting_visits` (review defect
   D), so a warm root's guard measures movement of *this* search, not disagreement with the prior.
   The guard is a cheap necessary-condition filter, no more: KL is quadratic in small perturbations,
   so a monotonically drifting search passes every local check while accumulating large total
   drift, and mass that moves and returns between checkpoints is invisible to it. What it does
   buy: a predictor gone wrong can never stop a position whose distribution is *visibly* still
   moving over the last interval — one concrete v20 failure channel, not all of them. The labels,
   the ceiling on false stops and the external anchor carry the actual safety argument.
2. Otherwise evaluate the predictor; non-shadow and `u < thresholds[i]` → `task.stopped = true`,
   `stop_reason = SearchStopReason::LearnedEarlyStop`, record `stop_checkpoint_index = i`.
3. Always record the guard value and `u` into per-checkpoint vectors on the task, returned in the
   result (audit shadow verdicts come from this — no second code path).

Deleted from the executor: `assignReadyPredictedBudgets` (lines 497-535) and the budget-assignment
task states (`budget_assigned`, `selected_budget_index`, `spend_residual`); parallelism reverts to
`searchParallelism(B)` computed at task creation (the limit is known up front again — no more
16-then-reassign special case, line 347-348).

`GameSearchResult` (SearchTypes.hpp:266-289) drops `predicted_budget_curve`,
`selected_budget_index`, `assigned_additional_visits`, `spend_residual`, `root_prior_top_share/_entropy`
(the raw-prior features move into the recorded feature vector) and gains
`stop_checkpoint_index` (−1 = ran to limit), `stop_probabilities`, and keeps
`final_visits`/`starting_visits`/`stop_reason`/`checkpoints`.

### 5.3 Checkpoint-drain overhead

The drain (scheduling fence at each checkpoint) now applies to *all* stop-eligible self-play searches,
not only labelled ones. With parallelism ≤16 and 4 checkpoints the worst case is 4 short pipeline
stalls per move; across 512-position inference batches drawn from many concurrent games the stalls of
different games interleave, so the expected cost is small but not zero. Budget: **≤2% self-play
cadence overhead with stopping disabled but checkpoints active**, measured in Stage N (Release build,
bench under `documentation/benchmarks/`); if it exceeds budget, drop to 2–3 checkpoints before any
other mitigation. This measurement also isolates the mechanism cost from the policy effect in the
production A/B.

### 5.4 Bindings and contract changes

- `SearchBindings.cpp:161-216`: replace `SearchBudgetPolicy` / `SearchBudgetSelectionFeatures` /
  `PredictedSearchBudgetLimit` / `select_budget_index` / `correct_budget_curve` bindings with
  `SearchStopPolicy` / `StoppableSearchLimit` / a `stop_decision(policy, features)` helper exposed for
  Python-side tests. `SearchStopReason` gains `LEARNED_EARLY_STOP`, loses `PREDICTED_BUDGET`
  (lines 144-147).
- **Inference contract shrinks from 3 tensors to 2.** The TorchScript output tuple loses the
  search-budget tensor: `InferencePipeline.cpp` lines 127-138 (refresh validation), 363-364 and
  563-591 (staging allocation/validation), 391-424 (staging copy), 803-804 (narrow);
  `InferenceTypes.hpp:16-18` (`SEARCH_BUDGET_CURVE_POINTS`, `SearchBudgetCurvePrediction`) and `:44`
  (`search_budget_curve` in `SearchInferenceResult`); `SearchTree.hpp:110`. Python export side:
  `training/network.py` `InferenceNetwork.forward` (line 374) returns 2 tensors,
  `ZeroSearchBudgetHead` (346-348) and `_search_budget_head` (536-541) deleted. This is a binding
  contract change → Python-side test required (`cpp/AGENTS.md`), and old checkpoints' exported
  inference models become incompatible — acceptable for new runs, called out for the user since it
  forecloses warm-starting a stopping run from a v2x exported model without re-export.
- `SelfPlay.hpp`: `SelfPlaySearchParameters` (21-56) carries `SearchStopPolicy`; `m_budgetAllocator`
  (255) and its plumbing (161, 207, 212) deleted; `SelfPlaySearchRequest` gains the audit flag;
  arena capacity uses the `2B` cap when the gate is open and `B` when closed
  (`maximumArenaCapacity`, lines 41-45 — shrinks 4x from today's 8B reservation).

---

## 6. Spend-pinned eps, fixed beta, and policy publication

**One adaptive quantity (user decision, revision 2 of defect B).** An earlier draft fixed `eps_pi`
for the life of a run and added a self-tuning per-checkpoint threshold calibrator on the
resignation pattern. Both halves are superseded. Two coupled adaptive loops on one system —
eps moving the certainty definition while beta moves the thresholds — interact (loosening eps
shifts the false-stop base rate, which moves beta, which moves spend, which moves eps), and the
premise that eps cannot move mid-run was simply wrong: the audit records store the **raw**
per-checkpoint `KL`-to-reference and value gaps, not labels, so relabelling the entire trailing
window under a new eps is a free recomputation on stored numbers — no new searches, no mixed
window. The design is therefore:

- **`eps_pi` is anchored per generation to the measured noise floor** (superseding the earlier
  spend-pinned solve — see section 3.2): `eps_pi = clamp(noise_floor_multiple × median paired-audit
  cross-seed KL, [eps_pi_minimum, eps_pi_maximum])`, computed by
  `solver.solve_noise_floor_anchored_eps` on the trailing window. Spend is an output, bounded
  above only. `noise_floor_multiple` is the single tuned scalar (O2 suggests 0.5–1.0; the argmax
  data favors ≤0.75, the spend data 1.0); `eps_v` and `movement_guard_epsilon` stay fixed per run.
- **The clamp `[eps_pi_minimum, eps_pi_maximum]`** guards against a corrupted or drifting floor
  measurement in either direction; clamping is reported in the generation report.
- **`beta` (`false_stop_rate_ceiling`) is a fixed configured value** chosen from the O2 curves
  (initial suggestion 0.10: under instantaneous labels beta is the literal rate of stops whose
  written target exceeds eps, and the O2 holdout frontier puts flat-equivalent quality at
  beta ≈ 0.10, spend ≈ 0.75; 0.01 was measured unreachable — thresholds collapse and the cap
  makes spend exceed 1). Per checkpoint, per generation, the threshold is a **stateless
  solve**, cheapest checkpoint first on the simulated-survivor population: the largest threshold
  whose trigger count is ≥ `minimum_evidence_trigger_count` and whose one-sided binomial upper
  bound on the false-stop rate (`src/util/binomial.py::one_sided_binomial_upper_bound`, at
  `confidence_level`) is ≤ beta. A checkpoint with no qualifying threshold is *attenuated*
  (threshold 0, stops nothing) while other checkpoints still stop; when **no checkpoint anywhere**
  qualifies — or the predictor is rejected — the publication is the `closed` policy (flat to `B`),
  never an open policy that stops nothing, which under the 2x cap would silently double spend.
  There is no walk-back state machine, no candidate grid stepping, no relaxation schedule and no
  journal: the resignation-mirror calibrator, its state, its telemetry and its tests are deleted
  from the design. What survives of it is the binomial bound and the asymmetric principle above.
- **Configuration keys** (explicit, no implicit defaults): `eps_pi_minimum`, `eps_pi_maximum`,
  `false_stop_rate_ceiling`, `minimum_evidence_trigger_count`, `confidence_level`,
  `first_production_generation` (the warmup: the policy publishes `closed` until enough audit
  generations exist — suggest 10) and `maximum_realized_mean_spend` (the circuit breaker, 6.2).

Ordering within a generation: the predictor is refit once under the previous generation's eps
(its `u` is a ranking, robust to small eps drift); the eps solve then reuses that predictor's
recorded `u` values across candidate eps, and the published thresholds are the solve's output at
the chosen eps. The realized false-stop rate remains **measured** and reported, bucketed by ply
and entropy (section 8), even though beta is fixed — it is the primary quality telemetry and the
trigger for changing `eps_pi_maximum` between runs.

What this loop can and cannot do, stated so nobody mistakes one for the other: it holds spend at
1.0 within the quality clamp and holds false stops under beta. It cannot tell us the mechanism is
**worth** anything: 0% false stops at 1.02x effective compute passes every ceiling and is still a
failure. Worth is decided *offline*, once, by the Stage-O2 go/no-go bar (≥1.40x, section 10)
before any production run is spent; inside the run the only worth-tracking is telemetry (section
8) feeding the between-runs decision.

### 6.2 Publication

Per finalized audit generation the manager publishes `{checkpoint_multiples, thresholds per
checkpoint, movement_guard_epsilon, predictor path+sha256, apply_learned}` for the next *unstarted*
generation through the existing publication state machine
(`src/search_budget/calibration.py::publish_fail_closed`, `publication_for_generation`,
`load_calibration_state_fail_closed` — the one calibration structure that survives deletion,
reworked payload). Fail-closed publishes the `closed` policy (flat to `B`, bit-identical, tested)
and applies to **structural** failures — unreadable or sha-mismatched state (the config-pin defect
class), a predictor that fails its load probe, an audit pipeline lagging beyond
`maximum_unstarted_generation_lag`. Rate excursions are *not* structural failures; they are handled
by the calibrator's walk-back above.

**Spend circuit breaker (review defect B): a bound, not a controller.** Configuration key
`maximum_realized_mean_spend` (suggest 1.3): if a finalized generation's realized mean visit
multiple exceeds it, the next publication is `closed` regardless of threshold safety, with decision
reason `SPEND_BREAKER`, until a subsequent calibration produces thresholds whose *simulated* spend
on the window is back under the limit. This is a one-sided safety bound evaluated on a measurement,
with no target, no stepping and no state — it does not reintroduce the dual; it converts "a human
notices the 2x burn between runs" into "the run caps its own downside at one generation".

The three historical defects, addressed by construction: **config pin** — same sha-guarded state
loading, re-evaluated every generation rather than latched; **inverted isotonic projection** — there
is no monotonic-transform code left anywhere in the decision path; **stale dual** — there is no
dual, and threshold selection is a fresh evidence evaluation each generation whose staleness is
bounded by the lag rule.

---

## 7. Policy targets of early-stopped positions (the crux, constraint 5)

**Recommendation: use the stopped position's policy target at full weight.** The argument: KataGo's
split exists because their fast searches are *unconditionally* fast — the target is degraded whenever
the position needed search. Here a position is stopped only when the guard has observed the
distribution settled and the criterion certifies `KL(pi_N || pi_stop) < eps_pi` against the **2B cap
distribution**. Under the capped design every position's target is, up to eps and false stops, at
least as good as flat's: stopped positions carry a target certified within eps of the *deeper* 2B
distribution (≥ the flat `pi_B` target in expectation), and unconverged positions run past `B` toward
`2B` and get strictly deeper targets than flat — this is where the oracle's +26.7% lives. The
targets-drift-toward-the-prior failure mode is excluded up to (a) eps and (b) false stops — and (b)
is what the guard, β, the audit stream, and the gate control.

What makes this not naive optimism: the certification is a *prediction*, and false stops concentrate
precisely on positions that look converged but are not (deep tactics). Three defenses:

1. The thresholds are self-tuning against ground-truth labels every generation on
   production-distribution audits (section 6.1): a realized false-stop rate whose binomial upper
   bound crosses the ceiling walks the threshold back promptly — the resignation-calibrator pattern,
   attenuation instead of a kill switch.
2. Clustered blind spots are checked, not just the average: realized false-stop rate is reported per
   ply bucket and per entropy bucket in the generation report (the `CurvePointReport` slot in the
   manager report is repurposed). A rule that is 1% wrong on average but 15% wrong in sharp
   middlegames would pass a mean check and still poison training; bucketed telemetry catches it.
3. The per-generation yardstick (recovery plan §1) remains the run-level pass/fail; a stopping run
   that drifts below yardstick is stopped on existing rules regardless of its internal metrics.

**`use` is implemented; there is no `stopped_policy_target` configuration key** (user decision — one
behaviour, clean code, no option carried "just in case"). The KataGo split (`exclude`: drop the
stopped position's row from replay ingestion; the game still advances, positions searched past the
stop are unaffected) remains the documented fallback, but as a *code change made if and when* the
production A/B trails its control — the self-play worker already knows `stop_reason` per position
(`worker.py` maps stop reasons), so the change is a small filter at completed-game assembly, cheap to
write on the day it is earned. A middle setting (down-weighting via `sample_weight`) is deliberately
not considered: it blurs the A/B interpretation, and `use` vs `exclude` brackets it from both sides.

---

## 8. Measuring the saving, and keeping it honest

- **Spend:** realized mean visit multiple `(final_visits − starting_visits) / B` per generation,
  overall and per checkpoint; fraction stopped per checkpoint; fraction running past baseline. Design
  center is 1.0; sustained drift beyond ±10% is flagged in the generation report and corrected
  between runs via eps or the cap — never by an online controller. Wall-clock cadence (games/hour,
  generation duration) must hold at the flat control's level, since at unit spend the mechanism's
  claim is better targets at equal throughput (this is where the predecessor's +10.7% KL evaporated).
- **Training-data quality:** realized `KL(pi_N || pi_stop-would-have)` distribution on audit
  positions (mean, p95), realized false-stop rate (overall + bucketed), argmax-swap rate among
  stop-eligible audits, and the external-anchor series `KL(pi_4B || pi_capped_target)` (section
  3.3) in absolute terms — the drift-toward-prior watch. (An earlier draft watched
  `KL(pi_stop || raw prior)` against the *window median*; review defect C: a relative watch drifts
  with the collapse it is meant to detect, so the anchor is absolute and outside the training loop.)
- **Run-level:** the per-generation yardstick and the 64-search Stockfish ladder at matched
  wall-clock against a concurrently launched flat control — the only metric on which the idea is
  ultimately judged (v13–v18 lesson: retire only on Elo evidence, and never accept KL-proxy wins as
  success).
- All measurements carry `experiment_configuration_sha256`; benchmark numbers land under
  `documentation/benchmarks/adaptive-stopping-<hardware>-<date>/README.md` with config SHA, source
  SHA, node and raw numbers, per the evidence rules.

---

## 9. Deletion inventory (user decision: delete outright; only what stopping needs survives)

The predecessor's record is `77bcfe96` plus
`documentation/analysis/adaptive-search-budget-negative-result-20260901.md`; nothing needs to survive
in the tree for history. **`py/src/search_budget/` is deleted as a package.** New code lives in
`py/src/search_stopping/`. No compatibility shims, no dead configuration keys, no retained-but-unused
modules: the `search_budget:` block leaves the configuration schema entirely (v10–v20 configuration
files stay in the tree as historical artifacts and will no longer resolve — accepted, breaking
changes are fine until the final run).

**The complete list of survivors, each moved into `py/src/search_stopping/` because stopping
load-bears on it — everything else in the package goes:**

| survives from | what, and why stopping needs it |
|---|---|
| `targets.py` | `policy_kl`, `policy_entropy`, `top_visit_share`, `PolicyDistribution` — the label math of section 3.1. (`shadow_gain` does not survive.) |
| `corrector.py` | the fit/holdout/rejection/`torch.jit.script`-export/probe-validation machinery, refitted as the stop-predictor trainer (BCE head, new feature vector, operational gate per 4.3) — the one component of the old subsystem that measurably worked. |
| `calibration.py` | only the publication state machine (`publish_fail_closed`, `publication_for_generation`, `load_calibration_state_fail_closed`, decision-reason reporting, the `maximum_unstarted_generation_lag` freshness rule) with the stop-policy payload. The lambda solver, sigma/spend EMAs, trust/reseed ratios, evidence models die with the dual. |
| `sampling.py` | seeded deterministic identity sampling, re-targeted to per-position audit selection. |
| `artifacts.py` | atomic immutable artifact writes and persisted-model IO (predictor export/load). Shard-coverage validation does not survive (no shards). |
| `analysis_log.py` | the fixed-width append-only per-position record pattern, new dtype per 3.3. |
| `resignation.py` (referenced, not moved) | the calibrator pattern section 6.1 mirrors — `one_sided_binomial_upper_bound` becomes a shared utility if not duplicated. |

Deleted with no successor: `policy.py` (Lagrangian selection, isotonic projection, curve constants —
the checkpoint multiples are redeclared in the new configuration, not inherited), `labeling.py`
(fresh-root deep-search reconstruction, shard artifacts, `finalize_generation`, replay writeback),
`worker.py` (GPU-claimed label workers), `manager.py` (job queue, shard orchestration, retry,
cleanup receipts — the successor manager is written fresh at a tenth the size: fit + calibrate +
publish + report), `retention.py`, `configuration.py` (replaced by `SearchStoppingConfiguration`:
`audit_sample_fraction`, `checkpoint_multiples`, `eps_pi`, `eps_v`, the calibrator keys of 6.1,
`window_generations`, `maximum_unstarted_generation_lag` — all explicit, no implicit defaults, no
`stopped_policy_target` key per section 7).

### Python — training/replay/self-play seams

| location | change |
|---|---|
| `training/network.py:346-348, 359, 363, 374, 536-541, 817` | delete search-budget head, `ZeroSearchBudgetHead`, head layout case; inference export returns `(policy, wdl)`. |
| `training/objective.py:86-147, 225` | delete `ResolvedSearchBudgetLoss`, dedicated-batch weighting, masked Huber. |
| `replay/contracts.py:72-114`, `replay/columnar.py`, `replay/layout.py`, `replay/store.py` | delete `EligibleSearchBudgetTarget`/`IneligibleSearchBudgetTarget` and the `AUXILIARY_VALUE`/`AUXILIARY_RAW_KL`/lineage columns (replay layout version bump). Add nothing: stopped-target exclusion is a filter before ingestion, not a column. |
| `self_play/worker.py, parameters.py, protocol.py, process_runtime.py` | `SearchBudgetPolicy` publication seam becomes `SearchStopPolicy` publication (same generation-boundary refresh cadence); spend-residual plumbing deleted; audit-record emission added. |
| `training/search_budget_tensorboard.py`, `training/telemetry.py`, `reporting.py` | replace budget dashboards with stopping dashboards (section 8 quantities). |
| `training/coordinator.py`, `self_play_group.py`, `trainer/*` | remove label-manager wiring and dedicated-batch scheduling; wire the (much smaller) stopping manager. |

### Native — `cpp/src/search/`

| file | change |
|---|---|
| `SearchTypes.hpp` | delete `SearchBudgetPolicy` (65-104), `projectNonIncreasing` (109-115), `correctBudgetCurve` (117-142), `selectBudgetIndex` (147-166), `PredictedSearchBudgetLimit` (168-182), `SearchBudgetAllocator` (204-243), `AssignedSearchBudget`. Add `SearchStopPolicy`, `StoppableSearchLimit`. Retain `SearchCheckpoint` (36-40), `SearchCheckpointDetail`, `searchParallelism`, `GameSearchResult` (fields per 5.2). |
| `SearchExecutor.hpp` | delete `assignReadyPredictedBudgets` (497-535) and budget task state; retain and extend the checkpoint seam (417-429, 585-593) and `rootSelectionFeatures`/`rootPriorFeatures` (433-495) into the feature builder; add the stop decision. |
| `SearchBudgetCorrector.hpp/.cpp` | **retain, rename** `SearchStopPredictor`: same pimpl/libtorch-CPU/probe-validation structure, new input/output arity. |
| `InferencePipeline.cpp/.hpp`, `InferenceTypes.hpp` | 3-tensor → 2-tensor contract (5.4). |
| `SearchBindings.cpp` | replace budget bindings (161-216), stop-reason enum (144-147), result fields (217-239). |
| `SelfPlay.hpp`, `SelfPlayBindings.hpp` | policy type swap, allocator removal, audit flag, result fields (5.4). |
| `Analysis.hpp`, `SearchEngine.hpp`, benchmark tools, native-gated tests | mechanical follow-through on the seam (the recent commits `c1246cae`/`b3038e03` show the touch set). |

Flip-harness note: no encoding or action-id change anywhere in this plan, so the colour-symmetry
harness is not triggered; it runs anyway before the production stage as standard hygiene.

---

## 10. Staged validation plan

**Stage O — offline study, before any implementation and before any GPU time beyond one authorized
hour.** Two parts.

- **O1, done in substance (v20 measurement, 2026-09-01):** the oracle table in section 1.2 is the
  feasibility read this stage existed for. It establishes the ceiling (1.64x effective at unit
  spend, eps 0.10), kills the pure-saving regime, and mandates the 2x cap. The remaining free work
  on local data (v17–v19 analysis records, ~739k positions; plus the v20 records): the
  ply/entropy structure of the eps-eligible mass, the Equ.-7-style checkpoint-subset scan, and a
  first feature-signal check (fit the stop MLP on 8x-referenced proxy labels, measure AUC and the
  false-stop/saving trade at proxy scale). Note the constraint verified during planning: the bulky
  deep-search shard artifacts (full checkpoint distributions) were removed on-node by
  `artifact_retention: remove_bulky_after_finalization` in every fetched archive and in live v20's
  config, so exact 2B-referenced labels cannot be computed from data we currently hold.
- **O2, ~1 GPU-hour, authorized and collected 2026-09-01:** a standalone driver on the node (a
  scratch script, *not* the deep-label machinery — that machinery hard-rejects root noise and warm
  roots, `labeling.py:177` / `worker.py:251`, which is review defect A) ran 37,000 single + 5,000
  paired positions from recent v20 replay games at generation-365 settings (B=800): fresh root,
  **root noise ON, forced playouts ON**, searched to the 2B cap with the policy-target distribution
  recorded at 9 checkpoints (0.125…1.75×B). The paired searches differ only in their Dirichlet
  noise draws, so the noise floor is measured with noise, as it must be.

  **O2's scope is fresh-root feasibility (review defect A, resolution (b)).** Production searches
  run on warm roots (0.6 retained visit fraction) with noise; `m_i·B` *new* visits on a warm root
  is a different evidence level than on a fresh root, varying per position. Rather than modifying
  the production worker contract before the mechanism has earned it, the fresh-root O2 numbers
  gate only whether to *build and try* (the go/no-go below), and every deployed operating quantity
  is re-derived in the true production frame during Stage-P warmup, where it is free: warmup audits
  are full production searches (warm root, noise, current model) run to the cap with labels but no
  stopping. **Warmup re-derivation gate:** before `apply_learned` first publishes true, the
  warm-frame audit stream must reproduce the O2-derived eligible-mass-vs-eps curve and noise floor
  within a stated tolerance (suggest: warm-frame eligible mass at the chosen eps within ±30%
  relative of the O2 curve, and the warm-frame noise-floor median not above the chosen eps); if it
  does not, eps and the guard epsilon are re-chosen from the warm-frame curves — the O2 frame is
  never deployed, only used to decide whether deployment is worth attempting.

**Go/no-go gate — throughput axis (revised 2026-09-01 with the O2 measurement; the earlier
1.40x-effective-compute bar was framed on the quality axis and is superseded).** Evaluated at the
warm-frame warmup re-derivation gate before `apply_learned` first publishes true: **realized mean
spend ≤ 0.80** with (a) realized `P(written ≥ eps | stop) ≤ 0.10`, (b) p90 written ≤ the live
paired-audit floor median, and (c) mean written ≤ 1.25 × mean `KL(pi_2B || pi_B)` (flat
equivalence within 25%). 0.80 is chosen because the fresh-frame O2 holdout already delivers 0.75
under (a)–(b), leaving margin for warm-frame surprises, and anything above 0.80 is <1.25x search
throughput — below the complexity bar. If the gate fails, Stage P is not spent, the study is
written up, and the adaptive-search line is retired (final run on v13/v17-era configurations).
This bar exists once, at the gate; inside the run the quality control is the threshold solve and
the spend bound is the circuit breaker (6.2). Everything before Stage P costs at most one
authorized GPU-hour plus local compute.

**Stage U — Python unit tests** (`py/test/test_search_stopping_*.py`, importlib mode): label
extraction from checkpoint records including the future-max clause; eps boundary semantics; the
stateless threshold solve (survivor-sequential evidence accounting, binomial upper bound,
minimum-evidence and attenuation semantics, global fallback to the `closed` policy); the eps solve
(relabel-the-window equivalence, spend monotonicity, clamp saturation and its report); the spend
circuit breaker; predictor-fit rejection rules;
publication state machine fail-closed transitions (unreadable state, sha mismatch, lag) —
parametrized over the failure modes.

**Stage N — native tests and bench.** New suite in the single `NativeTests` executable (runner in
`test/TestRunner.hpp`, registered in `test/TestMain.cpp`, sources added to the target — no per-suite
executable). Tests: checkpoint arrival with `in_flight == 0` and exact visit counts under parallelism;
`apply_learned = false` / null predictor / empty thresholds each bit-identical to
`AdditionalSearchLimit(B)` (the fail-closed identity, the single most important test); shadow mode
records verdicts and never stops; feature vector against a hand-constructed tree (golden values);
TorchScript predictor probe validation and load-failure handling; 2-tensor inference contract —
with the matching Python-side binding test. Bench: checkpoint-drain overhead (5.3) in Release on the
node, recorded under `documentation/benchmarks/`.

**Stage P — production A/B, user-authorized.** One stopping run (`use` targets) against one flat
control at matched wall-clock on the 8× RTX 4070 SUPER node, judged on the per-generation yardstick
and the 64-search ladder. The policy publishes `closed` (flat to `B`) for the warmup generations
by construction (`first_production_generation`); only audit searches run to the cap during warmup,
and the warmup re-derivation gate (Stage O2 section above) must pass before `apply_learned` first
publishes true. If the stopping run trails, the `exclude` fallback is implemented as a code change
(section 7) and rerun once. If both trail, the idea is retired on Elo evidence, the findings memory
updated, and the final run uses v13/v17-era configurations.

Branch: `adaptive-stopping`, rebased on `master`, feature-sized commits after validation, merged only
after the Stage-P acceptance. The two approved-but-deferred master fixes (with-replacement labelled
batches, duplicate ladder-Elo write) intersect this work — the labelled-batches fix dissolves
(dedicated batches are deleted); the ladder-Elo fix should land on `master` first.

---

## 11. Risks

1. **The headroom is thin and known: 1.64x oracle vs a 1.22x live system that already lost.** The
   most likely failure is not that the mechanism breaks but that even a clean 1.4–1.6x effective
   compute is worth < 60 Elo at these operating points, in which case the predecessor's deficit was
   never explainable by target degradation alone and this family of ideas is exhausted. The Stage-O
   go/no-go gate and the Stage-P A/B are designed to reach that verdict at minimum cost.
2. **The 2B-referenced labels may look materially different from the 8x-referenced table** (both
   directions: eligible mass grows because the reference is nearer, but the gain-vs-flat also
   shrinks for the same reason), and the O2 data is additionally in the fresh-root frame while
   production is warm-root (review defect A). The eps recommendation is provisional until O2
   re-derives the table, and the deployed values are provisional until the warm-frame warmup
   re-derivation passes; no production decision rests on the 8x frame, and no *activation*
   decision rests on the fresh-root frame.
3. **Spend variance does not convert to cadence cleanly.** Mean spend 1.0 with high per-move
   variance changes batching patterns (stopped moves free pipeline slots that extended moves
   consume unevenly); the drain overhead (≤2% budget) eats from the same account. Stage N measures
   the mechanism cost; Stage P measures conversion. Treat cadence, not visits, as the accounting
   unit — this is where the predecessor's headline number evaporated.
4. **Clustered false stops** (tactically deep positions that look converged). Bucketed realized
   false-stop telemetry (7.2), β with headroom, and the fallback split.
5. **Certified targets still underperform** — eps-close in KL may not be close in training-signal
   terms (the criterion certifies similarity to `pi_B`, not usefulness of `pi_B`). This is the
   residual unknown the A/B exists for; the `exclude` fallback fully de-risks target quality at the
   cost of sample count.
6. **Label future-max clause is grid-coarse:** divergence between checkpoints is invisible, so labels
   are slightly optimistic about certainty. Bounded by eps headroom and by the fact that stops can
   only occur at the same grid the labels observe.
7. **Predictor trained on shadow audits shifts once stopping is live** (survivors reaching late
   checkpoints are a harder population). Sequential calibration on simulated survivors (§6.2) models
   this; audits remain unbiased because they ignore stops.
8. **Concurrency with the negative-result writeup:** no file under `py/src/search_budget/` or the
   benchmarks note is touched until that lands; new code lives in `py/src/search_stopping/` and the
   deletion inventory (section 9) executes as its own later commit.

## 12. Decision record and remaining open items

Decided by the user (2026-09-01 review): `Nmax = 2B`; no online spend controller; CPU TorchScript
predictor; audits ride live self-play; 3→2-tensor contract break accepted; Stage O2 authorized
(node 38.49.42.120:53893 free, v20 stopped checkpoint-safe); self-tuning threshold calibrator on the
resignation pattern instead of any runtime kill; `use` for stopped policy targets with no
configuration key; old subsystem deleted outright; five-checkpoint set confirmed by the subset table.

Defect B revision 2 (user, 2026-09-01): `eps_pi` becomes the single adaptive quantity — a
per-generation one-shot solve on the recorded audit window pinning mean spend at 1.0, clamped to
`[eps_pi_minimum, eps_pi_maximum]` with the clamp as the hard quality floor; beta fixed from the O2
curves; the resignation-mirror threshold calibrator deleted (stateless per-generation threshold
solve remains); the canonical `closed` state and the spend circuit breaker stand.

Adversarial-review round (2026-09-01, post-acceptance): defect A resolved as (b) — O2 stays
fresh-root feasibility (collected with noise ON), warm-frame re-derivation gates activation; defect
B resolved by the single `closed` state, calibrator fallback routing and the spend circuit breaker
(sections 5.1, 6.1, 6.2); C external 4B anchor (3.3, 8); D zeroth checkpoint + warmth feature
(4.2, 5.2); E holdout split by game (4.3); F raw cumulative visits for the segment feature (4.2);
G window segmentation at schedule steps (3.3); H guard framing corrected (1.1, 5.2).

O2-driven course corrections (2026-09-01, all measured on the fetched O2 data): labels switched
to instantaneous exceedance (future-max kept as diagnostic); eps anchored to the live cross-seed
noise floor with spend as an output; beta suggestion 0.10; go/no-go moved to the throughput axis
(spend ≤ 0.80 with quality conditions); cap = 2B retained on the reference-noise argument (floor
0.103 at 2B vs 0.141 at B); guard-only stopping measured and rejected (16–41% exceedance);
**per-position eps (second floor head) evaluated and rejected** — the floor is predictable
(held-out Spearman 0.61, R² 0.53) but at matched spend the per-position frontier is equal-or-worse
on written KL and argmax swaps, and the swap excess sits in high-floor positions that per-position
eps loosens further; support-count and top-3-share joined the feature vector (17 → 19), raw top-k
vectors rejected.

Open, resolved by Stage O2 evidence rather than by fiat:

1. The eps clamp `[eps_pi_minimum, eps_pi_maximum]`, `eps_v` and `beta` (section 3.2/6: chosen
   from the 2B-referenced oracle table, the noise floor and the realized predictor curves; the
   solved eps will live inside the clamp, and under a false-stop ceiling it will sit higher than
   the oracle table's 0.10 unit-spend point).
2. Whether the checkpoint subset table survives re-derivation at the 2B reference (section 2).
3. The go/no-go verdict itself (≥ 1.40x at ≤1% realized false stops, section 10) — user calls it
   after seeing the O2 numbers.
