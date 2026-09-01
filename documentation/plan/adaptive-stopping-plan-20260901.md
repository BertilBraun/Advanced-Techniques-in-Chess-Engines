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
that: a position is stopped only when its own tree's distribution has been **observed** to have
stopped moving (a hard measured guard, section 5.2) *and* a predictor certifies it will stay within
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

The retire-only-on-Elo rule from the v13–v18 findings still applies to the final decision, but the
entry gate is Stage O: a realizable predictor must demonstrably approach the ceiling under the
quality constraint before any production time is spent.

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
U_i = 1  ⟺  max_{j ∈ {i, …, K}} KL(pi(.|s,N) || pi(.|s,c_j)) ≥ eps_pi
            ∨  max_{j ∈ {i, …, K}} |V(s,c_j) − V(s,N)| ≥ eps_v
```

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
- **Why the max over remaining checkpoints, not just `j = i`.** This is DS-MCTS's third observation
  discretized onto our grid: a distribution can sit near `pi_N` at `c_i` and wander away before
  returning (forced playouts and Dirichlet noise make this real at low visit counts). Without the
  future clause the label says "close now", which is not "safe to stop now". The sup over all
  `n' ≥ n` is unobservable; the max over the recorded checkpoints is the computable lower bound, and
  it is exactly the set of states a stopped search could have been stopped in.
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

**Can eps be tuned during a run? No.** `eps_pi` defines the label semantics: every label in the
trailing window, every published predictor, and every calibrated threshold is conditioned on it.
Changing it mid-run silently invalidates the window (labels computed under two definitions mix in one
fit) and re-poses the class-balance the thresholds were calibrated against. The online knob is `beta`
via the self-tuning threshold calibrator (section 6); `eps_pi`/`eps_v` are explicit configuration
keys with **no defaults** (missing key = error, per the repo configuration rule), fixed for the life
of a run, changed only between runs against a new config SHA.

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

Because audits ride live searches, labels come from the production distribution (with tree reuse,
noise, current model) — no train/deploy gap. `sample_fraction` ~0.01–0.02 gives 5,000–15,000 audit
positions per generation at v20 game rates, i.e. 20,000–60,000 labelled checkpoint examples per
generation; a 10-generation trailing window is ample for a 2×64 MLP.

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
| 5 | `KL(pi_{c_i} || pi_{c_{i-1}})` | recent movement: the finite-difference version of the very quantity being predicted (DS-MCTS temporal channels, collapsed to a scalar); for `i = 1` use the prior as the previous distribution |
| 6 | top share of the *latest-segment* distribution `(n_i·pi_{c_i} − n_{i-1}·pi_{c_{i-1}})/(n_i−n_{i-1})` | DS-MCTS channels 4–6, scalarized: is the recent search still choosing the leader |
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
1e-3, batch 4096, ≤30 epochs, fixed-stride holdout — the `fit_curve_corrector` /`export_corrector`
machinery in `src/search_budget/corrector.py` carries over nearly line-for-line (different feature
vector, BCE head, sigmoid clamp instead of correction clamp). Fail-closed rejection rules, extending
the corrector's:

- any non-finite parameter or buffer → reject;
- holdout BCE not better than predicting the window base rate → reject;
- **operational check:** at the thresholds that Stage C calibration would publish, the holdout
  false-stop rate must be ≤ β and the implied visit saving ≥ a configured floor (a predictor that
  saves nothing must not be published just because its BCE improved);
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
    std::shared_ptr<const SearchStopPredictor> predictor;  // null => never stop
    bool apply_learned;                            // false => run to cap? no: run to baseline, flat
};

struct StoppableSearchLimit {                      // replaces PredictedSearchBudgetLimit
    std::uint32_t baseline_visits;                 // B (additional visits, as today)
    SearchStopPolicy policy;
    std::uint64_t model_generation;
    bool shadow_only;                              // audit positions: record verdicts, never stop
};
```

Validation in constructors mirrors `SearchBudgetPolicy`'s (finite, ordered, `apply_learned` with a
null predictor or empty thresholds ⇒ behaves as flat — the fail-closed identity). Note the
fail-closed direction carefully: **`apply_learned = false` means a flat search to `B`**, not an
uncapped search to `2B` — the closed gate must reproduce the baseline configuration exactly, and the
2x cap exists only when the stop rule that pays for it is active.

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
   below `movement_guard_epsilon`, the position cannot stop at `c_i` — no predictor call. This keeps
   the verifiability property partially *unconditional*: a stop always implies the distribution was
   observed to have already settled over the last checkpoint interval; the predictor only certifies
   that it stays settled. A predictor gone wrong can therefore never stop a position whose
   distribution is visibly still moving — the exact v20 failure channel.
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

## 6. Self-tuning threshold calibration and policy publication

**No online spend controller — by design, argued against once and settled.** The 2x cap
reintroduces the possibility of overspend, which is where the instinct for a per-generation dual
comes from. That instinct is rejected: three of the four run-killing defects in this subsystem's
history (inverted projection, stale dual, and the config pin's blast radius) lived in exactly that
control loop, and the oracle table says the chosen eps already centers mean spend at ≈0.96 without
any controller. Spend is measured and reported per generation; if a run drifts materially from unit
spend, `eps` or the cap is changed **between runs** with the change recorded against the config SHA.
The only online quantities are the thresholds, and they are quality-calibrated, not spend-calibrated.

**The asymmetry that sets β.** A true stop at checkpoint `m_i` saves `(2 − m_i)·B` visits — bounded,
worth a fraction of one search. A false stop writes a policy target that is ≥ eps away from what the
search would have produced, on a position the guard-plus-predictor judged settled — i.e. exactly the
prior-shaped corruption that KataGo predicts and v20 exhibited, whose cost compounds through the next
generation's training and is *not* bounded by one search. v20 quantifies the exchange rate: an
aggregate of such corruptions more than cancelled a 1.22x compute equivalent (≥60 Elo). The
threshold policy follows: positive-class ("keep searching") recall is driven toward 100% and the
saving is whatever survives, never the reverse. DS-MCTS uses the same asymmetric tuning for the
weaker reason that one bad move loses a game — our reason is that one bad target outlives the game.

### 6.1 Self-tuning thresholds on the calibrated-resignation pattern (user decision)

There is **no hard kill switch inside the run**. The precedent is the resignation calibrator
(`py/src/self_play/resignation.py`, `kind: calibrated`): when its false-nonloss rate crosses a
ceiling it walks its threshold back promptly, and relaxes it again slowly when the evidence says it
is safe — it never kills resignations outright. The stop-threshold controller mirrors that
implementation's structure, naming and telemetry rather than inventing a new one:

- **Candidate grid** of stop thresholds per checkpoint (`candidate_threshold_minimum/maximum/step`,
  validated as an integral grid exactly as `CalibratedResignationConfiguration.validate_candidate_grid`).
- **Evidence:** every audit position scores *all* candidates at once (which candidates would have
  stopped it, and whether its label says `U_i = 1`) — the analog of `CandidateAuditEvidence` /
  `TriggeredContinuationGame`, evaluated on the simulated survivor population per checkpoint
  (cheapest first, since the population reaching checkpoint `i` is conditioned on not stopping
  earlier — the one place the stop calibrator differs structurally from resignation's flat grid).
- **Selection per generation**, mirroring `ResignationCalibrator._recalibrate`
  (`resignation.py:277-334`): a candidate is *safe* when its trigger count ≥
  `minimum_evidence_trigger_count` and its **one-sided binomial upper bound** on the false-stop rate
  (`one_sided_binomial_upper_bound`, `resignation.py:116`, at `confidence_level`) is ≤
  `false_stop_rate_ceiling`. Tightening to the safe target is immediate; relaxation is bounded by
  `maximum_relaxation_per_generation`, grid-snapped, at most once per generation. No safe candidate
  ⇒ the checkpoint's threshold falls to the never-stop end of the grid — stopping quietly attenuates
  instead of the run gating off.
- **Configuration keys** (explicit, no implicit defaults): `false_stop_rate_ceiling` (initial
  suggestion 0.01), `candidate_threshold_step`, `minimum_evidence_trigger_count`,
  `confidence_level`, `maximum_relaxation_per_generation`, `first_production_generation` (the warmup:
  thresholds publish as never-stop until enough audit generations exist — suggest 10, much less than
  the old 30 since labels are no longer starved by an 8x deep-search budget). Telemetry mirrors
  `ResignationDiagnostics`: selected threshold + safety flag, trigger counts, false-stop count/rate/
  upper bound, average stop checkpoint, saved-visit totals — per checkpoint.

What the self-tuner can and cannot do, stated so nobody mistakes one for the other: it controls the
false-stop **rate** — the target-corruption channel. It cannot tell us the mechanism is **worth**
anything: 0% false stops at 1.02x effective compute passes every ceiling and is still a failure.
Worth is decided *offline*, once, by the Stage-O2 go/no-go bar (≥1.40x, section 10) before any
production run is spent; inside the run the only worth-tracking is telemetry (section 8) feeding the
between-runs decision.

### 6.2 Publication

Per finalized audit generation the manager publishes `{checkpoint_multiples, thresholds per
checkpoint, movement_guard_epsilon, predictor path+sha256, apply_learned}` for the next *unstarted*
generation through the existing publication state machine
(`src/search_budget/calibration.py::publish_fail_closed`, `publication_for_generation`,
`load_calibration_state_fail_closed` — the one calibration structure that survives deletion,
reworked payload). Fail-closed applies only to **structural** failures — unreadable or
sha-mismatched state (the config-pin defect class), a predictor that fails its load probe, an audit
pipeline lagging beyond `maximum_unstarted_generation_lag` — and publishes never-stop thresholds
(bit-identical to flat, tested). Rate excursions are *not* structural failures; they are handled by
the calibrator's walk-back above.

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
  stop-eligible audits, and fraction of stopped targets whose `KL(pi_stop || raw prior)` is below the
  window median (drift-toward-prior watch: if the stopped population's prior-deviation collapses,
  targets are degrading even if eps holds).
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
- **O2, ~1 GPU-hour, needs user authorization:** a short collection job on the current node — the
  existing deep-label machinery pointed at recent v20 replay with `deep_search_multiple: 2` (search
  to the 2B cap), checkpoints at the candidate multiples, `retention: retain_all`, plus the
  paired-seed noise-floor runs (3.2), ~5,000 positions each. Output fetched via
  `run_control.sh preserve/fetch`. This re-derives the oracle table at the true `pi_2B` reference
  (the 8x-referenced table overstates KLs, so the 2B-frame eligible mass will be somewhat larger),
  validates eps 0.10 against the noise floor, fixes the checkpoint subset and guard epsilon, and
  measures predictor recall/false-stop trade-offs on exact labels.

**Go/no-go gate on the offline study — the only place a "worth it" bar exists (user decision:
no kill switch inside a run; runtime quality control is the self-tuning calibrator of section 6.1,
which attenuates, never kills).** The oracle is 1.64x effective compute at unit spend, and 1.22x
live *already lost* — the realizable rule must land well inside that corridor to be worth a
production run at all. On held-out O2 generations, the trained predictor behind the observational
guard, at thresholds giving realized false-stop rate ≤ 1%, must achieve **≥ 1.40x effective compute
at mean spend in [0.9, 1.1]** (i.e. capture ≥ ~45% of the oracle's gain over flat, vs the retired
system's 26.5% of its oracle). Below that, the mechanism cannot clear the bar the retired system
failed even if targets are perfectly clean — Stage P is not spent, the study is written up, and the
adaptive-search line is retired (final run on v13/v17-era configurations). This bar is evaluated
exactly once, offline; it does not exist inside the run. Everything before Stage P costs at most one
authorized GPU-hour plus local compute.

**Stage U — Python unit tests** (`py/test/test_search_stopping_*.py`, importlib mode): label
extraction from checkpoint records including the future-max clause; eps boundary semantics; the
threshold calibrator mirrored on the resignation-calibrator tests (candidate-grid evidence
accounting on the sequential survivor population, binomial upper bound, immediate tighten vs
step-bounded relax, no-safe-candidate fallback to never-stop); predictor-fit rejection rules;
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
and the 64-search ladder. Thresholds publish as never-stop for the first ~10 audit generations by
construction (`first_production_generation`). If the stopping run trails, the `exclude` fallback is
implemented as a code change (section 7) and rerun once. If both trail, the idea is retired on Elo
evidence, the findings memory updated, and the final run uses v13/v17-era configurations.

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
   shrinks for the same reason). The eps recommendation is provisional until O2 re-derives the
   table; O2 exists precisely so no production decision rests on the 8x frame.
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

Open, resolved by Stage O2 evidence rather than by fiat:

1. `eps_pi` / `eps_v` (section 3.2: chosen from the 2B-referenced oracle table, the noise floor, and
   the realized predictor curves; 0.10 is provisional and the launch value will likely be higher).
2. Whether the checkpoint subset table survives re-derivation at the 2B reference (section 2).
3. The go/no-go verdict itself (≥ 1.40x at ≤1% realized false stops, section 10) — user calls it
   after seeing the O2 numbers.
