# Chess search follow-up plan — 2026-08-27

Continuation from [`analysis/chess-search-findings-20260827.md`](../analysis/chess-search-findings-20260827.md).
Nothing here is started without an explicit instruction. Work is grouped by what it depends on, and each item
states the decision it unblocks.

## WP-S1 — Make the evaluation search honest (small, do first)

The measurement apparatus is biased and everything downstream inherits it.

- `create_evaluation_search` inherits self-play `first_play_urgency` instead of forcing zero, and self-play
  `exploration_constant` instead of the configured 1.0.
- The override plumbing already exists on branch `search-evaluations`
  (`EvaluationTreeSearchOverrides`, `resolved_evaluation_parameters`), threaded as a call argument so
  `experiment_configuration_sha256` is unchanged. What is missing is making inheritance the *default* rather than
  an opt-in override.
- Decide what happens to recorded history: past ladder Elo is systematically pessimistic by roughly 76 Elo and is
  not comparable to anything measured after this change. Either re-measure the frozen four-day baseline under the
  corrected search, or annotate the yardstick with a discontinuity marker.

**Unblocks:** every future ladder number, and comparability with the recovery-plan yardstick.
**Acceptance:** a frozen checkpoint scores measurably higher under the corrected evaluation search, and the
per-generation yardstick carries an explicit note about which side of the change it was measured on.

## WP-S2 — Difficulty head, phase 1: is it learnable at all?

The oracle says perfect per-position budgeting of the full searches is worth up to 3.6× effective compute
(noise-corrected). That
is not reachable, but the gap to the hand-made signals (ρ ≈ 0.21, and in the wrong shape) is wide enough that this
deserves a real attempt.

Do the cheap offline experiment before touching the trainer.

1. **Generate labels.** `tools.measure_policy_target_fidelity --per-position-output` already emits, per position,
   the divergence at every fixed budget. The label is `benefit = KL(fast budget) − KL(full budget)`. One pass is
   3,000 positions in ~45 minutes; scale to 30,000–100,000 positions over a few hours.
2. **Fit an offline predictor** on the existing network's own features — start from the trunk activations of the
   frozen checkpoint, predicting `benefit` as a scalar. This is a probe, not a head: no training changes yet.
3. **Score it the way it will be used**, not by regression loss. Replay the predicted budgets through the
   per-position fidelity curves the labels came from and report where the predictor's frontier sits between the
   flat frontier and the oracle frontier at equal mean compute. "Captures X% of the available oracle gain at 600
   mean visits" is the number that decides this; R² is not.

**Decision gate.** If a probe on frozen features captures a useful share of the oracle gain, proceed to WP-S3. If
it captures nearly nothing, the difficulty signal is not present in the trunk and a trained head is unlikely to
find it either — stop, and keep the flat schedule.

**Cost:** a few GPU-hours plus offline fitting. No training run. No production risk.

## WP-S2b — superseded

The label and allocator design is now settled and lives in its own document:
[`adaptive-search-budget-20260827.md`](adaptive-search-budget-20260827.md). It supersedes everything below in
this section, in particular the convergence-residual label proposed here, which measurement rejected in favour of
KL against a deep search.

## WP-S2b (superseded) — What the head predicts, and how the label is built

### The head already exists and predicts the wrong quantity

`search_correction_target` is computed in `cpp/src/search/SearchExecutor.hpp:161` as

```
value_correction  = 0.5 * |root_value_after_search - network_root_value|
policy_correction = 0.5 * sum |searched_probability - raw_prior|
search_correction_target = max(policy_correction, value_correction)
```

It flows into training as a scalar auxiliary target (`replay/materialization.py:174`) and the prediction is read
back at the root by `SearchCorrectionGate`. So the plumbing for a per-position head — target, materialization,
training, and a native read at root expansion — is already built and does not need inventing.

What it measures is **how wrong the network was**, which is a *learning-value* signal, not a *marginal-return*
one. The two come apart exactly where budget allocation lives:

- A tactic the network missed: `policy_correction` is high, but search resolves it in a few dozen visits, so the
  marginal return of more visits is low. The existing target says "spend more" when the right answer is "you
  already have it".
- Two near-equal moves the network ranks roughly right: `policy_correction` is low, but the target keeps moving
  for hundreds of visits. The existing target says "stop" when the right answer is "keep going".

This is the inverted-U of findings §2 in another form, and it is the most likely reason the learned gate never
earned its keep.

### What the label should be — measured, not argued

Three candidate quantities were compared on the 3,000-position dataset by handing out the *oracle's own multiset
of budgets* in the order each signal implies, so the budget distribution is identical and only the ordering
differs. At a mean of 600 visits, flat is KL 0.2971 and the oracle is 0.1249.

| Ordering signal | KL | Share of the oracle gain captured |
|---|---|---|
| True Lagrangian optimum (the oracle itself) | 0.1249 | 100% |
| **True remaining error, KL(target@600 ‖ truth)** | **0.2036** | **54.3%** |
| True benefit, KL(200) − KL(600) | 0.2751 | 12.8% |
| True benefit, KL(300) − KL(600) | 0.2981 | −0.6% |
| Observable `top_visit_share` at 200 visits | 0.4881 | −110.9% |
| Observable `top_two_margin` at 200 visits | 0.4953 | −115.1% |
| Random ordering (control) | 0.4249 | −74.2% |

Three conclusions, and the first two overturn the earlier proposal in this document.

**Predict remaining error, not movement.** An earlier draft proposed the convergence residual
`TV(target@B/2, target@B)` — how much the target moved over the second half of the search. Ranking by the closest
measurable analogue of that captures essentially nothing (−0.6% at 300→600, 12.8% at 200→600), while ranking by
how far the target still sits from truth captures 54.3%. Movement is a poor proxy for distance: a position can
move a great deal and remain far from converged, or barely move because the search is stuck.

**A weak head is actively harmful.** Randomly dispersing the oracle's budget multiset scores −74.2%, far worse
than flat. This is the Jensen convexity of findings §2 again: any non-uniform allocation starts in a deep hole and
must order positions well simply to break even. The two observable within-search statistics score *below* random,
so no hand-made rule reading them can help — which is the strongest argument for a learned head, and equally the
strongest argument for blending it in cautiously and gating on beating flat.

**Even a perfect predictor of remaining error reaches only 54%** of the oracle, because remaining error alone does
not determine the optimal budget — the shape of each position's marginal-return curve matters too. Treat 54% as
the realistic ceiling for a single-scalar head, not 100%.

### Getting a trainable label for remaining error

Remaining error is not observable in production: it needs a reference deeper than the search being labelled.
Obtain it by **sampling**, not by proxy.

Run a small fraction of full searches — 1–2% — at a multiple of the base budget, and label those positions with
`KL(target@B ‖ target@deep)`. At 4× depth on 2% of full searches this costs about 6% of full-search compute and
under 2% of total self-play compute, and it yields the quantity that measurably ranks best. Everything else rides
the scalar rails that `search_correction_target` already uses.

This also solves the feedback problem: the deep-labelled subsample is drawn at random, so the label distribution
stays covered regardless of what the head does with the rest.

### Calibration and the output parameterisation

Do **not** bucket into a narrow menu. Measured at a mean of 600 visits:

| Budget menu | KL | Share of oracle gain |
|---|---|---|
| Flat 600 | 0.2971 | 0% |
| 3 levels, 0.5× / 1× / 2× (300 / 600 / 1200) | 0.2062 | 52.8% |
| 3 levels, wide (100 / 600 / 2400) | 0.1730 | 72.0% |
| 3 levels, wider (100 / 600 / 3200) | 0.1699 | 73.9% |
| Full 14-level menu | 0.1249 | 100% |

A narrow 0.5×–2× menu throws away nearly half the available gain; the dynamic range matters far more than the
number of levels. The oracle's own allocation is heavily right-skewed — 36% of positions sit at the 100-visit
minimum, with a tail to 10,000 — so the useful multiplier range is roughly **0.17× to 16×** of the base budget,
not 0.5× to 2×.

That shape also rules out mapping a mean-0.5 output linearly onto [minimum, maximum]: it would pile budget into
the middle, where the oracle spends almost none. The parameterisation that satisfies both constraints is:

- **Head output** `s ∈ [0, 1]`: the position's predicted **quantile of remaining error** within the current
  population. Uniform by construction, so its mean is 0.5 without the loss having to enforce it, and the heavy
  skew of the underlying quantity cannot swamp the tail the way a raw-magnitude regression would.
- **Allocator** `b = Q(s)`: a fixed budget-quantile function shaped like the oracle's distribution and normalised
  so the mean lands on the generation's scheduled budget. Continuous, not bucketed, and it reproduces the right
  skew.

Collapse to a constant 0.5 is not a failure mode to engineer around — it is precisely the null result the WP-S2
probe exists to detect. If the features carry no signal, the head should say so.

## WP-S3 — Difficulty head, phase 2: wire it in

Only if WP-S2 clears its gate. Two uses, in increasing order of risk:

1. **Per-position budget scaling on the full searches (do this first).** Keep the random selection of which
   positions get a full search, keep the generation-level visit schedule, and scale it per position by a factor
   driven by the head. This is where the 3.6× bound lives, and because the set of training positions is unchanged
   it carries no distributional hazard. Hold the *mean* budget fixed so the throughput budget is undisturbed.
2. **Fast/full selection (only after 1 works).** Replacing the random `full_search_probability` draw with a
   top-fraction-by-predicted-benefit selection looks free because compute is unchanged, but findings §1.3 shows
   contested positions carry the *least* reliable targets at a fixed budget — the contested quartile needs about
   1,600 visits to match a random quartile's accuracy at 600. Selection must therefore come with a budget
   increase for the positions it selects, which is why it depends on (1) rather than preceding it.

Design notes that follow from the measurements:

- **Blend the head in gradually.** Early generations have no useful difficulty signal; the head must earn
  authority. A generation-scheduled blend weight from 0 toward 1 is the natural mechanism, mirroring
  `learned_gate_start_generation`.
- **Predict benefit, not convergence.** The inverted-U result is the whole point: "the search looks decided" is
  the wrong target and is what the current adaptive rule gets wrong. Positions where two or three moves are
  genuinely contested are the ones worth searching.
- **The two uses do not simply reinforce.** A position predicted difficult is worth a longer search, but at a
  *fixed* budget it also yields a less reliable target (findings §1.3). One head can drive both decisions, but
  selecting a position for a full search and lengthening that search have to move together.
- **Guard against feedback.** The head is trained on labels produced by searches whose budgets the head itself
  chose. Positions it starves are never labelled at high budget again. Retain a random control fraction of
  full searches so the label distribution stays covered.

**Acceptance:** a training run where the head scales full-search budgets matches or beats the per-generation
yardstick at equal wall-clock, with mean target top-1 agreement improved at the same mean visit spend.

## WP-S4 — Retire the current adaptive budget

Independent of WP-S2/S3 and safe to do now.

- Remove `AdaptiveFullSearchBudgetConfiguration` from the recommended configuration set, or leave it in the schema
  but mark it superseded. It is a wash at the production cap, degrades as the cap rises, and its thresholds sit in
  the band where they select worse than random.
- Eight parameters retire with it: `minimum_visits`, `observation_interval`, `leader_stability_window`,
  `root_value_tolerance`, both threshold schedules, `threshold_relaxation_visits`, and the learned gate.
- Keep the checkpoint-trace machinery. `SearchCheckpointDetail::Policies` and the checkpoint struct are what make
  the fidelity study and the label generation in WP-S2 cheap; they are worth keeping regardless.

**If the budget is kept instead of retired,** the only defensible setting is a top-visit-share threshold of ≥0.85,
which stops the 13% of positions that genuinely do not need more search. That is worth under 9% of compute, of
which convexity takes back about 40% — a small honest gain, not worth eight parameters.

## WP-S5 — Self-play throughput and data freshness

Decide `parallel_games_per_process` against `parallel_searches` with the numbers in findings §3.3.

- **The status quo is defensible.** 512 × 4 is the throughput maximum among the configurations measured, and
  `parallel_searches: 4` is genuinely earning +20% by keeping the fast/full tail's batch full.
- **If fresher replay data is wanted**, 320 × 4 is the low-risk step: 1.47× fresher for 8% throughput at the same
  parallelism quality cost. 160 × 4 gives 2.6× fresher for 18%. Prefer 320 × 4 over 320 × 2 — it wins on both
  throughput and latency at equal quality cost.
- **Do not lower `virtual_loss_weight`.** It makes no measurable difference where it binds and 1.0 is the
  conservative choice.

### WP-S5a — The fast-search staggering may be the better lever

`initialFastSearchAdmissionCount` already computes a ratio-based admission count designed to make fast searches
finish alongside full searches, which would keep the batch full without needing parallelism at all. At generation
162 that value is 96, but it is overridden by the capacity-based value of 384, so all games start together and the
tail starves.

Worth testing whether honouring the ratio-based count fills the tail's batch without paying the parallelism
quality cost. **This is a hypothesis from reading the code and the batch-occupancy numbers, not a measured
result.** If it holds, it recovers the +20% while letting `parallel_searches` drop toward 1, the best quality
setting.

Raising `inference_workers` is **not** the lever: the measured A/B in
[`benchmarks/self-play-submission-8xrtx4070super-20260824`](../benchmarks/self-play-submission-8xrtx4070super-20260824/README.md)
took it from 2 to 1 and the average batch rose from 141 to 222, because two workers means two separate half-empty
batches plus doubled CUDA contexts, not one larger batch.

A second, stronger variant follows from fast searches producing no training samples: **give fast and full searches
different parallelism.** Fast searches have no target-quality requirement at all, so they can absorb high
parallelism to fill the batch, while full searches — which carry the entire training signal — run at parallel 1.
With staggering, roughly 128 full searches at parallel 1 plus about 96 concurrent fast searches at parallel 2
would fill a 320 batch exactly. `SelfPlaySearchParameters` currently has one `parallel_searches` for both, so this
needs a contained native change.

**Cost:** one afternoon with `tools.benchmark_self_play_search`, which already parameterises all of this.

## WP-S6 — Close the open measurement gaps

Ordered by value.

1. **Parallelism cost in target fidelity, not Elo** (~20 min). Compute 600-visit targets under parallel 1 and
   parallel 8 against a common parallel-1 10,000-visit reference and read the cost directly off the fidelity
   frontier. Every parallelism trade above currently routes through an Elo-to-visits translation that assumes the
   two mechanisms degrade quality alike. This replaces the assumption with a measurement.
2. **Raise the visit schedule and confirm it end to end.** Both instruments say 600 is short. The strength gain is
   large; the cost is linear wall-clock. A short run at a raised schedule against the yardstick settles it.
3. **Power up the parallelism contrast** (~2 h) only if the games/parallelism decision turns on whether the cost
   is 20 or 45 Elo. 800 games per arm resolves it.
4. **Evaluation visit budget.** Whether checkpoint *ranking* is preserved at low visits is untested — this study
   varied visits for one checkpoint, not the ordering across checkpoints. If ranking is preserved, cheap
   evaluation during training plus one expensive final evaluation is sound; that needs its own experiment, and a
   10,000-visit final evaluation needs a much harder rung than 3,500 nodes.

## Sequencing

WP-S1 and WP-S4 are small and independent — do them first. WP-S5 is a configuration decision that needs no new
work beyond §3.3, optionally informed by WP-S5a. WP-S2 is the gate for the only large opportunity in the study,
and WP-S3 follows only if it clears. WP-S6.1 is cheap and should be folded into whichever of these runs next.
