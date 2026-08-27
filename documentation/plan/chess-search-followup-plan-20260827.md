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

The oracle says perfect per-position budgeting is worth 4.4× effective compute, and perfect fast/full assignment
alone is worth 2.2× at zero extra cost. Neither is reachable, but the gap to the hand-made signals (ρ ≈ 0.21, and
in the wrong shape) is wide enough that this deserves a real attempt.

Do the cheap offline experiment before touching the trainer.

1. **Generate labels.** `tools.measure_policy_target_fidelity --per-position-output` already emits, per position,
   the divergence at every fixed budget. The label is `benefit = KL(fast budget) − KL(full budget)`. One pass is
   3,000 positions in ~45 minutes; scale to 30,000–100,000 positions over a few hours.
2. **Fit an offline predictor** on the existing network's own features — start from the trunk activations of the
   frozen checkpoint, predicting `benefit` as a scalar. This is a probe, not a head: no training changes yet.
3. **Score it the way it will be used**, not by regression loss:
   - Ranking quality — what fraction of the oracle's fast/full gain does a top-25%-by-prediction assignment
     capture? The oracle moves 261 → 580 equivalent visits; capturing even a third of that is large.
   - Because the top 10% of positions carry 76.7% of the benefit, precision at the head of the ranking is what
     matters. Report precision@10% and precision@25%, not R².

**Decision gate.** If a probe on frozen features captures a useful share of the oracle gain, proceed to WP-S3. If
it captures nearly nothing, the difficulty signal is not present in the trunk and a trained head is unlikely to
find it either — stop, and keep the flat schedule.

**Cost:** a few GPU-hours plus offline fitting. No training run. No production risk.

## WP-S3 — Difficulty head, phase 2: wire it in

Only if WP-S2 clears its gate. Two uses, in increasing order of risk:

1. **Fast/full assignment (do this first).** Replace the random `full_search_probability` draw with a
   top-fraction-by-predicted-benefit selection at the *same* full-search rate. Compute is unchanged by
   construction, so this is the cheapest possible test of the head in a live run, and it targets the largest
   measured gain.
2. **Per-position budget scaling.** Keep the generation-level visit schedule and scale it per position by a
   factor in [0, 1] driven by the head. This is the general form and subsumes (1), but it changes total compute
   per generation and so perturbs the throughput budget; it should follow (1), not precede it.

Design notes that follow from the measurements:

- **Blend the head in gradually.** Early generations have no useful difficulty signal; the head must earn
  authority. A generation-scheduled blend weight from 0 toward 1 is the natural mechanism, mirroring
  `learned_gate_start_generation`.
- **Predict benefit, not convergence.** The inverted-U result is the whole point: "the search looks decided" is
  the wrong target and is what the current adaptive rule gets wrong. Positions where two or three moves are
  genuinely contested are the ones worth searching.
- **The two uses reinforce.** A position predicted difficult is both worth a longer search *and* worth having as
  a full-search training target, which is the argument for driving the fast/full decision from the same head.
- **Guard against feedback.** The head is trained on labels produced by searches whose budgets the head itself
  chose. Positions it starves are never labelled at high budget again. Retain a random control fraction of
  full searches so the label distribution stays covered.

**Acceptance:** a training run where the head drives fast/full assignment matches or beats the per-generation
yardstick at equal wall-clock, with target top-1 agreement improved at the same visit spend.

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

Worth testing whether honouring the ratio-based count — or restoring `inference_workers: 2` to raise capacity —
fills the tail's batch without paying the parallelism quality cost. **This is a hypothesis from reading the code
and the batch-occupancy numbers, not a measured result.** If it holds, it recovers the +20% while letting
`parallel_searches` drop toward 1, which is the best quality setting.

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
