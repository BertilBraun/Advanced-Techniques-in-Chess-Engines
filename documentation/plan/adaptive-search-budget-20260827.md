# Adaptive search budget from a learned difficulty head — design of 2026-08-27

Replaces the threshold-based `AdaptiveFullSearchBudgetConfiguration` and the fast/full search split with a single
per-position budget predicted by an auxiliary network head.

Evidence: [`analysis/chess-search-findings-20260827.md`](../analysis/chess-search-findings-20260827.md) and
[`benchmarks/chess-search-evaluation-rtx3060-20260826/`](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md).
Measured on generation 162 of `vast-chess-4day-production-v9` at a 600-visit baseline; the shapes should carry to
larger baselines but the exact percentages are from that operating point.

Nothing here is started without an explicit instruction.

**Status — stopped at Gate 1.** The frozen-trunk predictor scored -13.95% of oracle gain at equal mean compute,
with a wholly negative 95% bootstrap interval. The allocator and training integration must not proceed from this
design without a new measured gate that beats flat. See the
[`RTX 4070 SUPER probe`](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md).

## 1. Why

Allocating search per position is worth up to **3.6× effective compute** at fixed total spend (noise-corrected
oracle at a mean of 600 visits). The existing adaptive rule captures none of it — its stopping signal is
inverted-U in the quantity it needs to predict, so a monotone threshold cannot work at any parameter setting, and
at the production thresholds it selects *worse than random*. The mechanism is sound; the criteria are not.

A learned head can express the shape a threshold cannot. The ceiling is bounded and known (§6), which is why this
is a bounded bet rather than an open-ended one.

## 2. The label

For a position sampled from the replay buffer, the offline job runs two searches and compares their policy
targets.

```
label_raw = KL( policy_target @ deep  ||  policy_target @ baseline )
label     = quantile rank of label_raw among the labelled batch      # in [0, 1]
```

**Metric: KL, not total variation.** Measured across five truth depths, KL ranks positions about ten points
better than TV at every depth — worth roughly a doubling of search depth, for free.

| Truth depth | ×baseline | TV label | KL label |
|---|---|---|---|
| 2,400 | 4.0× | 31.5% | 41.4% |
| 3,200 | 5.3× | 37.6% | 48.7% |
| **5,000** | **8.3×** | 40.4% | **52.2%** |
| 8,000 | 13.3× | 46.0% | 55.7% |
| 10,000 | 16.7× | 46.4% | 56.7% |

(Share of the ideal allocation gain a perfectly predicted label would capture.)

**Depth: about 8× baseline.** Returns flatten after that — 8.3× captures 92% of what 16.7× does, while 4×
captures only 73%. Beyond 13× there is nothing left to buy.

**Sample fraction: about 1%.** Cost scales linearly with depth, so prefer deeper labels on fewer positions: 1% at
8× costs about the same as 2% at 4× (~7% of self-play search compute) and captures 52% instead of 41%. For a
single scalar head, label quality matters more than label count.

**Normalisation: quantile rank, not raw KL.** The raw label spans four orders of magnitude — median 0.053, 90th
percentile 0.47, 99th 2.46, maximum 11.2 — so a plain regression would be dominated by a handful of extreme
positions and ignore the ordering in the bulk, which is where nearly all the decisions are. The quantile rank is
uniform on [0, 1], has mean 0.5 by construction without the loss enforcing it, and loses nothing measurable
because every capture number above is rank-based. Maintain the quantile map as a running estimate over recent
labelled batches so it tracks the model as it improves.

## 3. The head

- One scalar auxiliary output alongside policy and value, read at the same root forward pass that already
  produces the prior. With tree reuse the root was expanded during the previous move, so the prediction is
  available before the search starts.
- Trained **only** on deep-labelled positions; masked everywhere else. `IneligibleNextPolicyTarget` is the
  existing precedent for a masked auxiliary target.
- One head, not two. A continuous budget subsumes the fast/full decision — "fast search" is simply the bottom of
  the range — and two heads would need reconciling when they disagree.

## 4. The allocator

```
budget(s) = B · m(s)
```

where `s` is the predicted quantile, `B` the generation's scheduled budget, and `m` a fixed monotone curve from
quantile to multiplier, shaped like the oracle's own allocation and normalised so the mean lands on `B`.

- **Floor 0.2×, ceiling at least 4×.** The ceiling carries the value: capping at 2× costs about 38% of the
  available gain, 4× keeps 85%, 8× keeps 97%. The floor barely matters — 0.17× against 0.33× is worth 2.4 points.
- **`m` is not linear.** The noise-corrected oracle sends **73% of positions below baseline** and 22% above, so
  the median prediction must map well below 1×, not to 1×. A linear map from a mean-0.5 output would pile budget
  into the middle, where the oracle spends almost nothing.
- **Mean preservation is a property of `m`, not of the loss.** Total compute per generation is unchanged by
  construction; the head only redistributes it.
- **Blend in gradually.** A generation-scheduled weight from 0 toward 1, mirroring `learned_gate_start_generation`.
  This is not optional politeness: randomly dispersing a good budget distribution scores **−74%** against flat,
  far worse than doing nothing, because a non-uniform allocation starts in a convexity hole and must order
  positions well simply to break even. **A weak head is actively harmful.**

## 5. Parallelism follows the budget

Scale `parallel_searches` with the position's budget so sequential depth stays roughly constant — about 150
rounds: 100 visits → 1, 300 → 2, 600 → 4, 1,600 → 8, 2,400 → 16. This keeps the inference batch full when only a
handful of positions in a step are searching deeply, which is otherwise the same tail starvation measured for the
fast/full split.

**Untested assumption.** Parallelism was measured at *fixed* 600 visits, so "more parallel" was confounded with
"fewer rounds"; whether holding rounds constant makes parallelism free was never checked. If it is false, deep
searches quietly degrade exactly where most of the budget goes. Worth one measurement before relying on it.

`SelfPlaySearchParameters` currently has a single `parallel_searches` for all searches, so this needs a contained
native change.

## 6. What this cannot exceed

- A perfect label at 16.7× depth captures **56.7%** of ideal allocation; at the recommended 8× it is **52.2%**.
  The head's own prediction error then comes off the top of that. The realistic target is a fraction of 52%, not
  of 100%.
- Everything is measured as policy-target fidelity, not training outcome. Better targets are assumed to train
  better; that assumption is not tested here.
- Measured at generation 162 with a 600-visit baseline and `parallel_searches` 1, from fresh roots. Production
  runs a larger baseline, parallelism 4, and 60% root retention.

## 7. The offline labelling job

- **Cadence: per generation**, triggered after each training run, on positions from the newest generation. Labels
  must reflect the current model's search behaviour, and the replay buffer is already generation-indexed.
- **Separate from the evaluation manager.** That runs on wall-clock cadence and can be disabled independently;
  neither is right for this. Same access pattern, different lifecycle.
- **Runs in the background** and writes results back when finished, so it never stalls training — which is the
  bottleneck early on.
- **Deep searches are not overhead.** They produce the best policy targets in the system, so write them into the
  replay buffer as training samples in their own right. The label falls out of compute that was already worth
  spending. Because the sample is drawn at random, the label distribution stays covered whatever the head does
  with everything else — no bootstrapping, no feedback loop.
- **Duplicates are fine.** The same position already recurs across games with different noise and targets;
  `duplicate_multiplicity_weight_cap` is `null` in v9 and unused.

The write-back into materialization is the messiest part of the implementation and where cost will concentrate.

## 8. What is removed

- `AdaptiveFullSearchBudgetConfiguration` and its eight parameters: `minimum_visits`, `observation_interval`,
  `leader_stability_window`, `root_value_tolerance`, both threshold schedules, `threshold_relaxation_visits`, and
  the learned gate.
- `SearchCorrectionGate` and `minimum_search_correction_to_unlock_tail`. Note the `search_correction` head was
  **never enabled** in any run — v9 configures only `next_policy` and `remaining_game_length` — so the gate has
  always been reading an untrained output.
- The fast/full split. Every position becomes a training sample; the floor budget replaces the fast search.
- **No downweighting by budget.** Under head-driven allocation a position gets the floor *because* its target has
  settled, so its label is good. Downweighting would discard signal and adds a parameter for no measured benefit.

Keep the checkpoint-trace machinery (`SearchCheckpointDetail::Policies`): it is what makes label generation and
the fidelity analyses cheap.

## 9. Implementation surface

**Native**
- A `SearchLimit` variant that reads the predicted budget at root expansion and sets the visit limit, alongside
  `FixedSearchLimit` and the retired adaptive limit. The root node already carries a network scalar
  (`search_correction`), so there is precedent for the plumbing.
- Per-search `parallel_searches` derived from the assigned budget (§5).

**Python**
- A `search_budget` auxiliary target kind through `targets.py`, `materialization.py`, `columnar.py` and `store.py`,
  with eligibility masking — following the `search_correction` scalar rails, which already run end to end.
- The offline labelling tool and its write-back.
- Configuration: depth multiple, sample fraction, floor and ceiling, the quantile-to-multiplier curve, and the
  blend schedule.
- Materialization stops filtering on `full_search`.

## 10. Gates

1. **Offline probe.** Fit a predictor of the label on the frozen checkpoint's own trunk features. Score by the
   share of oracle gain captured at equal mean compute, not by regression loss. **Stop if it captures almost
   nothing** — the signal is then not in the trunk and a trained head is unlikely to find it. Cost: a few
   GPU-hours, no training changes.
2. **Shadow mode.** Run the head in a live run without acting on it. Check that predicted budgets would preserve
   the mean and that calibration holds as the model improves.
3. **A/B.** A run where the head drives budgets matches or beats the per-generation yardstick at equal
   wall-clock, with mean target top-1 agreement improved at the same mean visit spend.

Gate 1 is cheap and decisive; nothing downstream should start before it clears.
