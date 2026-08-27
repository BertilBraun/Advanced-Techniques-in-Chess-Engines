# Learned adaptive search budget with continuous shadow calibration

**Status:** implementation plan, revised 2026-08-28

## Decision

Replace both the threshold-based adaptive full-search mechanism and the fast/full search split with one learned per-position search budget.

The network predicts a scalar search-difficulty quantile from the root forward pass. A fixed nonlinear allocation curve maps that quantile to a mean-preserving visit multiplier. The allocator is introduced gradually through a blend coefficient, but that coefficient is not advanced on a fixed calendar and is not chosen from training loss or correlation alone. It is calibrated continuously from the counterfactual search checkpoints produced by the deep-label job.

The first live training run starts with the learned allocator completely inert. Until approximately 30 source-generation label jobs have completed:

- the head trains;
- random positions receive deep labels;
- the allocator is evaluated in shadow mode;
- every production search still receives the flat baseline budget;
- the published allocator blend remains zero.

After the warm-up, a small deterministic controller may increase the blend only when fresh random shadow evidence says that a candidate blend improves policy-target fidelity at equal mean search compute. It reduces the blend immediately when the evidence deteriorates and falls back to zero when no candidate has positive current and trailing utility.

The earlier offline probe is no longer a production gate. It was useful as a small-data baseline, but only a representative live run can answer whether the jointly trained trunk and auxiliary head learn the signal at the scale and distribution of actual self-play.

## Why the existing mechanism must be removed

The current adaptive rule has the wrong functional form. The marginal benefit of searching longer is inverted-U-shaped in the top-visit share read by that rule:

- diffuse positions gain about 0.16 times the population mean;
- genuinely contested positions gain about 1.71 times the population mean;
- already-decided positions gain about 0.30 times the population mean.

The production thresholds, decaying from 0.7 to 0.5, sit in a region where selection is worse than random. No monotone threshold on that signal can solve the problem, and the seven-parameter tuning grid consequently changed nothing.

The learned `search_correction` gate is not a rescue path. The v9 run trains only `next_policy` and `remaining_game_length`; `search_correction` was never enabled, so `SearchCorrectionGate` has always consumed an untrained output.

The fast/full split also wastes training value. Replay materialization currently drops every observation whose `full_search` flag is false, although fast searches consume roughly 36-43% of self-play search compute. In the replacement design every played position becomes a training sample. The allocator floor takes the role of a cheap search without making that sample ineligible.

## What the measurements establish

Perfect per-position allocation is worth approximately 3.6 times effective compute at fixed total visit spend. A practical learned allocator cannot capture all of that value: even perfect labels at finite search depth capture only part of the oracle ranking.

KL divergence between the deep and baseline policies is materially more informative than total variation. Approximate label-depth results are:

| Deep-search visits | Share of oracle allocation gain captured by a perfect label |
| ---: | ---: |
| 2,400 | 41.4% |
| 3,200 | 48.7% |
| 5,000 | 52.2% |
| 8,000 | 55.7% |
| 10,000 | 56.7% |

The deep-label limit is exactly eight times the baseline configured for its source generation. It captures about 92% of what a sixteen-times label provides, whereas four times captures about 73%. Going deeper on fewer randomly selected positions is more useful than going shallower on more positions because label cost scales approximately linearly with visits.

Non-uniform allocation has a serious convexity penalty before ordering quality is considered. Randomly dispersing the measured good budget distribution scored approximately 74% worse than a flat budget. The ceiling carries most of the available value, while a 2x ceiling retains only about 38% of the gain. The production curve therefore needs a floor near 0.2x and a ceiling of at least 4x, but it must not be activated at full strength until the head has demonstrated useful ordering.

The frozen-trunk probe trained fresh scalar predictors on 2,400 positions per fold and evaluated 3,000 out-of-fold predictions. It found a Spearman correlation of 0.592, but applying the complete oracle-shaped budget distribution captured -13.95% of oracle gain. This means that signal is visible in the frozen features while that small-data predictor is not good enough to drive a full-strength allocator. It does **not** show that a jointly trained head exposed to tens of thousands of new labels cannot work, and it does not justify vetoing the live experiment.

## Per-generation lifecycle

The system operates as an asynchronous loop:

1. Self-play for generation `g` uses the allocator policy published before that generation. During warm-up this is a flat budget because the blend is zero.
2. Training completes and publishes an immutable checkpoint for generation `g`.
3. A separate background label job samples positions uniformly from that generation's newly produced replay observations.
4. Each normal 512-position shard evaluates the frozen generation-`g` checkpoint and persists its raw search-budget scores before any deep target is known.
5. After all shard predictions complete, the coordinator performs one lightweight post-processing pass: it aggregates the bounded quantile predictions, constructs every candidate's exactly mean-preserving budget vector across the complete sample, and records each position's union of required policy-checkpoint visits.
6. The pool searches deterministic chunks of at most 512 roots to the baseline, all required counterfactual checkpoint budgets, and the configured deep limit while retaining policy snapshots.
7. Each worker atomically writes its shard artifact and returns only a lightweight completion record. The coordinator waits for every shard and verifies exact position coverage and checksums. Failed shards are retried; a partial generation is never finalized.
8. One generation-wide finalization pass computes every raw `KL(pi_deep || pi_baseline)`, converts the complete generation sample to quantile targets, and scores every configured blend candidate from the recorded checkpoint policies at equal mean visit spend.
9. The finalizer writes the final deep policies back as replay training samples together with their eligible `search_budget` targets and source-generation metadata.
10. Only after replay write-back succeeds, the controller updates its trailing candidate utilities and atomically publishes the new blend state for the next production generation that has not started.
11. The completed manifest makes the entire finalization idempotent. Reprocessing the same source generation neither duplicates evidence nor applies the blend update twice.
12. Training and self-play continue without waiting for the label job. A bounded queue and maximum useful lag prevent the background work from accumulating indefinitely.

The labeler is generation-triggered and separate from the evaluation manager. Evaluations are wall-clock-cadenced and independently disableable; search-budget labels and allocator evidence are part of the training loop and must not inherit those semantics.

## Label generation

### Target

For baseline policy `pi_baseline` and final deep policy `pi_deep`, the raw target is:

```text
raw_difficulty = KL(pi_deep || pi_baseline)
```

Raw KL spans several orders of magnitude, with a measured median near 0.053 and maximum near 11.2. Regressing it directly would overemphasize a few extreme samples. After every deep-search shard for one source generation has completed, the finalizer ranks the complete generation sample and converts it to an empirical quantile:

```text
search_budget_target = empirical_cdf(raw_difficulty)
```

The empirical CDF uses deterministic mid-ranks for ties, maps the lowest and highest non-tied values to zero and one, and assigns `0.5` in the degenerate one-position case. The persisted label record retains both raw KL and the normalized target. There is no rolling label-quantile window: each source generation supplies one representative random population, and the prediction was already recorded before any deep result existed.

### Sampling rate and scale

Sampling remains random and independent of the predicted budget. Prediction-dependent labelling would create a feedback loop and remove counterfactual evidence exactly where the allocator is uncertain.

The measurements favor approximately 1% of positions at 8x depth, costing roughly 7% additional self-play compute. The first live run deliberately uses 2% to shorten head warm-up and measures the resulting roughly 14-16% labelling overhead under shared GPU contention.

At 120,000 new positions per generation, a 2% sample produces about 2,400 new unique deep labels per generation and about 72,000 after 30 generations. Replay sampling can expose those labels many more times during training, but repeated draws are not counted as new positional coverage.

The sample fraction remains configurable for later runs, but 2% is the concrete first-run setting. Once all new observations for a source generation are known, the labeler selects exactly `floor(0.02 * position_count)` positions uniformly without replacement using the run seed, source generation, and stable position identity. This produces one deterministic generation sample of arbitrary size; 512 is only its execution shard size.

Every selected position participates in that generation's target quantile and shadow evaluation. Every successfully deep-searched position is then written into replay with its deep policy and eligible scalar target. There is no separate rolling calibration-sample window: older labelled positions remain available to ordinary training only for as long as the normal replay-capacity policy retains them.

### Deep-policy write-back

Deep searches are the highest-fidelity policy targets produced by the system. Their final search observations are written back into replay as ordinary training samples rather than discarded after producing the scalar label. Duplicate positions are acceptable; self-play already revisits positions, and each labelled observation retains its own source context and target identity.

No sample is downweighted because it received a low assigned budget. A low budget means the policy target was predicted to settle cheaply, not that its resulting target is untrustworthy.

### Counterfactual checkpoints

One deep search can cheaply retain policy snapshots at the baseline and at the visit limits needed to evaluate candidate allocator blends. `SearchCheckpointDetail::Policies` must remain available.

Prediction scores are computed inside the ordinary 512-position shards. The coordinator then waits at a lightweight barrier, aggregates only those scalars, and converts candidate blends into concrete per-position visit budgets with exact sample-wide mean normalization. The same shards are dispatched again for deep search with each position's union of required visit limits, plus the baseline and final deep limit. Normalizing independently inside each shard would answer a different allocation question and is not allowed. This two-phase shard flow permits exact shadow evaluation of all configured blend candidates without a separate monolithic prediction job or one deep search per candidate.

The production interface should express that directly: each deep-label request carries a sorted set of absolute policy-checkpoint visits and one final deep limit. Native search continues the root once and returns the requested policy snapshots. This is a small, explicit native API change and remains useful after the old adaptive threshold machinery is deleted.

Python roots do currently retain their native trees across calls, so staged calls can technically continue a search. That is not the intended implementation. The current public self-play budget is global; increasing it changes root arena capacity and invalidates existing roots, while workarounds would depend on the fast-search or adaptive-limit paths being removed. Re-entering Python at every checkpoint also adds synchronization and makes accidental repeated root-noise application easier. A single native deep search with explicit checkpoints has the simpler ownership and fidelity contract.

### Root reconstruction and noise

Each selected replay observation reconstructs a fresh root from its canonical position and game-history context. The labeler does not attempt to serialize or reproduce the production retained tree. Label searches disable Dirichlet root noise and continue one deterministic tree through the baseline, candidate checkpoints, and final deep limit.

This matches the fidelity measurements, which explicitly set `dirichlet_epsilon = 0.0` because root noise is exploration rather than target shape. It also avoids injecting a random realization that the pre-noise network head cannot predict. Fresh-root, noise-free labelling versus noisy production searches with retained roots remains a known distribution mismatch to monitor in the live run; the acting allocator is still judged by the matched live control.

## Deep-label worker topology

One generation produces one logical label job. The job is deterministically divided into consecutive 512-position shards, with one final remainder shard, so retries and multi-GPU execution preserve stable position ownership. `TreeArena` allocates the configured initial node capacity for every root, so this bound also limits host memory. A typical 2% sample of 120,000 positions creates about 2,400 labels and therefore four 512-position shards plus one approximately 352-position remainder.

The coordinator owns one run-lifetime `ProcessPoolExecutor` with the `spawn` multiprocessing context for both prediction and deep-search phases. Each worker claims one configured device during initialization, sets that CUDA device, and remains pinned to it; tasks never move one process between CUDA devices. Worker state lazily loads the first immutable checkpoint it receives and refreshes monotonically for later generation jobs, reusing the native process, CUDA context, and search engine. Generation jobs are serviced in source-generation order so a worker never needs to roll its model backward.

The pool has up to the configured device count and gives the next unclaimed shard to whichever worker finishes. With approximately 2,400 labels and a 512-position maximum, a generation normally has five shards and therefore uses at most five of eight eligible GPUs. A generation uses all eight only when its configured shard count reaches eight; otherwise the remaining devices continue their trainer and self-play work without a label task.

The first live-run configuration uses all eight GPU identifiers as eligible label devices and intentionally permits overlap with trainer and self-play processes. There is no special isolation for device 0 or device 1. Contention is accepted as part of the measured end-to-end wall-clock cost and must remain visible in trainer, self-play, and labeler throughput telemetry.

The native inference maximum batch size is 512 for these shards. Full shards can therefore submit one independent leaf per root, while the final remainder uses its actual smaller size. This is separate from the approximately 2,400 total positions in the generation job; no model batch contains all sampled positions.

Deep labels use `parallel_searches = 1`. Each device-pinned label process owns one inference worker with batch size 512 and two outstanding batches. With hundreds of independent roots, the executor can fill inference batches without multiple simultaneous descents in one tree, and sequential search avoids the measured policy-quality loss from parallel selection. Unlike production adaptive allocation, every label root has the same deep final limit, so there is little long-budget tail to keep fed. Any later change requires a direct label-fidelity and throughput measurement.

For sample fraction `f` and deep multiple `d`, label search costs approximately `f * d` of flat self-play visit compute: 2% at 8x is about 16%. Queue lag determines whether the shared eight-GPU pool keeps up under real contention. Workers write policy-heavy results directly to atomic shard artifacts rather than pickling them through the executor result pipe. The parent receives a typed manifest containing shard identity, device, counts, timings, checksums, and artifact paths. No in-memory coordinator owns irreplaceable search output.

## Network head and training target

The model gains one scalar `search_budget` output read from the root forward pass that already produces the policy prior. The output predicts relative search difficulty, not a raw visit count.

The target is eligible only on deep-labelled replay samples. Training uses the same explicit mask pattern as `IneligibleNextPolicyTarget`; unlabelled positions contribute normally to all primary losses and contribute nothing to the search-budget loss.

The head emits one unconstrained logit. Training applies `sigmoid` and masked L1 against the quantile target in `[0, 1]`. L1 uses the absolute prediction error directly and, unlike Smooth L1, introduces no transition-width parameter. The bounded output is the predicted quantile used by the allocator; there is no second empirical prediction-CDF calibration layer. The masked reduction divides by eligible sample weight, so the 2% label rate does not implicitly shrink the loss by 50x. The search-budget auxiliary loss weight is 0.2. It remains generation-schedulable as part of the common auxiliary-target interface, but 0.2 is the complete first-run schedule, and its gradient contribution is reported.

Telemetry persists both the raw logit and bounded quantile prediction so ordering, saturation, and calibration remain distinguishable.

The head and trunk train jointly during the live run. This is an important difference from the frozen-feature probe: the trunk can learn features useful for ranking search difficulty while still being dominated by the primary policy and value objectives.

The existing `search_correction` scalar path is useful plumbing precedent but not a semantic interface to preserve. It is replaced end to end by the precisely named `search_budget` output, prediction, replay target, eligibility flag, metrics, and native root field.

## Allocation rule

Let:

- `B` be the configured baseline number of **new simulations** for a position;
- `q` be the sigmoid-bounded quantile prediction from the network head;
- `m(q)` be a fixed nonlinear monotone multiplier curve normalized to mean one;
- `alpha` be the published allocator blend in `[0, 1]`.

The unrounded assigned budget is:

```text
budget = B * ((1 - alpha) + alpha * m(q))
```

For the first run, `B` is the visit schedule inherited by the existing v10 configuration from v9: 200 at generation 0, 300 at 10, 400 at 30, 500 at 50, 600 at 90, 700 at 180, 800 at 250, and 1,000 at 550. The deep-label target for source generation `g` is exactly `8 * B(g)`.

Consequently:

- `alpha = 0` is exactly the flat baseline;
- `alpha = 1` is the full learned allocation curve;
- intermediate values continuously reduce the variance and risk of the allocation;
- the ordering comes from the head, while the amount of dispersion comes from the calibrated blend.

The multiplier curve is fixed for a run. Refitting it from each generation's deep results would use information unavailable when those positions were searched, change the meaning of `alpha` from one generation to the next, and make shadow comparisons unstable. Changes to the curve are versioned experiment-configuration changes made between runs. The per-generation KL quantile target already absorbs changes in the raw KL scale and shape.

The first-run default is the measured noise-corrected oracle allocation histogram at mean 600 visits, with source multipliers above 4x clipped to 4x and the result divided by its exact clipped mean, `3761 / 4500`. It is the following right-open step function, with the final interval closed at one:

| Predicted quantile `q` | Exact multiplier `m(q)` | Approximation |
| ---: | ---: | ---: |
| `[0, 1186/3000)` | `750/3761` | 0.199415 |
| `[1186/3000, 1570/3000)` | `1500/3761` | 0.398830 |
| `[1570/3000, 1838/3000)` | `2250/3761` | 0.598245 |
| `[1838/3000, 2048/3000)` | `3000/3761` | 0.797660 |
| `[2048/3000, 2188/3000)` | `3750/3761` | 0.997075 |
| `[2188/3000, 2347/3000)` | `4500/3761` | 1.196490 |
| `[2347/3000, 2547/3000)` | `6000/3761` | 1.595320 |
| `[2547/3000, 2699/3000)` | `9000/3761` | 2.392981 |
| `[2699/3000, 2806/3000)` | `12000/3761` | 3.190641 |
| `[2806/3000, 1]` | `18000/3761` | 4.785961 |

The implementation uses the exact ratios rather than renormalizing the displayed decimal approximations. This curve has mean one under a uniform quantile distribution, a 0.199x floor, a 4.786x ceiling, and sends 72.9% of quantiles below baseline.

Shadow allocation and label calibration operate on different values but share one generation-wide barrier. Candidate budgets are normalized across the complete 2% generation sample, never independently inside 512-position execution shards. The KL values from that same complete sample are separately converted to the head's mid-rank targets.

Production allocation applies the published curve and blend to every self-play position, not only the 2% audit sample. It additionally maintains a deterministic cumulative compute ledger: prediction miscalibration, rounding, and finite-sample residuals are carried forward and corrected in later assignments without violating the configured floor or ceiling. Each generation therefore closes with total assigned new simulations matching the flat baseline total within a stated integer tolerance. Monitoring an expected mean of one is insufficient; the realized mean must be enforced. This ledger is allocator state, not another tuning parameter or label-calibration window.

Budgets are defined as additional simulations, not an absolute root visit count. This keeps compute accounting stable when production retains roots with existing visits. The native limit converts the assigned additional budget to its stopping condition using the root's starting visit count.

## Continuous shadow evaluation

Every randomly labelled position records:

- stable position and source-generation identifiers;
- the immutable model generation used for prediction and search;
- raw logit and bounded quantile prediction;
- baseline, candidate-budget, and final deep policies;
- concrete visit checkpoints;
- raw KL and quantile target;
- allocator-curve and calibrator configuration identity.

The evaluator considers a configured grid of blend candidates including zero and one. For every candidate it reconstructs the exact, mean-preserving vector of position budgets and reads the corresponding policy checkpoints. Its primary utility metric is improvement over the flat baseline in divergence from the deep policy:

```text
gain(alpha) = mean KL(pi_deep || pi_flat)
              - mean KL(pi_deep || pi_candidate(alpha))
```

Positive gain means that candidate produced better policy targets at equal mean search spend. The report also expresses this as a share of the deep-label oracle gain when the denominator is stable.

Rank correlation, calibration plots, regression loss, decile ordering, and target top-1 agreement are diagnostics. They help explain behavior but do not authorize allocation: a head can correlate with the target and still lose after the convexity cost of non-uniform budgets.

The shadow evaluation is computationally cheap once the deep search exists because it reuses its policy checkpoints. The deep search itself is not free and remains explicit training overhead.

## Automatic blend calibration

The controller consumes one complete source generation at a time in source-generation order. For candidate `alpha` and completed source generation `g`, it computes the paired position-level mean:

```text
generation_gain[g, alpha] =
    mean_i(KL(pi_deep_i || pi_flat_i)
           - KL(pi_deep_i || pi_candidate_i(alpha)))
```

The persisted EMA is initialized to the first completed generation's gain and thereafter updated as:

```text
ema_gain[g, alpha] =
    (1 - ema_decay) * ema_gain[g - 1, alpha]
    + ema_decay * generation_gain[g, alpha]
```

The default `ema_decay` is `0.2`. The EMA carries information across the approximately 2,400 random positions in each completed generation without introducing a second evidence window, bootstrap, confidence level, or minimum-count system.

Until the configured number of completed warm-up label generations, default 30, the published blend is hard-clamped to zero. Afterwards a nonzero candidate is eligible exactly when both its current `generation_gain` and its `ema_gain` are strictly positive and its reconstructed mean compute matches the flat baseline within the allocator's integer tolerance.

For publication, consider every eligible candidate at or below the current blend and every eligible candidate no more than the configured maximum upward step above it. Select the candidate in that set with the largest EMA gain; ties select the lower blend. The default maximum upward step is `0.1`, so the default candidate grid advances at most from 0 to 0.1 to 0.2 across completed label generations, while decreases apply immediately. If no nonzero candidate is eligible, publish zero.

An unfinished background job does not change the published blend; production keeps using the latest completed state. A terminal label-job failure, incompatible configuration or model lineage, unreadable state, or invalid mean-compute reconstruction publishes zero. State publication is atomic and names the first not-yet-started production generation to which it applies. Reprocessing a completed source generation is idempotent and cannot update the EMA or blend twice.

## Warm-up and permanent audit stream

Until approximately 30 source-generation label jobs have completed, `alpha` is hard-clamped to zero regardless of apparent early shadow performance. This accumulates approximately 72,000 representative labels at the first-run scale while preserving the known flat-budget behavior. If background labelling falls behind, the clamp lasts longer than 30 production generations rather than activating from too little evidence.

Completing the warm-up only permits activation; it does not force it. If no candidate has positive current and EMA gain, the allocator stays flat.

Random audit labelling continues after activation. Positions in that stream receive their prescribed deep search independently of the production-assigned budget. Without this permanent exploration stream, an acting allocator would preferentially observe positions it already believes are difficult and could no longer measure its own counterfactual errors across the full position distribution.

Published decisions apply only to later generations, with an explicit lag that prevents a generation from calibrating itself using targets unavailable when its searches ran.

## Parallelism follows the assigned budget

Per-search parallelism scales with the assigned visit budget so the number of sequential inference rounds stays approximately bounded. The production rule is:

```text
parallel_searches = min(16, next_power_of_two(ceil(assigned_new_visits / 200)))
```

This produces:

| Assigned new visits | Parallel searches |
| ---: | ---: |
| 100 | 1 |
| 300 | 2 |
| 600 | 4 |
| 1,600 | 8 |
| 2,400 or more | 16 |

This mapping is per search task, not one global value shared by every active root. It never exceeds 16, including when the late-generation baseline and 4.786x curve ceiling assign more than 2,400 visits.

The quality assumption remains unverified. Previous measurements changed parallelism at fixed total visits and therefore confounded more parallel work with fewer sequential rounds. The shadow evaluator can validate target fidelity at the intended checkpoint budgets, but only a live run can establish GPU utilization, inference-batch health, latency, and any quality loss caused by parallel selection. The rule is configuration and must be instrumented independently from allocator quality, but the formula and cap above are the first-run defaults.

## Background execution and replay ownership

The generation labeler must not stall training. It uses immutable checkpoint and replay references, bounded concurrency, and a persistent job record with explicit states. A maximum generation lag determines when stale unstarted work is skipped rather than allowed to grow without bound.

Completed label and replay writes are transactional or otherwise idempotent. A retry must not create duplicate journal evidence accidentally; deliberate duplicate deep replay samples remain acceptable because they are identified as separate training observations.

Each deep replay sample preserves the normal game and position context required by the canonical replay schema, replaces the policy target with the final deep policy, and adds the eligible scalar target. Materialization no longer filters observations on `full_search`.

## Configuration model

Configuration is divided by ownership without duplicating fields:

### Deep labelling

- random sample fraction, default `0.02`, converted to `floor(fraction * new_generation_positions)`;
- deep-search multiple, default and first-run value `8` times the source-generation baseline;
- one active logical generation job and all unique trainer GPU identifiers eligible by default;
- maximum unstarted generation lag, default `2`;
- counterfactual checkpoints derived from the candidate grid, curve, and generation baseline rather than separately authored.

### Search-budget target

- masked L1 on the sigmoid-bounded scalar;
- generation-schedulable auxiliary loss weight, default and complete first-run schedule `0.2`;
- eligible-count and gradient-contribution telemetry.

### Allocation

- baseline visit schedule inherited from v10;
- the exact stepwise quantile-to-multiplier curve above;
- floor `750/3761` and ceiling `18000/3761`;
- exact mean-preservation, cumulative-ledger, and rounding policy.

### Blend calibration

- completed warm-up label generations, default `30`;
- blend candidate grid, default `[0.0, 0.1, ..., 1.0]`;
- EMA decay, default `0.2`;
- maximum upward step, default `0.1`.

These defaults belong to the canonical typed calibrator configuration, so an ordinary run need not repeat them. The positive current-gain and EMA-gain tests, lower-blend tie break, immediate retreat, and zero fallback are controller semantics rather than additional tuning parameters.

### Parallel execution

- next-power-of-two rule targeting 200 sequential rounds;
- minimum one and maximum 16 parallel searches.

### Label-worker execution

- eligible device identifiers, initially all eight live-run GPUs;
- spawned process-pool worker initialization and device pinning;
- positions per persisted shard;
- roots per native search chunk;
- one inference worker, maximum batch size 512, and two outstanding batches per process;
- fixed label-search parallelism, one.

## Required telemetry

The live run must make both learning and acting behavior inspectable.

### Label pipeline

- selected, started, completed, failed, retried, and skipped jobs;
- source generation, completion lag, queue depth, and GPU time;
- deep-labelled positions and fraction by generation;
- raw KL and quantile-target distributions;
- deep replay write counts and idempotency conflicts.

### Head quality

- eligible sample count and masked auxiliary loss;
- raw-logit and bounded quantile-prediction distributions;
- Spearman rank correlation and calibration by prediction decile;
- realized raw KL and oracle utility by prediction decile;
- target top-1 agreement by candidate budget.

### Shadow allocator

- realized mean, variance, floor share, and ceiling share for every blend candidate;
- candidate divergence from deep policy and gain over flat;
- share of oracle gain when stable;
- current-generation gain and trailing EMA for every candidate;
- selected blend, prior blend, application generation, decision reason, and failed eligibility conditions.

### Acting allocator and runtime

- assigned visits and parallel searches per position;
- exact generation mean-spend residual;
- retained-root starting visits versus assigned new visits;
- inference batch size, sequential rounds, search latency, and GPU utilization by budget band;
- self-play positions per second and wall-clock generation time;
- total deep-label overhead;
- target-fidelity yardsticks and Elo trend.

## Removal and migration scope

The replacement is complete only when the superseded mechanisms are removed rather than left dormant.

### Native code

Remove:

- the threshold-driven adaptive search limit and its deterministic stop machinery;
- `SearchCorrectionGate` and learned-gate stop paths;
- the disabled/adaptive limit variants that exist only for the old design;
- stop reasons and telemetry used only by threshold relaxation or the learned gate;
- fast/full admission staging, admission counts, request flags, and separate fast/full limits;
- the assumption that one `parallel_searches` value applies to every search.

Add:

- a predicted-budget search-limit variant that reads the root scalar once the root is expanded;
- an assigned-additional-visits field with explicit retained-root semantics;
- per-search parallelism derived from the assigned budget;
- a per-request sorted policy-checkpoint visit set for the shadow label job;
- one-pass continuation to the final deep limit without Python checkpoint round trips.

Retain `SearchCheckpointDetail::Policies`.

### Python code

Remove `AdaptiveFullSearchBudgetConfiguration` and all eight old adaptive controls:

- `minimum_visits`;
- `observation_interval`;
- `leader_stability_window`;
- `root_value_tolerance`;
- both top-visit-share threshold schedules;
- `threshold_relaxation_visits`;
- the learned-gate control.

Also remove:

- `minimum_search_correction_to_unlock_tail`;
- the `search_correction` target, head, output, result fields, and telemetry;
- `full_search_probability`, fast-search limits, forced-fast-after-ply controls, and `full_search` booleans;
- materialization and archive/restart filters based on `full_search`;
- full-search-only sample and performance telemetry;
- obsolete adaptive calibration commands and reports whose concepts no longer exist.

Replace them with:

- the eligible `search_budget` target through `targets.py`, `materialization.py`, `columnar.py`, and `store.py`;
- the generation-triggered deep-label and replay-write-back job;
- persistent shadow evidence and blend-calibration state;
- one production allocator configuration;
- updated analysis, validation, configuration, generated bindings, and tests.

All played positions become replay samples. There is no weighting adjustment based on assigned budget.

## Implementation sequence

1. **Target and model path.** Add the scalar head, eligibility mask, replay schema, columnar storage, materialization support, losses, metrics, and generated native bindings. Remove the `full_search` materialization filter.
2. **Deep-label pipeline.** Add deterministic generation sampling, shard-local prediction, the scalar-aggregation barrier, exact generation-wide candidate-budget construction, fixed 512-position shards plus a final remainder, a run-lifetime spawn-based device-pinned process pool, explicit native checkpoint-visit requests, monotonic immutable-checkpoint refresh, noise-free baseline/candidate/deep policy capture from one continued fresh root, atomic shard artifacts, quantile normalization, persistent job state, and deep-policy replay write-back. Use one inference worker with batch size 512 and two outstanding batches per process, and label-search parallelism one.
3. **Shadow evaluator and calibrator.** Add exact candidate-budget reconstruction, equal-compute generation utility scoring, EMA state, deterministic selection, atomic publication, fail-closed behavior, and complete diagnostics.
4. **Native allocator in shadow-safe form.** Add predicted budget plumbing, mean-preserving generation allocation, retained-root accounting, and per-search parallelism. Ship with the published blend clamped to zero until calibrator conditions permit otherwise.
5. **Remove superseded systems.** Delete threshold adaptation, `SearchCorrectionGate`, `search_correction`, fast/full admission and configuration, old filters, obsolete tools, stale telemetry, and compatibility layers. Migrate production configuration and documentation in the same phase.
6. **Component and integration validation.** Exercise target masking, replay round trips, job retries, quantiles, exact mean preservation, checkpoint scoring, calibrator publication and fallback, retained roots, native stopping, and heterogeneous parallel searches.
7. **Live 8-GPU training run.** Accumulate the warm-up evidence, inspect the shadow distribution, and then allow the controller to advance or retreat the blend automatically from the documented defaults.

Feature-sized commits should follow these boundaries so each stage is independently reviewable and reversible.

## Live experiment

The meaningful validation is an actual training run on representative self-play, not a larger frozen offline classifier exercise.

The first run uses the available eight GPUs concurrently for training, self-play, and eligible label-worker devices. It keeps `alpha = 0` until approximately 30 source-generation label jobs have completed while the head trains and the shadow evaluator accumulates evidence. At 2% labelling this is approximately 72,000 new unique labelled positions before activation becomes possible.

Before the first acting generation, inspect:

- label throughput and lag;
- prediction and target distributions;
- ordering by decile;
- exact equal-compute candidate gains;
- whether current-generation gain and the EMA react reasonably across generations;
- per-budget parallelism behavior;
- label-worker memory, GPU contention, pool occupancy, and generation lag;
- total wall-clock overhead.

The run then continues with automatic blend calibration. The central operational questions are:

- Does the blend leave zero and remain safely above it?
- Does it rise as label coverage and model quality improve?
- Does realized mean search spend remain equal to the baseline?
- Does target top-1 agreement improve at equal mean visits?
- Does self-play throughput remain acceptable when deep-budget searches receive more parallel work?
- Does the per-generation strength yardstick and eventual Elo trend match or beat the flat-budget control at equal wall-clock time?

The labeler overhead must be included in wall-clock comparisons, even though its deep policies also improve the replay targets. A clean later A/B should compare the acting allocator against a control that runs the same deep-label pipeline with `alpha = 0`, isolating allocation value from the value and cost of deep-policy write-back.

## Risks to watch

- **Parallelism may not be free at fixed sequential depth.** If additional parallel selections reduce policy quality, the deepest assigned searches are harmed exactly where the allocator spends most.
- **Target fidelity may not translate to training strength.** All current measurements optimize policy-target agreement, not downstream learning or Elo.
- **Non-stationarity can outrun calibration.** The head, trunk, game distribution, and baseline search all evolve. EMA lag and publication lag must remain visible.
- **Retained roots can break compute accounting.** Allocation must govern new simulations and report starting visits separately.
- **The deepest policy is a reference, not truth.** Its own search noise and finite depth limit both labels and shadow utility estimates.
- **Quantile drift can distort the curve.** The head is trained against a separately normalized source-generation population while production receives an evolving distribution. Prediction and realized multiplier distributions must remain visible, and exact spend accounting must not assume perfect calibration.
- **Acting creates feedback.** Permanent random audit labelling is required even after the blend becomes nonzero.
- **Asynchronous jobs can train on stale labels.** Source model generation and completion lag must be preserved and bounded.
- **Deep replay samples change more than the scalar head.** Their stronger policy targets are desirable, but they mean the first live run measures the combined training system. The later matched-labeler A/B isolates allocator value.

## Settled decisions

- Use `KL(pi_deep || pi_baseline)` rather than total variation.
- Quantile-normalize the scalar target.
- Generate labels at exactly 8x the source-generation baseline depth on a small random fraction of positions.
- Train one scalar auxiliary head jointly with the trunk and mask its loss to labelled samples.
- Keep an independent random audit stream permanently.
- Use a 2% label sample for the first live run.
- Compute predictions inside the normal shards, then aggregate their scalars before constructing generation-wide candidate checkpoint budgets.
- Run deterministic 512-root shards plus a final remainder through a spawn-based, device-pinned process pool over all eight eligible GPUs.
- Use one label inference worker per process, batch size 512, two outstanding batches, and `parallel_searches = 1`.
- Write policy-heavy shard output atomically and return lightweight manifests to the coordinator.
- Reconstruct fresh label roots, disable Dirichlet noise, and continue one tree through baseline and deep checkpoints.
- Train the sigmoid-bounded quantile prediction with masked L1 at loss weight 0.2.
- Use the bounded head output directly as the predicted quantile; do not add a second prediction-CDF calibration layer.
- Write final deep policies back into replay.
- Make every played position a replay sample and remove the fast/full split.
- Use the fixed measured-oracle multiplier curve above with exact floor `750/3761` and ceiling `18000/3761`.
- Preserve exact mean new-simulation spend by construction.
- Scale parallel searches per assigned budget.
- Hold the blend at zero until approximately 30 source-generation label jobs have completed.
- Finalize labels and shadow evidence once per complete source generation, never per shard.
- Choose and continuously update the blend from current and EMA-smoothed equal-compute shadow utility, not loss or correlation.
- Use default blend candidates `[0.0, 0.1, ..., 1.0]`, EMA decay `0.2`, and maximum upward step `0.1`.
- Increase blend by at most the configured upward step, reduce it immediately, and fail closed to zero.
- Remove all old threshold, learned-gate, `search_correction`, and fast/full machinery.

## Decisions to make from live shadow data

- run length and matched-control schedule.

## Acceptance criteria

Implementation is ready for the live experiment when:

- eligible and ineligible `search_budget` targets round-trip through replay and train with the correct mask;
- batches with no eligible labels contribute zero finite search-budget loss, while eligible reduction is normalized independently of the 2% incidence;
- every self-play position materializes regardless of its assigned budget;
- deep-label jobs are generation-triggered, asynchronous, bounded, retryable, and idempotent;
- prediction is recorded before the deep target and remains attributable to an immutable source checkpoint;
- deterministic sampling selects exactly `floor(0.02 * new_generation_positions)` positions without replacement for the first run;
- label shards have stable ownership and can be processed by one or multiple workers without duplicate evidence;
- generation-wide candidate budgets are constructed after shard-local prediction and before deep-search dispatch, preserving the global sample mean;
- label-search root chunks contain at most 512 roots and bound host memory while sustaining measured inference occupancy;
- spawned workers remain pinned to one CUDA device and reuse one loaded checkpoint and search engine;
- shard artifacts survive worker or coordinator failure and the parent verifies exact coverage and checksums;
- partial generations cannot produce labels, replay write-back, shadow evidence, or a blend update;
- label searches disable root noise and retries reproduce checkpoint policies;
- one native continuation returns policies at every explicitly requested visit checkpoint;
- generation-wide KL values deterministically produce mid-rank quantile targets in `[0, 1]`;
- every completed deep-labelled position contributes to that source generation's quantiles and shadow score, then enters replay with its deep policy;
- deep policies and their finalized scalar targets are written back as valid replay samples before blend publication;
- the exact default multiplier curve integrates to one and reproduces its documented floor, ceiling, and below-baseline share;
- shadow candidate budgets preserve exact mean new-simulation spend;
- shadow utility is reconstructed from exact policy checkpoints and reproduces known offline cases;
- the warm-up clamp prevents nonzero production blend before the configured number of source-generation label jobs has completed;
- no nonzero blend is published unless both current-generation and EMA gain are positive;
- selection maximizes EMA gain with lower-blend tie breaking, upward changes respect the configured cap, and downward changes apply immediately;
- an unfinished job preserves the latest completed state, while terminal failure, invalid compute, or incompatible state returns the allocator to zero;
- each completed source generation updates EMA and blend state exactly once and publishes only for a production generation that has not started;
- native search stops at the assigned additional-visit budget with retained roots;
- parallelism differs correctly between simultaneous searches with different budgets and follows the documented rule without exceeding 16;
- old adaptive threshold, learned gate, `search_correction`, and fast/full code and configuration are gone;
- telemetry can explain every label, candidate score, blend transition, spend residual, and fallback;
- an 8-GPU live run completes warm-up and demonstrates whether the allocator can safely capture positive oracle gain.

Success after activation means matching or beating the flat-budget, matched-labeler control at equal wall-clock time while preserving mean search spend and improving mean policy-target agreement. The blend need not reach 1.0; a stable partial blend that captures reliable value is a successful outcome.

## Evidence and related artifacts

- [Chess search findings](../analysis/chess-search-findings-20260827.md)
- [RTX 3060 chess search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md)
- [Frozen-trunk adaptive budget probe](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md)
- `measure_policy_target_fidelity --per-position-output`
- `analyse_budget_allocation`
- `sample_chess_search_positions`
- `validate_adaptive_replay` (to be replaced or repurposed as part of the migration)
