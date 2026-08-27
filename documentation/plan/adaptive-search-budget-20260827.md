# Learned adaptive search budget with continuous shadow calibration

**Status:** implementation plan, revised 2026-08-27

## Decision

Replace both the threshold-based adaptive full-search mechanism and the fast/full search split with one learned per-position search budget.

The network predicts a scalar search-difficulty quantile from the root forward pass. A fixed nonlinear allocation curve maps that quantile to a mean-preserving visit multiplier. The allocator is introduced gradually through a blend coefficient, but that coefficient is not advanced on a fixed calendar and is not chosen from training loss or correlation alone. It is calibrated continuously from the counterfactual search checkpoints produced by the deep-label job.

The first live training run starts with the learned allocator completely inert. For approximately the first 30 generations:

- the head trains;
- random positions receive deep labels;
- the allocator is evaluated in shadow mode;
- every production search still receives the flat baseline budget;
- the published allocator blend remains zero.

After the warm-up, a conservative calibrator may increase the blend only when held-out shadow evidence says that a candidate blend improves policy-target fidelity at equal mean search compute. It reduces the blend immediately when the evidence deteriorates and falls back to zero when no candidate is demonstrably safe.

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

Around eight times the baseline is the intended operating point: it captures about 92% of what a sixteen-times label provides, whereas four times captures about 73%. Going deeper on fewer randomly selected positions is more useful than going shallower on more positions because label cost scales approximately linearly with visits.

Non-uniform allocation has a serious convexity penalty before ordering quality is considered. Randomly dispersing the measured good budget distribution scored approximately 74% worse than a flat budget. The ceiling carries most of the available value, while a 2x ceiling retains only about 38% of the gain. The production curve therefore needs a floor near 0.2x and a ceiling of at least 4x, but it must not be activated at full strength until the head has demonstrated useful ordering.

The frozen-trunk probe trained fresh scalar predictors on 2,400 positions per fold and evaluated 3,000 out-of-fold predictions. It found a Spearman correlation of 0.592, but applying the complete oracle-shaped budget distribution captured -13.95% of oracle gain. This means that signal is visible in the frozen features while that small-data predictor is not good enough to drive a full-strength allocator. It does **not** show that a jointly trained head exposed to tens of thousands of new labels cannot work, and it does not justify vetoing the live experiment.

## Per-generation lifecycle

The system operates as an asynchronous loop:

1. Self-play for generation `g` uses the allocator policy published before that generation. During warm-up this is a flat budget because the blend is zero.
2. Training completes and publishes an immutable checkpoint for generation `g`.
3. A separate background label job samples positions uniformly from that generation's newly produced replay observations.
4. The frozen generation-`g` checkpoint predicts a raw search-budget score for each selected position before its deep target is known.
5. A dedicated label worker searches each sampled position to the baseline, all required counterfactual checkpoint budgets, and the configured deep limit while retaining policy snapshots. The generation job is processed in bounded root chunks rather than one native call containing every sampled position.
6. The job computes `KL(pi_deep || pi_baseline)` and converts it to the current quantile-normalized target.
7. The final deep policy is written back as a replay training sample, together with the eligible `search_budget` target and source-generation metadata.
8. The shadow evaluator scores every configured blend candidate from the recorded checkpoint policies at equal mean visit spend.
9. The calibrator updates its persistent evidence journal and atomically publishes the blend and calibration state for a future generation.
10. Training and self-play continue without waiting for the label job. A bounded queue and maximum useful lag prevent the background work from accumulating indefinitely.

The labeler is generation-triggered and separate from the evaluation manager. Evaluations are wall-clock-cadenced and independently disableable; search-budget labels and allocator evidence are part of the training loop and must not inherit those semantics.

## Label generation

### Target

For baseline policy `pi_baseline` and final deep policy `pi_deep`, the raw target is:

```text
raw_difficulty = KL(pi_deep || pi_baseline)
```

Raw KL spans several orders of magnitude, with a measured median near 0.053 and maximum near 11.2. Regressing it directly would overemphasize a few extreme samples. The training target is therefore its empirical quantile in a recent, generation-aware label population:

```text
search_budget_target = empirical_cdf(raw_difficulty)
```

The persisted label record retains both raw KL and the normalized target. Quantile estimation must be deterministic and robust to asynchronous completion. Its exact windowing policy remains a configuration decision; it must not leak a position's deep result into the prediction recorded for that same position.

### Sampling rate and scale

Sampling remains random and independent of the predicted budget. Prediction-dependent labelling would create a feedback loop and remove counterfactual evidence exactly where the allocator is uncertain.

The measurements favor approximately 1% of positions at about 8x depth, costing roughly 7% additional self-play compute. A first live run may deliberately use 2% to shorten head warm-up if the available 4-8 GPU allocation can sustain the roughly 14-16% labelling overhead.

At 120,000 new positions per generation, a 2% sample produces about 2,400 new unique deep labels per generation and about 72,000 after 30 generations. Replay sampling can expose those labels many more times during training, but repeated draws are not counted as new positional coverage.

The final choice between 1% and 2% belongs in the run configuration and capacity plan, not in the architecture.

### Deep-policy write-back

Deep searches are the highest-fidelity policy targets produced by the system. Their final search observations are written back into replay as ordinary training samples rather than discarded after producing the scalar label. Duplicate positions are acceptable; self-play already revisits positions with different noise and targets.

No sample is downweighted because it received a low assigned budget. A low budget means the policy target was predicted to settle cheaply, not that its resulting target is untrustworthy.

### Counterfactual checkpoints

One deep search can cheaply retain policy snapshots at the baseline and at the visit limits needed to evaluate candidate allocator blends. `SearchCheckpointDetail::Policies` must remain available.

For each label batch, prediction scores are collected first. Candidate blends are then converted into concrete per-position visit budgets with exact batch-mean normalization. The deep searches retain the union of those visit limits, plus the baseline and final deep limit. This permits exact shadow evaluation of all configured blend candidates without running a separate search for each candidate.

The production interface should express that directly: each deep-label request carries a sorted set of absolute policy-checkpoint visits and one final deep limit. Native search continues the root once and returns the requested policy snapshots. This is a small, explicit native API change and remains useful after the old adaptive threshold machinery is deleted.

Python roots do currently retain their native trees across calls, so staged calls can technically continue a search. That is not the intended implementation. The current public self-play budget is global; increasing it changes root arena capacity and invalidates existing roots, while workarounds would depend on the fast-search or adaptive-limit paths being removed. Re-entering Python at every checkpoint also adds synchronization and makes accidental repeated root-noise application easier. A single native deep search with explicit checkpoints has the simpler ownership and fidelity contract.

## Deep-label worker topology

One generation produces one logical label job. The job is deterministically divided into bounded shards and root chunks so retries and optional multi-GPU execution preserve stable position ownership.

The initial deployment uses one label-worker process on one explicitly reserved GPU, preferably device 1 rather than rank-zero device 0. Merely selecting GPU 1 does not isolate the worker when the trainer's DDP topology also includes GPU 1; the live-run topology must actually reserve that device or accept and measure contention.

The worker does not submit all approximately 2,500 roots in one native call. `TreeArena` allocates the configured initial node capacity for every root, so root count multiplied by the roughly 8x deep limit can consume substantial host memory. The worker instead processes chunks large enough to saturate inference while bounding tree memory, initially testing 512 and 1,024 roots per chunk.

Inference batch size is independent of the total number of labelled positions. A maximum batch size of 1,024 is a reasonable benchmark candidate and has already run in the search experiments, but it is not assumed to be optimal for the live checkpoint and GPU. Benchmark 320, 512, and 1,024 for simulations per second, actual average batch size, CUDA memory, graph-capture cost, and host memory before fixing the run configuration.

Deep labels should initially use `parallel_searches = 1`. With hundreds of independent roots, the executor can fill inference batches without multiple simultaneous descents in one tree, and sequential search avoids the measured policy-quality loss from parallel selection. Unlike production adaptive allocation, every label root has the same deep final limit, so there is little long-budget tail to keep fed. Any later increase requires a direct label-fidelity and throughput measurement.

Additional label GPUs are a throughput scaling option, not a separate architecture. For sample fraction `f` and deep multiple `d`, label search costs approximately `f * d` of flat self-play visit compute: 2% at 8x is about 16%. One label GPU can therefore keep up with roughly six equally fast self-play GPU-equivalents before accounting for trainer contention and throughput differences. Start with one worker, monitor queue lag, and add another identically configured worker when the bounded queue cannot finish each generation's labels before the maximum useful lag. Workers claim deterministic persisted shards; no in-memory coordinator owns irreplaceable progress.

## Network head and training target

The model gains one scalar `search_budget` output read from the root forward pass that already produces the policy prior. The output predicts relative search difficulty, not a raw visit count.

The target is eligible only on deep-labelled replay samples. Training uses the same explicit mask pattern as `IneligibleNextPolicyTarget`; unlabelled positions contribute normally to all primary losses and contribute nothing to the search-budget loss.

The head and trunk train jointly during the live run. This is an important difference from the frozen-feature probe: the trunk can learn features useful for ranking search difficulty while still being dominated by the primary policy and value objectives.

The existing `search_correction` scalar path is useful plumbing precedent but not a semantic interface to preserve. It is replaced end to end by the precisely named `search_budget` output, prediction, replay target, eligibility flag, metrics, and native root field.

## Allocation rule

Let:

- `B` be the configured baseline number of **new simulations** for a position;
- `s` be the raw network scalar;
- `q = F_prediction(s)` be its calibrated percentile under a recent independent shadow-prediction distribution;
- `m(q)` be a fixed nonlinear monotone multiplier curve normalized to mean one;
- `alpha` be the published allocator blend in `[0, 1]`.

The unrounded assigned budget is:

```text
budget = B * ((1 - alpha) + alpha * m(q))
```

Consequently:

- `alpha = 0` is exactly the flat baseline;
- `alpha = 1` is the full learned allocation curve;
- intermediate values continuously reduce the variance and risk of the allocation;
- the ordering comes from the head, while the amount of dispersion comes from the calibrated blend.

The multiplier curve is configuration, with an intended floor near 0.2x and ceiling of at least 4x. It must map the median prediction below 1x: the measured oracle assigns approximately 73% of positions less than the baseline budget.

The curve is normalized to mean one under the calibrated prediction-percentile distribution. Production additionally maintains a deterministic cumulative compute ledger: rounding and finite-sample residuals are carried forward and corrected in later assignments without violating the configured floor or ceiling. Each generation therefore closes with total assigned new simulations matching the flat baseline total within a stated integer tolerance. Monitoring an expected mean of one is insufficient; the realized mean must be enforced.

Budgets are defined as additional simulations, not an absolute root visit count. This keeps compute accounting stable when production retains roots with existing visits. The native limit converts the assigned additional budget to its stopping condition using the root's starting visit count.

## Continuous shadow evaluation

Every randomly labelled position records:

- stable position and source-generation identifiers;
- the immutable model generation used for prediction and search;
- raw prediction and calibrated prediction percentile;
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

The calibrator follows the safety shape of resignation calibration: persistent evidence, configuration identity, conservative confidence bounds, an explicit production-generation boundary, and asymmetric updates.

Its state consists of:

- an idempotent position-level evidence journal;
- per-generation candidate metrics;
- a bounded rolling evidence window;
- a trailing EMA of candidate utility for non-stationary trend tracking;
- the currently published blend and the generation from which it applies;
- configuration and model-lineage hashes;
- diagnostics explaining every selection or fallback.

The EMA smooths whether a candidate continues to help as the model changes. It is not, by itself, an uncertainty estimate. Conservative confidence bounds come from the rolling position-level evidence, using a generation-aware block bootstrap or an equivalent method that does not treat repeated or same-generation positions as fully independent.

Before a nonzero candidate can be published, all of the following must hold:

- the configured first production generation has been reached; the initial proposal is generation 30;
- minimum counts of labelled positions and represented generations have been reached;
- realized candidate mean compute is within tolerance of the flat baseline;
- its conservative lower confidence bound on equal-compute gain exceeds the configured safety margin;
- recent generation-level evidence and the trailing EMA do not indicate deterioration;
- required prediction-distribution and calibration sanity checks pass;
- the condition has held for the configured number of consecutive calibration updates.

Among eligible candidates, the calibrator chooses the blend with the best conservative utility, subject to a maximum upward step per generation. It may tighten cautiously, for example from 0 to 0.1 to 0.2, rather than jumping to the largest passing candidate.

Relaxation is asymmetric. If the current blend stops meeting the safety rule, the calibrator immediately publishes the highest lower blend that remains safe, or zero when none does. No evidence, stale evidence, incompatible configuration, a failed label job, or an unreadable state artifact all fail closed to zero.

The exact candidate grid, evidence window, EMA half-life, confidence level, safety margin, minimum counts, consecutive-update requirement, and maximum upward step are deliberately unresolved. They must be selected from live shadow distributions before the first acting generation. The implementation must expose them as one canonical typed calibrator configuration and emit enough diagnostics to reproduce every choice.

State publication is atomic. Reprocessing the same completed job is idempotent. A newly started worker can reconstruct the same decision from the persisted journal without relying on in-memory state.

## Warm-up and permanent audit stream

During the first approximately 30 generations, `alpha` is hard-clamped to zero regardless of apparent early shadow performance. This accumulates representative labels while preserving the known flat-budget behavior.

Reaching generation 30 only permits calibration; it does not force activation. If the evidence thresholds are not met, the allocator stays flat.

Random audit labelling continues after activation. Positions in that stream receive their prescribed deep search independently of the production-assigned budget. Without this permanent exploration stream, an acting allocator would preferentially observe positions it already believes are difficult and could no longer measure its own counterfactual errors across the full position distribution.

Published decisions apply only to later generations, with an explicit lag that prevents a generation from calibrating itself using targets unavailable when its searches ran.

## Parallelism follows the assigned budget

Per-search parallelism scales with the assigned visit budget so the number of sequential inference rounds stays approximately bounded. An initial mapping to validate is:

| Assigned new visits | Parallel searches |
| ---: | ---: |
| 100 | 1 |
| 300 | 2 |
| 600 | 4 |
| 1,600 | 8 |
| 2,400 | 16 |

This mapping is per search task, not one global value shared by every active root.

The assumption remains unverified. Previous measurements changed parallelism at fixed total visits and therefore confounded more parallel work with fewer sequential rounds. The shadow evaluator can validate target fidelity at the intended checkpoint budgets, but only a live run can establish GPU utilization, inference-batch health, latency, and any quality loss caused by parallel selection. The parallelism curve is consequently configuration and must be instrumented independently from allocator quality.

## Background execution and replay ownership

The generation labeler must not stall training. It uses immutable checkpoint and replay references, bounded concurrency, and a persistent job record with explicit states. A maximum generation lag determines when stale unstarted work is skipped rather than allowed to grow without bound.

Completed label and replay writes are transactional or otherwise idempotent. A retry must not create duplicate journal evidence accidentally; deliberate duplicate deep replay samples remain acceptable because they are identified as separate training observations.

Each deep replay sample preserves the normal game and position context required by the canonical replay schema, replaces the policy target with the final deep policy, and adds the eligible scalar target. Materialization no longer filters observations on `full_search`.

## Configuration model

Configuration is divided by ownership without duplicating fields:

### Deep labelling

- random sample fraction;
- deep-search multiple or absolute limit derived from the generation baseline;
- maximum concurrent jobs and GPU allocation;
- maximum generation lag;
- label-quantile window;
- retained counterfactual checkpoints.

### Allocation

- baseline visit schedule;
- quantile-to-multiplier curve;
- floor and ceiling;
- prediction-percentile calibration window;
- exact mean-preservation, cumulative-ledger, and rounding policy.

### Blend calibration

- first permitted production generation;
- blend candidate grid;
- rolling evidence window and minimum represented generations;
- EMA half-life;
- confidence method and level;
- safety margin;
- minimum labelled positions;
- consecutive-safe-update requirement;
- maximum upward step;
- conservative fallback.

### Parallel execution

- assigned-budget-to-parallel-searches curve;
- implementation limits imposed by inference batching and worker capacity.

### Label-worker execution

- reserved device identifiers and worker count;
- positions per persisted shard;
- roots per native search chunk;
- inference workers, maximum batch size, and outstanding batches;
- fixed label-search parallelism, initially one.

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
- raw prediction and calibrated-percentile distributions;
- Spearman rank correlation and calibration by prediction decile;
- realized raw KL and oracle utility by prediction decile;
- target top-1 agreement by candidate budget.

### Shadow allocator

- realized mean, variance, floor share, and ceiling share for every blend candidate;
- candidate divergence from deep policy and gain over flat;
- share of oracle gain when stable;
- confidence interval, trailing EMA, represented generations, and effective sample count;
- selected blend, prior blend, decision reason, and all failed safety conditions.

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
2. **Deep-label pipeline.** Add deterministic generation sampling and sharding, explicit native checkpoint-visit requests, immutable-checkpoint execution, bounded root chunks, baseline/candidate/deep policy capture, quantile normalization, persistent job state, and deep-policy replay write-back. Start with one reserved-GPU worker and label-search parallelism one.
3. **Shadow evaluator and calibrator.** Add exact candidate-budget reconstruction, equal-compute utility scoring, confidence calculation, EMA state, atomic publication, fail-closed behavior, and complete diagnostics.
4. **Native allocator in shadow-safe form.** Add predicted budget plumbing, mean-preserving generation allocation, retained-root accounting, and per-search parallelism. Ship with the published blend clamped to zero until calibrator conditions permit otherwise.
5. **Remove superseded systems.** Delete threshold adaptation, `SearchCorrectionGate`, `search_correction`, fast/full admission and configuration, old filters, obsolete tools, stale telemetry, and compatibility layers. Migrate production configuration and documentation in the same phase.
6. **Component and integration validation.** Exercise target masking, replay round trips, job retries, quantiles, exact mean preservation, checkpoint scoring, calibrator publication and fallback, retained roots, native stopping, and heterogeneous parallel searches.
7. **Live 4-8 GPU training run.** Accumulate the warm-up evidence, inspect the shadow distribution, freeze the concrete calibration constants before the first acting generation, and then allow the calibrator to advance or retreat the blend automatically.

Feature-sized commits should follow these boundaries so each stage is independently reviewable and reversible.

## Live experiment

The meaningful validation is an actual training run on representative self-play, not a larger frozen offline classifier exercise.

The first run should allocate 4-8 GPUs according to measured labeler and self-play throughput. It starts with approximately 30 generations at `alpha = 0`, while the head trains and the shadow evaluator accumulates evidence. At 2% labelling this is approximately 72,000 new unique labelled positions before activation becomes possible.

Before the first acting generation, inspect:

- label throughput and lag;
- prediction and target distributions;
- ordering by decile;
- exact equal-compute candidate gains and uncertainty;
- whether the proposed EMA and window react reasonably across generations;
- per-budget parallelism behavior;
- label-worker chunk size, inference batch size, memory, and generation lag;
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
- **Non-stationarity can outrun calibration.** The head, trunk, game distribution, and baseline search all evolve. Windows, EMA lag, and publication lag must be visible and conservative.
- **Retained roots can break compute accounting.** Allocation must govern new simulations and report starting visits separately.
- **The deepest policy is a reference, not truth.** Its own search noise and finite depth limit both labels and shadow utility estimates.
- **Quantile drift can distort the curve.** Prediction percentiles and target quantiles require explicit, generation-aware calibration populations.
- **Acting creates feedback.** Permanent random audit labelling is required even after the blend becomes nonzero.
- **Asynchronous jobs can train on stale labels.** Source model generation and completion lag must be preserved and bounded.
- **Deep replay samples change more than the scalar head.** Their stronger policy targets are desirable, but they mean the first live run measures the combined training system. The later matched-labeler A/B isolates allocator value.

## Settled decisions

- Use `KL(pi_deep || pi_baseline)` rather than total variation.
- Quantile-normalize the scalar target.
- Generate labels at approximately 8x baseline depth on a small random fraction of positions.
- Train one scalar auxiliary head jointly with the trunk and mask its loss to labelled samples.
- Keep an independent random audit stream permanently.
- Write final deep policies back into replay.
- Make every played position a replay sample and remove the fast/full split.
- Use a fixed nonlinear multiplier curve with a floor near 0.2x and ceiling of at least 4x.
- Preserve exact mean new-simulation spend by construction.
- Scale parallel searches per assigned budget.
- Hold the blend at zero for an initial live warm-up of approximately 30 generations.
- Choose and continuously update the blend from conservative equal-compute shadow utility, not loss or correlation.
- Increase blend cautiously, reduce it immediately, and fail closed to zero.
- Remove all old threshold, learned-gate, `search_correction`, and fast/full machinery.

## Decisions to make from live shadow data

- 1% versus 2% deep-label sampling for the first run;
- exact deep limit and retained checkpoint set;
- one versus multiple label-worker GPUs after measuring generation lag;
- label root-chunk and inference batch sizes;
- quantile and prediction-calibration windows;
- multiplier-curve control points and final ceiling;
- blend candidate grid;
- rolling evidence window and EMA half-life;
- confidence method, level, and safety margin;
- minimum positions, represented generations, and consecutive safe updates;
- maximum blend increase per generation;
- per-budget parallelism curve and worker limits;
- calibrator publication lag;
- GPU division, run length, and matched-control schedule.

## Acceptance criteria

Implementation is ready for the live experiment when:

- eligible and ineligible `search_budget` targets round-trip through replay and train with the correct mask;
- every self-play position materializes regardless of its assigned budget;
- deep-label jobs are generation-triggered, asynchronous, bounded, retryable, and idempotent;
- prediction is recorded before the deep target and remains attributable to an immutable source checkpoint;
- label shards have stable ownership and can be processed by one or multiple workers without duplicate evidence;
- label-search root chunks bound host memory while sustaining measured inference occupancy;
- one native continuation returns policies at every explicitly requested visit checkpoint;
- deep policies are written back as valid replay samples;
- shadow candidate budgets preserve exact mean new-simulation spend;
- shadow utility is reconstructed from exact policy checkpoints and reproduces known offline cases;
- the warm-up clamp prevents nonzero production blend before the configured generation;
- no nonzero blend is published without sufficient positive conservative evidence;
- unsafe, stale, missing, or incompatible evidence returns the allocator to zero;
- native search stops at the assigned additional-visit budget with retained roots;
- parallelism differs correctly between simultaneous searches with different budgets;
- old adaptive threshold, learned gate, `search_correction`, and fast/full code and configuration are gone;
- telemetry can explain every label, candidate score, blend transition, spend residual, and fallback;
- a 4-8 GPU live run completes warm-up and demonstrates whether the allocator can safely capture positive oracle gain.

Success after activation means matching or beating the flat-budget, matched-labeler control at equal wall-clock time while preserving mean search spend and improving mean policy-target agreement. The blend need not reach 1.0; a stable partial blend that captures reliable value is a successful outcome.

## Evidence and related artifacts

- [Chess search findings](../analysis/chess-search-findings-20260827.md)
- [RTX 3060 chess search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md)
- [Frozen-trunk adaptive budget probe](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md)
- `measure_policy_target_fidelity --per-position-output`
- `analyse_budget_allocation`
- `sample_chess_search_positions`
- `validate_adaptive_replay` (to be replaced or repurposed as part of the migration)
