# Learned adaptive search budget with live curve calibration

**Status:** implementation plan, revised 2026-08-28

## Decision

Replace both the threshold-based adaptive full-search mechanism and the fast/full search split with one learned
per-position search budget. The network predicts a bounded search-difficulty quantile from the root forward pass. A
live ten-bucket monotone curve maps that quantile to a visit multiplier while deterministic allocation preserves the
baseline's cumulative mean spend.

The production curve is learned continuously from the random deep-label stream. It is not the historical measured
oracle curve, is not refit from a retained cross-generation position sample, and has no fixed `0.2` floor. The
historical measurements remain useful evidence in the linked analysis and benchmark documents, but their exact
sample boundaries and ratios must not appear as production constants.

For the first 30 completed source-generation label jobs, the head and shadow curve learn while production remains
flat. A curve proposed from one generation is evaluated at equal mean compute on the next completed label generation
before it can be published. Publication requires strictly positive current and EMA validation gain. Ineligible or
invalid state falls back to a flat curve.

## Why the existing mechanism must be removed

The threshold rule has the wrong functional form. Marginal benefit is inverted-U-shaped in its top-visit-share
signal, and the historical threshold schedules operate where selection is worse than random. The former
`search_correction` gate was never trained in v9. The fast/full split also discards training value by filtering every
fast observation even though those searches consume substantial compute.

The replacement materializes every played position, trains one explicitly masked `search_budget` target, and assigns
additional simulations without changing sample weight. A low assigned budget means the policy was predicted to
settle cheaply; it does not make the resulting replay observation ineligible.

## Evidence and its limits

The offline measurements establish that per-position allocation can be valuable, that
`KL(pi_deep || pi_baseline)` is a useful ranking target, and that labels at eight times baseline retain most of the
available sixteen-times ranking signal. They also show that excessive dispersion can be harmful when the ordering is
wrong. They do not establish a timeless production curve.

The measured RTX 3060 oracle allocation and frozen-trunk probe remain documented in the evidence artifacts at the
end of this plan. They motivate a low, convex analytic initializer and a conservative warm-up, but the executable
system must learn its bucket multipliers from the current run's models, game distribution, and baseline schedule.
The roughly one-fifth low-budget ratio is an initialization estimate, not an invariant, clamp, or promised minimum.

## Per-generation lifecycle

The system operates asynchronously:

1. Self-play generation `g` uses the curve published before that generation starts. During warm-up this is flat.
2. Training publishes an immutable checkpoint for `g`.
3. A separate background job samples positions uniformly from generation `g`'s new replay observations.
4. Prediction shards persist raw logits and bounded predictions before any deep result is known.
5. After complete prediction coverage, the coordinator constructs sample-wide flat, pending-validation, and local
   bucket-probe budget vectors with exact mean preservation and records each position's union of checkpoints.
6. Deep-search workers search each fresh root once through the baseline, every required checkpoint, and exactly
   `8 * B(g)` while retaining policy snapshots.
7. Workers atomically persist policy-heavy shard artifacts. The coordinator retries failures and rejects partial
   coverage or checksum mismatch.
8. One generation-wide finalizer calculates every raw KL, assigns deterministic mid-rank quantile targets, scores
   the pending curve, and computes per-bucket marginal-utility aggregates.
9. The strongest deep policy and eligible scalar target are written back as ordinary replay records.
10. Only after replay write-back commits does the controller update EMA state, prepare the next pending curve, and
    atomically publish the validated state for the next production generation that has not started.
11. A completed manifest makes replay write-back, EMA updates, proposal construction, and publication idempotent.
12. Training and self-play never wait for labelling. One logical source generation runs at a time in source order,
    and an unstarted job more than two production generations late is skipped.

The labeler is generation-triggered and separate from the wall-clock evaluation manager.

## Label generation

### Sampling and target

For source generation `g`, select exactly

```text
floor(0.02 * new_generation_position_count)
```

stable position identities uniformly without replacement using the run seed and source generation. Sampling is
independent of predicted search budget. Every completed sampled position participates in that generation's target
and curve statistics.

The final deep limit is exactly `8 * B(g)`, where `B(g)` is the source generation's configured baseline. v10 inherits
the v9 schedule: 200 visits at generation 0, 300 at 10, 400 at 30, 500 at 50, 600 at 90, 700 at 180, 800 at 250,
and 1,000 at 550.

For every sampled position:

```text
raw_difficulty = KL(pi_deep || pi_baseline)
```

After all shards complete, rank all generation KL values together. Use deterministic mid-ranks for ties, map the
lowest and highest non-tied values to zero and one, and use `0.5` for a degenerate one-position or all-tied sample.
Persist raw KL and normalized `search_budget_target`. There is no cross-generation quantile window.

### Worker topology and checkpoints

Use one run-lifetime spawn-based `ProcessPoolExecutor`. All unique configured trainer GPU IDs are eligible. Each
persistent process is pinned to one GPU, owns one inference worker, and refreshes immutable checkpoints only
monotonically. Persist deterministic execution shards of 512 positions plus one final remainder. Each inference
worker uses batch size 512 with two outstanding batches.

Deep labels use `parallel_searches = 2`. Two simultaneous descents per labelled root are required to keep both
outstanding inference batches fillable. Workers reconstruct fresh roots, disable Dirichlet noise, and accept a sorted
per-request set of absolute policy-checkpoint visit limits. Native search continues one tree through all requested
checkpoints and the final deep limit. `SearchCheckpointDetail::Policies` remains part of the contract.

Prediction and deep-search phases use the same deterministic shard ownership. Normalization is always across the
complete generation sample, never independently inside a 512-position execution shard. Policy-heavy artifacts are
atomic and checksummed; worker results are lightweight typed manifests with identity, device, counts, timings,
checksum, and path. Retry failed shards, verify complete coverage, and never finalize a partial generation.

### Replay write-back

The final deep policy is the strongest target produced by the job. Insert every completed deep-labelled sample into
ordinary replay with the eligible scalar target and model-lineage metadata. Normal replay capacity and sampling
control its lifetime. Duplicates from distinct observations are acceptable, but retrying the same generation job
must not duplicate replay evidence. Curve state changes only after transactional replay write-back succeeds.

## Model and loss contract

Replace `search_correction` end to end with the precisely named `search_budget` concept. Every applicable network
configuration has one unconstrained scalar logit. Apply sigmoid once and use that bounded output directly as the
predicted quantile; do not add a prediction-CDF calibration layer.

Carry `search_budget_target` and its eligibility mask through targets, materialization, columnar replay, and storage,
following `IneligibleNextPolicyTarget`. Train with masked plain L1, normalized by eligible sample weight, at auxiliary
loss weight `0.2`. A batch with zero eligible labels contributes finite zero search-budget loss. Preserve raw KL,
normalized target, raw prediction, bounded prediction, source generation, checkpoint generation, and model SHA-256.

## Live ten-bucket curve

### Representation and initializer

Use ten fixed quantile buckets:

```text
[0.0, 0.1), [0.1, 0.2), ..., [0.8, 0.9), [0.9, 1.0]
```

Each bucket owns one positive multiplier. The vector is monotone nondecreasing and its arithmetic mean is exactly
one, which is the uniform-quantile integral of the step curve.

Initialize from the clean analytic function

```text
m0(q) = 0.2 + 4.8 * q^5
```

Use the exact average over each bucket. For bucket `[a, b]` of width `0.1`:

```text
initial_multiplier(a, b) = 0.2 + 8 * (b^6 - a^6)
```

The ten averages have exact arithmetic mean one. This initializer qualitatively provides many inexpensive searches
and a small expensive tail without embedding historical sample counts. Neither its low end nor its high end is a
runtime bound. Minimum assigned additional visits is one; local probes and shadow candidates cannot exceed the deep
reference limit of `8 * B(g)`.

### Per-bucket evidence

Let the current shadow multiplier for bucket `k` be `m_k`. Before searching generation `g`, construct lower and upper
local probes at `m_k / 1.1` and `1.1 * m_k`. Convert the complete lower and upper vectors to concrete visits using the
same generation-wide exact-mean allocator used for all shadow candidates. Checkpoint de-duplication is allowed after
integer allocation.

For each position in bucket `k`, measure the normalized marginal fidelity benefit:

```text
u_i = (
    KL(pi_deep_i || pi_lower_i)
    - KL(pi_deep_i || pi_upper_i)
) / (upper_multiplier_i - lower_multiplier_i)
```

The generation statistic `u[g,k]` is the arithmetic mean across the bucket. A nonempty bucket initializes its EMA
from its first observation; later observations use:

```text
ema_u[g,k] = 0.8 * ema_u[g-1,k] + 0.2 * u[g,k]
```

An empty bucket retains its prior EMA and reports no generation statistic. Persist only counts and aggregate state;
do not retain a cross-generation collection of labelled positions for curve calibration.

### Curve update

Compute the common compute price as the position-count-weighted mean of current EMA utilities across nonempty
buckets. Center each available bucket EMA on that price. Divide by the largest absolute centered value so its update
signal lies in `[-1, 1]`; if all centered values are zero, propose no update. The raw log update is:

```text
delta_k = log(1.1) * normalized_centered_utility_k
```

Buckets above the common price gain multiplier and buckets below it lose multiplier. Empty buckets have zero raw
update. Apply deterministic equal-weight isotonic regression in log space, then multiply all buckets by one common
factor to restore arithmetic mean one. If projection or normalization makes any final bucket ratio leave
`[1 / 1.1, 1.1]`, halve every raw update and repeat until the bound holds. This produces the next shadow curve.

The `0.2` EMA coefficient, ten bucket boundaries, ten-percent probe ratio, ten-percent maximum multiplicative update,
projection method, and analytic initializer are versioned run configuration. Curve values themselves are durable
calibrator state.

## Delayed validation and publication

The system maintains three explicit curve lineages:

- `shadow`: the latest curve learned from bucket EMA state;
- `pending`: a publication candidate prepared after a completed generation;
- `published`: the curve assigned to production generations.

A proposal derived from generation `g` cannot be scored on `g`, because its exact checkpoint budgets were not known
before that generation's deep search. The pending proposal is therefore checkpointed and scored on generation
`g + 1`. This generation delay is the out-of-sample guard; do not split the 2% sample or reuse the construction
generation to authorize its own proposal.

The pending curve is the deterministic projection of `shadow` reachable from the current published curve with at
most a ten-percent multiplicative change per bucket and exact mean one. Its validation metric is:

```text
generation_gain[g] = mean_i(
    KL(pi_deep_i || pi_flat_i)
    - KL(pi_deep_i || pi_pending_i)
)
```

Initialize the scalar validation-gain EMA from the first scored pending curve, then update it with decay `0.2`.

Keep the published curve exactly flat until 30 complete source-generation label jobs have finalized. Afterwards,
publish the scored pending curve only when:

- current generation gain is strictly positive;
- validation-gain EMA is strictly positive;
- exact sample-wide compute reconstruction is valid;
- configuration and model lineage are compatible.

An eligible publication already satisfies the per-bucket ten-percent upward or downward step bound. If no nonflat
proposal is eligible, publish flat immediately. An unfinished job retains the latest completed publication. Terminal
failure, unreadable state, incompatible state, or invalid compute reconstruction publishes flat. Publication applies
only to the next production generation that has not started. Reprocessing a source generation must not duplicate
evidence, EMA changes, replay write-back, pending proposals, or publication transitions.

## Production allocation

After root expansion, map the bounded prediction directly to its published bucket multiplier. The assigned budget is
the number of additional simulations, not an absolute retained-root visit limit. Use deterministic integer rounding
and a signed cumulative residual ledger so prediction miscalibration, bucket occupancy, and rounding cannot change
cumulative mean spend. Do not run a generation-global production prediction pass. Residual is allocator state, not a
calibration parameter.

Production allocation applies to every self-play position. For assigned additional visits `V`:

```text
parallel_searches = min(16, next_power_of_two(ceil(V / 200)))
```

This maps 100 to 1, 300 to 2, 600 to 4, 1,600 to 8, and 2,400 or more to 16. Never exceed 16. Simultaneous roots may
have different budgets and parallelism. Explicit label and evaluation searches bypass production curve and residual
state.

## Configuration ownership

Canonical typed configuration owns these resolved defaults:

- sample fraction `0.02`;
- deep multiple `8`;
- maximum unstarted source-generation lag `2`;
- all unique trainer GPU IDs eligible for labelling;
- shard size and inference batch size `512`;
- two outstanding inference batches and label `parallel_searches = 2`;
- masked L1 auxiliary weight `0.2`;
- ten equal-width quantile buckets;
- analytic initializer `0.2 + 4.8 * q^5`, stored by semantic version rather than historical table name;
- bucket-utility EMA decay `0.2`;
- validation-gain EMA decay `0.2`;
- local probe and maximum per-generation multiplicative step `1.1`;
- warm-up `30` complete source-generation jobs;
- production parallelism target 200 sequential rounds and cap 16.

Transport configuration composes this canonical type rather than mirroring it. Important values are explicit and
validated. v10 resolves every default and inherits the v9 baseline schedule.

## Required TensorBoard and persisted telemetry

The label pipeline must report selected, started, completed, failed, retried, and skipped jobs; source generation;
lag; queue depth; device timings; coverage; checksums; raw KL and target distributions; replay write counts; and
idempotency outcomes.

Head telemetry includes eligible count, masked loss, gradient contribution, raw-logit and bounded-prediction
distributions, rank correlation, calibration by prediction decile, top-1 agreement, and oracle-gain diagnostics.

For every bucket, write:

- sample count and empty flag;
- generation marginal utility and EMA marginal utility;
- shadow, pending, and published multiplier;
- raw log update and projection adjustment;
- lower and upper probe visits and checkpoint de-duplication count.

Curve-level telemetry includes current and EMA validation gain, exact candidate mean visits, exact published mean,
minimum and maximum multiplier, previous and selected lineage, application generation, exact spend residual, update
and publication reason, and every failed eligibility condition. Correlation, plots, loss, top-1 agreement, and
oracle-gain share are diagnostics only and cannot authorize production publication.

## Removal and migration scope

Delete rather than disable:

- `AdaptiveFullSearchBudgetConfiguration` and threshold-driven adaptive limits;
- `minimum_visits`, `observation_interval`, `leader_stability_window`, `root_value_tolerance`, both top-visit-share
  schedules, `threshold_relaxation_visits`, and learned-gate paths;
- `SearchCorrectionGate`, `minimum_search_correction_to_unlock_tail`, and every `search_correction` head, target,
  binding, result, and telemetry field;
- `full_search_probability`, fast-search limits, `force_fast_search_after_ply`, full-search flags, admission staging,
  counts, filters, and telemetry;
- obsolete calibration tools and compatibility layers for those removed systems;
- the assumption that one `parallel_searches` value applies to every simultaneous search;
- the fixed historical multiplier table, its curve identifier, and blend-grid machinery.

Retain unrelated search behavior and `SearchCheckpointDetail::Policies`. Do not add Python MCTS or another search
implementation.

## Implementation sequence

1. Add the model head, target, mask, replay schema, loss, metrics, and bindings.
2. Add native checkpoint continuation and heterogeneous per-search budget support.
3. Add deterministic sampling and two-phase persistent deep-label workers.
4. Add generation-wide quantiles and idempotent deep-policy replay write-back.
5. Add live bucket evidence, EMA state, projection, delayed validation, atomic publication, and TensorBoard metrics.
6. Add production residual allocation, retained-root accounting, and per-budget parallelism.
7. Remove every superseded adaptive, `search_correction`, fast/full, historical-curve, and blend-grid path.
8. Update the master-side v10 configuration with all resolved defaults.
9. Complete migration, integration tests, and documentation.

Commit each coherent phase after focused validation; do not squash feature commits.

## Acceptance criteria

Implementation is ready for the live experiment when tests demonstrate:

- eligible and ineligible replay round trips, zero eligible labels, and masked L1 normalization;
- exact deterministic 2% sampling, generation-wide mid-rank ties, and degenerate samples;
- exact `8 * B(g)` deep limits across schedule changes;
- shard retry, checksum and coverage verification, and partial-generation rejection;
- atomic, idempotent replay finalization and one-time curve-state transition;
- exact analytic initializer bucket averages, monotonicity, and mean one;
- no historical curve constants or hard one-fifth floor in executable code;
- bucket EMA initialization, update, and empty-bucket behavior;
- deterministic centered marginal-utility updates, isotonic projection, backtracking, positivity, monotonicity, exact
  mean, and ten-percent step bound;
- one-generation proposal-validation lag and no same-generation authorization;
- exact shadow mean preservation over the complete sample, never per shard;
- 30-generation flat warm-up, strictly positive current and EMA gain, immediate flat retreat, zero fallback, and
  next-unstarted-generation publication;
- TensorBoard coverage for pipeline, per-bucket learning, validation, publication, residual, and failure state;
- retained-root additional-visit accounting and strict integer residual bounds;
- simultaneous heterogeneous budgets and the documented production parallelism mapping with cap 16;
- configuration default resolution and migration from fixed-curve/blend state;
- complete removal of fast/full, threshold-adaptive, `search_correction`, historical production curve, and blend-grid
  behavior.

No implementation validation may start, stop, inspect, or reconfigure a local or remote training run, connect to the
active v9 node, spend GPU time, or launch the v10 experiment. The user launches the evaluation run after review.

## Risks to watch

- Marginal-utility probes may be noisy; bucket counts, raw statistics, EMA, and projection must stay visible.
- Quantile occupancy can drift; exact residual accounting must not assume calibrated or uniform predictions.
- A ten-percent step may still be too fast or slow; it is versioned and observable, not silently adaptive.
- Native parallel selection may change fidelity at fixed visits; report sequential rounds and batch occupancy.
- The deepest policy is a finite reference rather than truth.
- Acting changes future data, so independent random labelling continues permanently.
- Deep replay write-back changes more than allocator quality; a matched-labeler flat control is needed for causal A/B.

## Settled decisions

- Use `KL(pi_deep || pi_baseline)`, generation-wide mid-rank quantiles, a sigmoid output, and masked L1 weight `0.2`.
- Sample exactly 2% independently of prediction and search exactly eight times the source baseline.
- Use persistent spawned GPU-pinned workers, 512-position shards, batch size 512, two outstanding batches, and label
  parallelism two.
- Preserve final deep policies in ordinary replay and retain no cross-generation calibration sample window.
- Initialize a clean analytic ten-bucket curve and learn bucket multipliers continuously from marginal-utility EMAs.
- Preserve positivity, monotonicity, exact uniform mean, and a ten-percent per-generation multiplicative trust bound.
- Validate each pending curve on the following source generation before publication.
- Keep production flat for 30 completed label generations and require positive current and EMA validation gain.
- Publish only to the next unstarted generation; fail closed to flat.
- Apply production budgets to every position as additional visits with residual mean preservation and heterogeneous
  per-budget parallelism.
- Remove old adaptive, gate, `search_correction`, fast/full, fixed historical curve, and blend-grid code.

## Decisions to make from live data

- Run length and matched-control schedule.
- Whether later runs should change bucket count, EMA decay, probe ratio, or trust bound. Those are new versioned run
  configurations, never silent mutations of an active run.

## Evidence and related artifacts

- [Chess search findings](../analysis/chess-search-findings-20260827.md)
- [RTX 3060 chess search evaluation](../benchmarks/chess-search-evaluation-rtx3060-20260826/README.md)
- [Frozen-trunk adaptive budget probe](../benchmarks/adaptive-search-budget-probe-rtx4070super-20260827/README.md)
- `measure_policy_target_fidelity --per-position-output`
- `analyse_budget_allocation`
- `sample_chess_search_positions`
