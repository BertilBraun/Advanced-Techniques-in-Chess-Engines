# Learned adaptive search budget

The learned search-budget system is the only production self-play budget path. It replaces the randomized fast/full
split, threshold stopping, and the former `search_correction` gate. Native C++ remains the sole search
implementation; Python owns targets, durable background labelling, replay write-back, live curve learning, and
atomic curve publication.

## Model and replay contract

Every applicable network has one scalar `search_budget` head. The inference artifact returns its raw logit and the
native runtime applies sigmoid once to obtain the predicted quantile in `[0, 1]`. Training uses masked plain L1 on
that bounded value, divided by eligible sample weight. A batch with no eligible labels contributes finite zero loss.
The v10 auxiliary loss weight is `0.2`.

Ordinary self-play materializes every played position at its configured sample weight. An ineligible target occupies
the typed replay slot until deep labelling replaces it with an eligible record containing the normalized target, raw
KL, prediction logit, bounded prediction, source generation, checkpoint generation, and inference-model SHA-256.
The strongest `8 * baseline` policy becomes an ordinary replay sample; replay capacity and sampling alone determine
its lifetime.

## Live multiplier curve

The production curve is learned continuously. It has ten fixed-width predicted-quantile buckets and one positive,
monotone nondecreasing multiplier per bucket. The arithmetic mean of the ten multipliers is exactly one. There is no
historical lookup table, permanent `0.2` floor, fixed ceiling, or prediction-CDF layer in the production path.

The clean initializer is the exact bucket average of

```text
m0(q) = 0.2 + 4.8 * q^5
```

For bucket `[a, b]` of width `0.1`, its initial multiplier is
`0.2 + 8 * (b^6 - a^6)`. This analytic initializer is monotone and has exact uniform mean one. Its low end is a
rough starting estimate only; subsequent generation updates may move every bucket. Historical measured allocations
remain evidence in the benchmark documentation, not constants or defaults in executable code.

Each finalized label generation measures local marginal policy fidelity around the current shadow curve. For each
bucket, the label search records policies at log-symmetric lower and upper probes, initially `1 / 1.1` and `1.1`
times the bucket multiplier, after generation-wide mean normalization and deterministic integer allocation. Its
generation statistic is the mean reduction in `KL(pi_deep || pi_checkpoint)` divided by the multiplier interval.
Each bucket's marginal-utility EMA initializes from its first nonempty generation and subsequently uses decay `0.2`.
Empty buckets retain their EMA.

The common compute price is the count-weighted mean marginal utility over nonempty buckets. Centered bucket
utilities produce bounded log updates: the largest absolute proposed log change is `log(1.1)`. A deterministic
isotonic projection makes the curve nondecreasing, a common scale factor restores arithmetic mean one, and
backtracking shrinks the update until every final bucket ratio is within `[1 / 1.1, 1.1]`. This is allocator curve
state, not a cross-generation position window.

Production allocation happens after root expansion for every position. It assigns additional simulations, carries a
signed deterministic integer residual into later assignments, and preserves retained-root semantics. Per-request
parallelism is `min(16, max(2, next_power_of_two(ceil(assigned_additional_visits / 200))))`. Explicit evaluation and label
budgets bypass the production allocator and cannot mutate its spend ledger.

## Deep-label lifecycle

Replay ingestion atomically journals immutable sealed-shard metadata into one source-generation cohort before the
source files are deleted. A committed training quantum finalizes that cohort and enqueues it using the checkpoint
that generated the positions. Restart recovery replays every finalized or still-open unacknowledged cohort older than
the active checkpoint. Enqueue and acknowledgement are idempotent.

The deterministic sample contains exactly `floor(0.02 * population_positions)` stable position identities, chosen
uniformly without replacement from the run seed and source generation. One logical source generation runs at a time
in source order. A queued generation more than two production generations behind is skipped. One run-lifetime
spawned process pool uses every unique trainer GPU ID; each persistent process is pinned to one GPU, owns one
inference worker, uses inference batches of 512 with two outstanding batches, and refreshes immutable checkpoints
monotonically.

The prediction phase persists raw logits and bounded predictions in shards of 512 positions plus a remainder. Only
after complete prediction coverage is verified does the coordinator normalize the flat, validation-candidate, and
local-probe budget vectors across the complete generation sample. The deep-search phase uses the same shards, fresh
roots, no Dirichlet noise, `parallel_searches = 2`, and sorted policy checkpoints. Each root continues once through
the baseline, the union of all required checkpoint visits, and exactly eight times the source generation's configured
baseline. Artifacts and lightweight manifests are written atomically, checksummed, retried at most three times, and
rejected unless coverage is complete.

Generation finalization uses the documented `1e-6` policy-probability floor for finite KL. It computes
`KL(pi_deep || pi_baseline)` for every sampled position, assigns deterministic generation-wide mid-rank quantiles,
updates all observed bucket statistics, scores the previously prepared publication candidate at exact flat mean
spend, and writes every deep-labelled sample to replay. Replay write-back uses a prepared transaction receipt so a
retry cannot duplicate one generation's evidence. Curve state changes only after that receipt commits.

## Validation and publication

A curve proposed from source generation `g` is first checkpointed and scored on source generation `g + 1`. This
one-generation delay prevents the same evidence from both constructing and authorizing a curve. The report records

```text
generation_gain = mean(KL(pi_deep || pi_flat) - KL(pi_deep || pi_candidate))
```

and a scalar validation-gain EMA that initializes from its first scored proposal and then updates as
`0.8 * previous + 0.2 * current`.

The published curve remains exactly flat through 29 complete source-generation jobs. Thereafter the pending curve is
eligible only when its current validation gain and validation-gain EMA are both strictly positive and its reconstructed
sample compute is valid. An eligible publication moves each bucket by at most ten percent from the previously
published curve and preserves monotonicity and exact mean. A failed eligibility test publishes the flat curve
immediately. There is no blend coefficient or grid of fixed historical curves.

Publication names the first production generation that has not started. A running generation never changes curve.
Unfinished work retains the latest completed publication. Terminal job failure, invalid compute reconstruction, an
incompatible configuration hash, or unreadable state publishes flat. Finalization and publication are idempotent.

TensorBoard receives label-pipeline health; target and prediction distributions; generation and EMA validation gain;
and, for every bucket, sample count, generation marginal utility, EMA marginal utility, shadow, pending, and
published multiplier, raw update, projection adjustment, and empty-bucket status. Reports also preserve exact spend
residual, curve lineage, min/max multipliers, decision reason, and failed eligibility conditions. Correlation,
calibration plots, regression loss, top-1 agreement, and oracle-gain share remain diagnostics only.

Run state is under `search-budget-labels/` in the configured training save path. Replay cohort and write-back journals
are under `completed-games/`. Persisted curve calibration contains only generation aggregates and state lineage, not
a cross-generation collection of labelled positions.

Artifact retention is an explicit run policy. `retain_all` keeps the complete generation source plus every prediction
and policy-checkpoint shard for diagnostic smokes. Production uses `remove_bulky_after_finalization`. Cleanup starts
only after the replay write-back receipt is committed, calibration state and final report are atomic, and the manager
state durably marks the generation complete. It removes `source.json` and shard `artifact-attempt-*.json` payloads,
but retains every shard manifest, final report, calibration state, manager state, replay write-back journal,
TensorBoard event, log, failed-job artifact, and a compact `artifact-cleanup-receipt.json` recording exact removed
paths and bytes plus the preserved evidence references. A prepared receipt makes partial deletion retryable and a
cleanup failure never reclassifies or re-finalizes an already completed generation.
