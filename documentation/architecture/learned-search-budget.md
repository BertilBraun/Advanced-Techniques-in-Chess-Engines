# Learned adaptive search budget

The learned search-budget system is the only production self-play budget path. It replaces the randomized fast/full
split, threshold stopping, and the former `search_correction` gate. Native C++ remains the sole search
implementation; Python owns targets, durable background labelling, replay write-back, and blend publication.

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

## Fixed multiplier curve

The run-versioned `measured_oracle_600_v1` curve is a left-closed step function. Its upper quantile boundaries are

`1186/3000, 1570/3000, 1838/3000, 2048/3000, 2188/3000, 2347/3000, 2547/3000, 2699/3000, 2806/3000, 1`,

and the corresponding multipliers are

`750/3761, 1500/3761, 2250/3761, 3000/3761, 3750/3761, 4500/3761, 6000/3761, 9000/3761, 12000/3761, 18000/3761`.

The exact uniform mean is one, the floor is `750/3761`, the ceiling is `18000/3761`, and `2188/3000` of quantiles
are below baseline. A published blend interpolates exactly between multiplier one and this curve.

Production allocation happens after root expansion for every position. It assigns additional simulations, carries a
signed deterministic integer residual into later assignments, and preserves retained-root semantics. Per-request
parallelism is `min(16, next_power_of_two(ceil(assigned_additional_visits / 200)))`. Explicit evaluation and label
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
after complete prediction coverage is verified are candidate allocations normalized across the complete generation
sample. The deep-search phase uses the same shards, fresh roots, no Dirichlet noise, parallelism one, and sorted
policy checkpoints. Each root continues once through the baseline, the union of all candidate visits, and exactly
eight times the source generation's configured baseline. Artifacts and lightweight manifests are written atomically,
checksummed, retried at most three times, and rejected unless coverage is complete.

Generation finalization uses the documented `1e-6` policy-probability floor for finite KL. It computes
`KL(pi_deep || pi_baseline)` for every sampled position, assigns deterministic generation-wide mid-rank quantiles,
scores every candidate blend at exact flat mean spend, and writes every deep-labelled sample to replay. Replay
write-back uses a prepared transaction receipt so a retry cannot duplicate one generation's evidence. Calibration is
updated only after that receipt commits.

## Calibration and publication

Candidate blends are exactly `0.0, 0.1, ..., 1.0`. Each candidate's generation gain is the mean reduction in deep
policy KL relative to flat search. Its EMA initializes from the first finalized generation and then updates as
`0.8 * previous + 0.2 * current`. Blend remains zero through 29 complete label generations. Thereafter a nonzero
candidate requires strictly positive current and EMA gain and exact mean spend. The greatest EMA gain wins, ties go
to the lower blend, upward movement is limited to `0.1` per completed label generation, and decreases are immediate.
No eligible nonzero candidate publishes zero.

Publication names the first production generation that has not started. A running generation never changes blend.
Unfinished work retains the latest completed publication. Terminal job failure, invalid compute reconstruction, an
incompatible configuration hash, or unreadable state publishes zero. Candidate gains, EMA values, mean visits,
distributions, floor/ceiling shares, residual, decision reason, and failed eligibility conditions are persisted in
each final report. Correlation and calibration diagnostics never authorize activation.

Run state is under `search-budget-labels/` in the configured training save path. Replay cohort and write-back journals
are under `completed-games/`. These artifacts are part of restart correctness and must be preserved with the run.
