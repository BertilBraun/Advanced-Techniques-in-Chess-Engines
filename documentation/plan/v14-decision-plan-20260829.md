# V14 decision and execution plan

**Status:** proposed; V13 remains the only authorised live run

## Decision gate

Do not start V14 automatically. Review V13 at ten accumulated hours, preserve and fetch a fresh archive, and choose
one of these outcomes:

1. **Continue V13** when the allocator delivers stable positive equal-compute KL gain, wall-clock evaluations are
   competitive with V9, production spend is conserved, and the live curve remains operationally valid.
2. **Build V14 with adaptive production enabled** when the within-run gain is convincing but V13 is held back by
   known persistence, publication, or spend-accounting defects.
3. **Stop after preservation and reassess offline** when the adaptive evidence is weak or negative. Do not spend GPU
   time on a nominal V14 whose only change is disabling a feature that has not justified its complexity.

The adaptive signal is convincing when the final ten-hour window has positive current gain on at least eight of the
last ten completed label jobs, positive EMA gain throughout, EMA gain at least `0.01` KL, healthy non-collapsed head
predictions, and no evaluation regression that outweighs the allocation benefit. Gain is an absolute KL improvement,
not a percentage. Report it both directly and as a fraction of the flat candidate's mean KL from deep search.

## Current V13 evidence

The fetched point-in-time archive at generation 56 shows:

- current validation gain `0.03473` KL and EMA gain `0.03580` KL;
- flat-candidate mean KL from deep search `0.24498`, making the measured gain about 14% of that error scale;
- prediction mean `0.53065`, standard deviation about `0.2164`, range `0.0567` to `0.8856`, and no saturated tail;
- exact `floor(0.5%)` sampling: 602 of 120,438 positions;
- exact candidate mean spend of 500 visits with zero candidate residual;
- a positive monotone published curve from `0.2662` to `2.5952`, applied at generation 59;
- replay write-back of all 602 selected samples, no shard retries, and completion lag two.

This is a real algorithmic signal, but it is not yet sufficient causal evidence for a four-day commitment. V13 also
contains a historical stale publication, a flat retreat larger than the trust bound, and a generation-47 production
mean-spend discrepancy. Its early generation cadence is roughly 22–27% slower than comparable V9 generations.

## V14 work packages

### 1. Make cohort persistence asynchronous and batched

Keep the compact replay-backed locator representation. Move durability work out of the replay ingestion critical
section: write immutable locator shards without a synchronous central-journal rewrite per sealed source shard, then
atomically publish one generation/cohort manifest at the training boundary. Acknowledge completed cohorts with an
append-only or atomically replaced compact index outside the replay lock. Preserve crash recovery, idempotency, and
the invariant that a sealed game is never deleted before its locator is durable.

Acceptance requires a replay-ingestion benchmark against V9-sized cohorts, bounded inbox/staging depth, crash tests
at every durability transition, and no full replay payload in cohort artifacts. The target is to remove most of the
observed 12-second ingestion delta without weakening recovery.

### 2. Fix publication ownership and the trust bound

Reserve the application generation from authoritative coordinator generation-start state, not from a lagging label
manager callback. Publication must be a compare-and-publish operation: if the nominated generation has started,
advance to the next unstarted generation before writing state.

Apply the ten-percent multiplicative bound to every production transition, including retreat after failed gain. A
failed validation should move toward flat through the same bounded projection instead of jumping directly to flat.
Add concurrency tests where labelling completes while a production generation starts.

### 3. Prove exact production spend conservation

Reproduce the generation-47 discrepancy from persisted telemetry before changing the allocator. Record, per worker
and generation, baseline total, assigned total, position count, starting and ending residual, curve lineage, and
clamp count. The invariant is:

```text
assigned_total - baseline_total = ending_residual - starting_residual
```

The generation aggregate must be the sum of worker ledgers over exactly the same observations. Investigate generation
boundary resets, persistent games crossing schedule changes, telemetry population mismatch, and clipping separately.
Do not replace the online allocator with a generation-wide prediction pass.

### 4. Preserve causal observability

Retain the 0.5% label fraction, 8x deep limit, replay ratio eight, V9 visit schedule, five-million replay capacity,
512 games per worker, and `2/2/4/8/16` parallel mapping unless the ten-hour evidence identifies one as a direct
failure. Changing them together would erase the V13 comparison.

Add a shadow flat-policy counterfactual to each evaluation boundary: use the existing label sample's equal-compute
flat and published-policy KL measurements and report their confidence interval. If a future matched control is
needed, author it as a separate approved run rather than silently mixing it into production.

### 5. Preserve before any node release

Run checkpoint-safe stop only after user approval. After terminal state, run `preserve` and `fetch`, verify every
SHA-256 manifest, and confirm the newest V13 archive contains TensorBoard, logs, configurations, evaluations, latest
model/optimizer weights, all compact search-budget reports, calibration state, cohort locators, and replay write-back
receipts. Full replay and completed-game payloads are intentionally disposable.

## Validation before a V14 launch

- focused Python tests for replay cohort crash recovery, label-manager publication races, calibration transitions,
  and coordinator spend aggregation;
- native allocator tests covering multiple workers, generation resets, schedule boundaries, clipping, and long
  adversarial quantile sequences;
- `ruff format`, `ruff check --fix`, and the repository test suite with `--import-mode=importlib`;
- a bounded local or rented-node smoke only after the user accepts this plan and authorises the next phase;
- a complete standalone V14 configuration and fresh revision/configuration-bound approval.

## Local preservation state

As of 2026-08-29, SHA-256-verified workstation archives exist under `.codex-diagnostics` for terminal V9, V10,
V11, V12, and a live V13 snapshot. V9 includes both progressive stages' latest weights. V10 and V11 retain useful
performance and failed-design evidence; V12 is a short transition run without retained model weights. The enhanced
archive also retains compact learned-budget evidence without copying replay, restart states, or live game payloads.
