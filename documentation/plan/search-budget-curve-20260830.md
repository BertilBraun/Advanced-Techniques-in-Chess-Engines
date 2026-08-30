# Predicted-curve adaptive search budget

**Status:** implemented on branch `search-budget-curve`, 2026-08-30. Supersedes
`adaptive-search-budget-20260827.md` (the scalar-quantile head with the live bucket multiplier curve), which
production measurement (v14 vs v15, v16) showed to be ineffective at ~12% GPU cost.

## Design

The head predicts the whole marginal-return curve instead of a scalar: for each position, ten values

```text
curve[k] = log(KL(pi_deep || pi_{b_k}) + 1e-6),   b_k = max(1, round(BUDGET_CURVE_MULTIPLES[k] * B))
BUDGET_CURVE_MULTIPLES = (0.125, 0.2, 1/3, 0.5, 2/3, 1.0, 1.5, 2.0, 3.0, 4.0)
```

where `pi_deep` is the deep-label policy at `8 * B` and `pi_{b_k}` the same search's policy checkpoint at `b_k`.
The deep search checkpoints exactly the grid (deduplicated after rounding), so labelling adds no search cost beyond
the deep reference. The head is ten-wide, trains with masked Huber loss (weight 0.2) on ordinary batches, and its
targets live in replay as a ten-wide `AUXILIARY_VALUE` column following the next-policy precedent.

## Allocation (native, per self-play position)

1. Isotonic projection of the predicted curve: running minimum from the largest budget downward,
   `yhat[k] = min(yhat[k], yhat[k+1])` for `k = 8..0`.
2. `P[k] = Phi((log_tau - yhat[k]) / sigma[k])` with the standard normal CDF.
3. Select the lowest `k` with `P[k] > theta` (theta 0.8); the deepest point when none qualifies.
4. Assigned visits `round(multiple_k * B)` with the running spend-error correction, clamped to `[1, 8B]`.

`sigma[k]` is a per-grid-point EMA (decay 0.1, initialised 1.0) of the labelled `|yhat[k] - curve[k]|`.
`log_tau` is a dual variable stepped by `clamp(log(realized_mean_multiple), +-log(1.05))` per finalized label
generation so realized mean spend tracks the flat baseline. Both persist in the calibration state and cross the
binding seam inside `SearchBudgetPolicy` (grid multiples, sigma, log_tau, theta, apply_learned).

## Safety gate

Unchanged in spirit: shadow gain compares the policy at the learned assignment against the policy at flat budget,
both scored against the deep policy, on every finalized label generation. The learned rule is applied in production
only after 30 completed source generations with strictly positive current and EMA gain; any failure, unreadable or
incompatible state publishes `apply_learned = false` (flat) to the next unstarted generation.

## Analysis log

Per labelled position, one fixed-width numpy record (`analysis-generation-XXXXXXXX.np` under
`search-budget-labels/`): identity, baseline, raw `policy_kl[10]`, `value_error[10]` (per-checkpoint root values are
recorded natively), baseline top-visit share and entropy, `predicted_curve[10]`, `deep_half_kl`, assigned visits and
selected index. Append-only; failures are logged and never affect the label job or training.

## Removed

The bucket multiplier curve and everything that existed only to learn it: analytic initializer, local probes,
per-bucket marginal-utility EMAs, probe allocation purposes, shadow/pending/published curve lineages with delayed
validation, the exact-mean generation allocator, `allocate_next_production_budget`, the native `SearchBudgetCurve`
type, `probe_ratio` / `bucket_utility_ema_decay` / `initializer_version` / `maximum_step_ratio` configuration, and
their telemetry and dashboards. Configurations v11-v17 carry the new `calibration` schema
(`sigma_ema_decay`, `initial_tau`, `tau_step_ratio`, `selection_threshold`, warmup, gain-EMA decay).

## Evidence

- `documentation/plan/chess-search-followup-plan-20260827.md`, WP-S2b: remaining-error ordering captures 54.3% of
  oracle gain; movement-based ordering ~0%; a weak head is worse than flat; dynamic range beats level count; a
  single scalar caps at 54% because the shape of the marginal-return curve matters.
- v14 (curve enabled) matched v15 (disabled) at every evaluation boundary; v16 tuning changed nothing material.
