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

Lagrangian selection (2026-08-31, replacing the threshold rule, which offline replay on 220,814 v17
positions measured at -6.7% of oracle gain versus +12.5% for the Lagrangian and +21.3% with a linear
calibrator):

1. Linear calibration of the predicted curve using cheap root observables:
   `yhat_cal[k] = yhat[k] + b[k] + w[k] . f`, features
   `f = (yhat[k], top_visit_share, policy_entropy, ply, baseline_visits)`. Coefficients are ridge-fitted
   Python-side per label generation on a trailing window of analysis records (default 10 generations),
   standardisation folded into the shipped values; zeros (identity) until the first fit, after a
   non-finite fit, or when the fit does not reduce the in-window residual. Independently disableable via
   `calibration.calibrator.enabled`.
2. Isotonic projection of the calibrated curve: running minimum from the cheapest budget upward.
3. `k* = argmin_k exp(yhat_cal_projected[k]) + lambda * multiples[k]`, ties to the cheapest k. Raw KL
   space because the run-level objective is a sum of KLs, not of logs.
4. Assigned visits `round(multiple_k * B)` with the running spend-error correction, clamped to `[1, 8B]`.

`lambda` is the dual variable holding realized mean spend at the flat baseline: seeded on the first
label generation by bisecting the value whose Lagrangian selection on that generation's measured curves
spends a mean multiple of 1.0, then stepped multiplicatively by `clamp(realized_mean_multiple,
1/lambda_step_ratio, lambda_step_ratio)` (ratio 1.05) per finalized label generation. `sigma[k]`
(per-grid-point EMA, decay 0.1, of `|yhat[k] - curve[k]|`) is no longer used for selection but stays
computed and logged as the prediction-quality diagnostic. Lambda, calibrator coefficients and sigma
persist in the calibration state; the binding seam inside `SearchBudgetPolicy` carries grid multiples,
lagrange_multiplier, calibration_bias[10], calibration_weights[10][5] and apply_learned. Native computes
top_visit_share and policy_entropy from the root's retained visit distribution (raw priors on a fresh
root) and receives ply through the search request; the Python-side shadow selection during label
finalization uses the baseline-policy features from the deep-search checkpoints, which is the basis the
calibrator is fitted on.

## Safety gate

Unchanged in spirit: shadow gain compares the policy at the learned assignment against the policy at flat budget,
both scored against the deep policy, on every finalized label generation. The learned rule is applied in production
only after 30 completed source generations with strictly positive current and EMA gain; any failure, unreadable or
incompatible state publishes `apply_learned = false` (flat) to the next unstarted generation.

## Analysis log

Per labelled position, one fixed-width numpy record (`analysis-generation-XXXXXXXX.np` under
`search-budget-labels/`): identity, baseline, raw `policy_kl[10]`, `value_error[10]` (per-checkpoint root values are
recorded natively), baseline top-visit share and entropy, `predicted_curve[10]`, `calibrated_curve[10]`,
`deep_half_kl`, assigned visits and selected index. Append-only; failures are logged and never affect the label job or training.

## Removed

The bucket multiplier curve and everything that existed only to learn it: analytic initializer, local probes,
per-bucket marginal-utility EMAs, probe allocation purposes, shadow/pending/published curve lineages with delayed
validation, the exact-mean generation allocator, `allocate_next_production_budget`, the native `SearchBudgetCurve`
type, `probe_ratio` / `bucket_utility_ema_decay` / `initializer_version` / `maximum_step_ratio` configuration, and
their telemetry and dashboards. Configurations v11-v18 carry the current `calibration` schema
(`sigma_ema_decay`, `lambda_step_ratio`, warmup, gain-EMA decay, `calibrator` block); `initial_tau`,
`tau_step_ratio` and `selection_threshold` were removed with the threshold rule.

## Evidence

- `documentation/plan/chess-search-followup-plan-20260827.md`, WP-S2b: remaining-error ordering captures 54.3% of
  oracle gain; movement-based ordering ~0%; a weak head is worse than flat; dynamic range beats level count; a
  single scalar caps at 54% because the shape of the marginal-return curve matters.
- v14 (curve enabled) matched v15 (disabled) at every evaluation boundary; v16 tuning changed nothing material.
