# Predicted-curve adaptive search budget

**Status:** implemented on branch `search-budget-curve`, 2026-08-30. Supersedes
`adaptive-search-budget-20260827.md` (the scalar-quantile head with the live bucket multiplier curve), which
production measurement (v14 vs v15, v16) showed to be ineffective at ~12% GPU cost.

## Design

The head predicts the whole marginal-return curve instead of a scalar: for each position, eight values

```text
curve[k] = log(KL(pi_deep || pi_{b_k}) + 1e-6),   b_k = max(1, round(BUDGET_CURVE_MULTIPLES[k] * B))
BUDGET_CURVE_MULTIPLES = (0.125, 0.2, 1/3, 0.5, 2/3, 1.0, 1.5, 2.0)
```

where `pi_deep` is the deep-label policy at `8 * B` and `pi_{b_k}` the same search's policy checkpoint at `b_k`.
The grid was narrowed from ten points (up to 4x) on 2026-08-31: on 739,027 v17 positions the Lagrangian rule
captures 20.1% of oracle gain on the 0.125-2x grid versus 13.7% on the full grid at matched spend, because the
deep end is where predictions are least reliable and same-tree label bias largest. The deep search checkpoints
the grid plus the half-deep reference `4 * B` (kept only for the `deep_half_kl` label-quality diagnostic),
deduplicated after rounding, so labelling adds no search cost beyond the deep reference. The head is eight-wide,
trains with masked Huber loss (weight 0.2) on ordinary batches, and its targets live in replay as an eight-wide
`AUXILIARY_VALUE` column following the next-policy precedent.

## Allocation (native, per self-play position)

Lagrangian selection (2026-08-31, replacing the threshold rule, which offline replay on 220,814 v17
positions measured at -6.7% of oracle gain versus +12.5% for the Lagrangian and +21.3% with a linear
calibrator):

1. Correction of the predicted curve by a TorchScript MLP (2026-08-31, replacing the per-point ridge
   calibrator, which out-of-fold measurement on 150,000 positions showed captures 20.8% of oracle gain on
   the narrowed grid versus 32.9% for one MLP predicting the whole curve jointly; a nonlinear model on the
   head's own output alone recovers nothing, so the corrector exists to inject root observables, not to add
   capacity). Inputs: the eight predicted values plus `top_visit_share`, `policy_entropy`, `ply`,
   `baseline_visits`, `source_generation`; two hidden layers of width 64 with ReLU; output is one additive
   log-KL correction per grid point. Standardisation (train-window mean/sd) is folded into the exported
   module, so evaluation is raw. Fitted Python-side per label generation on a trailing window of analysis
   records (default 10 generations) with SmoothL1 on `(true_log_kl - predicted_log_kl)`, Adam lr 1e-3,
   batch 4096, 30 epochs; exported with `torch.jit.script` to
   `search-budget-labels/corrector-generation-XXXXXXXX.jit.pt` and referenced by path + sha256 in the
   published policy. A fit with non-finite parameters or whose held-out residual (fixed-stride split) does
   not improve on the uncorrected residual is never published: the previous corrector (identity if none)
   stays referenced and the rejection is logged. Independently disableable via
   `calibration.corrector.enabled`. Native loads the module on the model-refresh cadence when the policy
   crosses the seam at generation start and evaluates it on CPU at root selection.
2. Isotonic projection of the calibrated curve: running minimum from the cheapest budget upward.
3. `k* = argmin_k exp(yhat_cal_projected[k]) + lambda * multiples[k]`, ties to the cheapest k. Raw KL
   space because the run-level objective is a sum of KLs, not of logs.
4. Assigned visits `round(multiple_k * B)` with the running spend-error correction, clamped to `[1, 8B]`.

`lambda` is the dual variable holding realized mean spend at the flat baseline: seeded on the first
label generation by bisecting the value whose Lagrangian selection on that generation's measured curves
spends a mean multiple of 1.0, then stepped multiplicatively by `clamp(realized_mean_multiple,
1/lambda_step_ratio, lambda_step_ratio)` (ratio 1.05) per finalized label generation. `sigma[k]`
(per-grid-point EMA, decay 0.1, of `|yhat[k] - curve[k]|`) is no longer used for selection but stays
computed and logged as the prediction-quality diagnostic. Lambda, the corrector reference and sigma
persist in the calibration state; the binding seam inside `SearchBudgetPolicy` carries grid multiples,
lagrange_multiplier, corrector_path ('' = identity) and apply_learned, and the predicted limit carries
the model generation the corrector sees as `source_generation`. Native computes top_visit_share and
policy_entropy from the root's retained visit distribution (raw priors on a fresh root) and receives ply
through the search request; no post-search information enters the decision. The Python-side shadow
selection during label finalization uses the baseline-policy features from the deep-search checkpoints,
which is the basis the corrector is fitted on; the raw-prior root-time feature values are logged in the
analysis record beside the post-search ones so the deployment gap stays measurable.

## Safety gate

Unchanged in spirit: shadow gain compares the policy at the learned assignment against the policy at flat budget,
both scored against the deep policy, on every finalized label generation. The learned rule is applied in production
only after 30 completed source generations with strictly positive current and EMA gain; any failure, unreadable or
incompatible state publishes `apply_learned = false` (flat) to the next unstarted generation.

## Analysis log

Per labelled position, one fixed-width numpy record (`analysis-generation-XXXXXXXX.np` under
`search-budget-labels/`): identity, baseline, raw `policy_kl[8]`, `value_error[8]` (per-checkpoint root values are
recorded natively), baseline top-visit share and entropy, the raw-prior root-time share and entropy actually
available at selection, `predicted_curve[8]`, `corrected_curve[8]`, `deep_half_kl`, assigned visits and selected
index. Append-only; failures are logged and never affect the label job or training.

## Removed

The bucket multiplier curve and everything that existed only to learn it: analytic initializer, local probes,
per-bucket marginal-utility EMAs, probe allocation purposes, shadow/pending/published curve lineages with delayed
validation, the exact-mean generation allocator, `allocate_next_production_budget`, the native `SearchBudgetCurve`
type, `probe_ratio` / `bucket_utility_ema_decay` / `initializer_version` / `maximum_step_ratio` configuration, and
their telemetry and dashboards. The ridge calibrator (`calibrator` block with `ridge_coefficient`) was
removed 2026-08-31 with the MLP corrector; configurations v11-v19 carry the current `calibration` schema
(`sigma_ema_decay`, `lambda_step_ratio`, warmup, gain-EMA decay, `corrector` block); `initial_tau`,
`tau_step_ratio` and `selection_threshold` were removed with the threshold rule.

## Evidence

- `documentation/plan/chess-search-followup-plan-20260827.md`, WP-S2b: remaining-error ordering captures 54.3% of
  oracle gain; movement-based ordering ~0%; a weak head is worse than flat; dynamic range beats level count; a
  single scalar caps at 54% because the shape of the marginal-return curve matters.
- v14 (curve enabled) matched v15 (disabled) at every evaluation boundary; v16 tuning changed nothing material.
