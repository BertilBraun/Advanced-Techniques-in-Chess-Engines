# Predicted per-position search budgets do not convert to Elo

2026-09-01. Runs v16–v20 on the 8×RTX 4070 SUPER node (`38.49.42.120`).

Five production runs were spent on per-position adaptive search budgeting. The final implementation
meets every target it was designed to meet and is **worse on the ladder than runs without it**. This
note records what was predicted, what was built, what was measured, and the most likely reason, so the
result is citable and so the approach is not attempted again in this form.

## What was built

A network head predicts, per position, a curve of `log(KL(pi_deep || pi_at_budget_k) + 1e-6)` over
eight budget multiples of the baseline visit count:

    BUDGET_CURVE_MULTIPLES = (0.125, 0.2, 1/3, 0.5, 2/3, 1.0, 1.5, 2.0)

Ground truth comes from deep searches at 8× baseline on a `sample_fraction` of self-play positions.
A Lagrangian rule selects the budget:

    k* = argmin_k ( exp(yhat[k]) + lambda * multiple[k] )

with one dual `lambda` per generation holding mean spend at 1.0. A TorchScript MLP corrector
(2×64, 10 joint outputs) adds a correction to the predicted curve from features the trunk cannot see.
A four-condition gate (warm-up, current gain > 0, EMA gain > 0, mean spend within band) decides
whether the learned rule is applied or self-play falls back to flat allocation.

## What was predicted

From an offline study on ~740k labelled positions from v17 (recorded in the project memory
`az-adaptive-search-budget-findings`):

| claim | predicted |
| --- | --- |
| Lagrangian rule beats threshold rule | −6.7% → +12.5% of oracle gain |
| narrowing the grid 0.125–4× → 0.125–2× | capture 13.7% → 20.1% |
| MLP corrector vs none | capture 18.9% → 32.9% |
| oracle ceiling | 2.4–2.8× effective compute |

The study was explicit that conversion to Elo was unproven, and that the idea should be retired only
on Elo evidence rather than on the KL proxy.

## What was measured

v20, generations 260–268, 6,351 labelled positions, gate applying on 19–20 of every 20 generations.

    flat                                0.20292 nats
    oracle (best grid point per pos.)   0.10874 nats
    headroom                            0.09418 nats

| configuration | spend | KL | gain vs flat | capture |
| --- | --- | --- | --- | --- |
| live, as run (lagged dual, corrector) | 0.967 | 0.18127 | +10.7% | **23.0%** |
| corrected curve, dual solved on itself | 1.000 | 0.17800 | +12.3% | 26.5% |
| raw head prediction, no corrector | 1.000 | 0.18704 | +7.8% | 16.9% |
| true measured curve (ceiling at spend 1) | 1.000 | 0.12235 | +39.7% | 85.5% |

**The predictions were essentially met.** Capture 23.0% live against a predicted 20.1–32.9%; the
corrector is worth +9.6 points of capture (16.9% → 26.5%), close to the predicted doubling; the dual
lag costs 3.5 points. Converting to equivalent uniform search on the measured mean curve
(KL ∝ m^−0.85 at the deep end): live 0.18127 nats ≈ **1.18× uniform search at 0.967× spend, i.e.
~1.22× effective compute**. The oracle extrapolates to ≈2.27×, consistent with the 2.4–2.8× ceiling.

The allocation is also qualitatively sensible — it spends on uncertainty and skips decided positions:

| budget | share | mean ply | mean top-visit share |
| --- | --- | --- | --- |
| 0.125× | 8.7% | 83.5 | 0.780 |
| 0.200× | 2.7% | 67.0 | 0.821 |
| 0.333× | 6.5% | 78.1 | 0.699 |
| 0.500× | 15.2% | 53.6 | 0.734 |
| 0.667× | 10.2% | 56.0 | 0.665 |
| 1.000× | 30.0% | 55.6 | 0.582 |
| 1.500× | 10.0% | 56.9 | 0.454 |
| 2.000× | 16.7% | 59.5 | 0.393 |

Cheap budgets go to late-game, already-decided positions; deep budgets to contested ones.

## And it loses

64-search Stockfish-ladder Elo against wall-clock, all runs fitted with the same
`fit_ladder_elo` from per-rung W/D/L in the coordinator logs:

|     s | v13 | v14 | v15 | v16 | v17 | v18 | v20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9600 | 1414 | 1414 | 1426 | 1432 | 1491 | 1376 | 1431 |
| 14400 | 1537 | 1543 | 1537 | 1590 | 1636 | 1542 | 1601 |
| 21600 | 1656 | 1691 | 1648 | 1643 | 1729 | – | 1631 |
| 28800 | 1731 | 1704 | 1746 | 1718 | 1706 | – | 1688 |
| 36000 | 1817 | 1785 | – | – | 1823 | – | 1758 |
| 40800 | 1878 | 1809 | – | – | 1848 | – | 1820 |
| 45600 | – | 1862 | – | – | 1886 | – | 1788 |

v20 runs 60–100 Elo behind v13 and v17 from 21600s onward and does not close. Five runs — v16, v17,
v18, v19, v20 — have failed to beat runs without the subsystem. **v17, the strongest run of the
series, is the one in which the allocator was inert**: it ran the threshold rule with an inverted
isotonic projection and effectively searched flat.

## Why it fails

The direct costs do not explain the deficit. Deep labelling is ~1.6% of wall-clock. Comparing
generations by gate decision within the window where applied and closed interleaved (generations
60–130) gives 134.3 s/gen applied against 128.4 s/gen closed — a **4.6% cadence penalty**, with
credit-wait identical (48.5s vs 48.9s), so self-play is not stalling the trainer more when the
allocator runs. A 4.6% penalty against a 22% effective-compute gain should net positive.

The leading explanation is that **the proxy does not measure what matters**. `KL(pi_deep || pi_at_b)`
is a per-position fidelity measure. It cannot see the effect of a cheapened search on the *training
data* that search produces. KataGo (Wu 2019, "Accelerating Self-Play Learning in Go") reports that
policy learning needs on the order of 800 playouts before search deviates substantially from the
network's own prior, and therefore uses fast searches only to advance games, never as policy targets.
Our allocator cuts 36% of positions to a mean 0.364×, nearly 9% of them to 0.125× — at a 300–400
baseline that is 40–50 visits — and feeds all of them into policy training at equal weight. Those
targets are close to self-referential. The proxy reports a gain; the next generation's data is worse.

This is a hypothesis consistent with all the evidence, not a demonstrated mechanism. It is falsifiable:
excluding early-stopped or heavily-cheapened positions from the policy target, or down-weighting them,
should recover the loss if it is correct.

## Operational cost

The subsystem produced three run-killing defects, each caught only in production:

1. A second copy of configuration pins in `CurveCalibrationParameters.__post_init__` — killed v15 after
   ~16 generations.
2. An inverted isotonic projection (suffix minimum instead of prefix), which flattens a decreasing
   curve to its deepest value — v17 ran its entire length with the allocator effectively inert.
3. A dual seeded from *measured* curves, which carry exact plateaus, then applied to smooth *predicted*
   curves. The seed was ~1.4e-06 where the operating point is ~0.11, and a 5%/generation ratchet needed
   231 generations to cross the gap. v19 never applied the learned rule in 117 generations.

A fourth, latent: the corrector standardised window-constant features against an absolute 1e-6 floor,
so the first `baseline_visits` schedule step produced z-scores of ~1e8 and corrections of ~−1.2e7
log-KL, collapsing every position to the cheapest budget on generations 1, 10, 50 and 90.

## Conclusion

Predicted-budget-before-search is the *semi-static* category in the taxonomy of Lan et al. (AAAI-21),
who note of it that "once the model predicts a search depth, it cannot change it according to the
current searching result." We built the weakest category, and we built it without tree information —
necessarily, since the decision precedes the search. Our own feature ablation is consistent with this:
all recoverable signal came from `top_visit_share`, `policy_entropy`, `baseline_visits` and
`generation`, and none from the head's own output.

Retire per-position predicted budgets. For the next production run, remove the deep labelling, the
`head_training` block, and the `search_budget` auxiliary target at loss weight 0.2.

The plumbing is reusable and should be kept: deep-search label generation, analysis records,
TorchScript export and native consumption, calibration state and telemetry. What was wrong was the
decision rule and the feature set, not the infrastructure.

Successor work: `documentation/plan/adaptive-stopping-plan-20260901.md`.

## Sources

- Wu, D. J. (2019). *Accelerating Self-Play Learning in Go*. arXiv:1902.10565.
- Lan, Tsai, Wu, Wu & Hsieh (2021). *Learning to Stop: Dynamic Simulation Monte-Carlo Tree Search*.
  AAAI-21, arXiv:2012.07910.
