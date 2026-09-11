# Experiment plans

Plans record decisions and protocols. They do not authorize GPU spending or run lifecycle actions. Most files in
this directory belong to completed stages of the 2026 chess campaign and should be read as a chronological research
ledger rather than current instructions.

## Closing phase

- [v34 final evaluation and small-model distillation](v34-final-evaluation-and-distillation.md) — terminal
  measurement protocol. Replay compression is complete; terminal-run preservation and the final absolute-strength
  benchmark remain open.

## Campaign narrative

- [Post-four-day regression analysis](chess-post-four-day-regression-analysis-20260820.md) — diagnosis that led to
  the recovery campaign.
- [Chess recovery plan](chess-recovery-plan-20260820.md) — completed work-package ledger; no longer the active plan.
- [Chess search follow-up](chess-search-followup-plan-20260827.md) — measurement agenda that led to fixed-search
  decisions.
- [v29 handoff and next-run design](next-run-handoff-20260906.md) — evidence and questions that shaped v34.

## Superseded search designs

The following designs were implemented or evaluated and then retired. Their conclusions are summarized in
[`analysis/adaptive-search-conclusion-20260904.md`](../analysis/adaptive-search-conclusion-20260904.md).

- [Scalar adaptive search budget](adaptive-search-budget-20260827.md)
- [Predicted-curve adaptive search budget](search-budget-curve-20260830.md)
- [Learned early stopping](adaptive-stopping-plan-20260901.md)
- [v14 adaptive-run decision](v14-decision-plan-20260829.md)

## Supporting run records

- [Offline search evaluation plan](search-evaluation-plan-20260826.md)
- [Run 1 versus r3](run1-r3-diff.md)
- [Run 2 versus r3](run2-r3-diff.md)

Older preparation plans with explicit supersession banners live under [`archive/`](archive/). The
[experiment backlog](../../THINGS_TO_TRY.md) contains uncommitted ideas, and [references](../references.md) lists
papers and external resources.
