# Analysis

This directory holds investigations, literature reviews, and conclusions. An analysis explains evidence; it is not
an operational runbook or an accepted architecture by itself. Check each document’s date, source revision, and
status before applying a conclusion to current code.

## Compute-poor chess narrative

- [Reference recipes for a compute-poor run](reference-recipes-for-a-compute-poor-run.md) — comparison with
  AlphaZero, KataGo, lc0, and sample-prioritization research.
- [Adaptive search conclusion](adaptive-search-conclusion-20260904.md) — why learned allocation and stopping were
  removed.
- [Chess search findings](chess-search-findings-20260827.md) — fixed-budget and search-parameter evidence.
- [Elo scale and reporting](chess-elo-scale-and-reporting-20260911.md) — how the project’s Stockfish-node ladder
  relates to SSDF and why its values are not FIDE ratings.
- [v8 training-data comparison](v8-training-data-comparison-20260826.md) — replay-distribution investigation.
- [Conversion investigation](chess-conversion-investigation-20260826.md) — diagnosis of games that failed to
  convert winning positions.

## Regression and reproducibility investigations

- [V35-to-V42 regression audit](v35-v42-regression-audit-20260913.md) — complete source/configuration comparison.
  It found no demonstrated post-V35 defect explaining V42, identified the generation-zero seeding confound, and
  narrowed the only material fixed-model runtime difference to the warmup floor.
- [V35-to-V42 executable bisect](v35-v42-executable-bisect-20260913.md) — controlled follow-up that cleared typed
  template selection, phase-specific post-fold warmup, and live QAT sidecar identity for the compared settings. Its
  endpoint/control protocol should not be generalized into an optimizer or architecture conclusion.

## How these analyses feed the report

The [experiment catalog](../experiments/README.md) assigns a consistent status to each technique and links analyses
to their primary measurements. The [technical report](../report/README.md) is the narrative synthesis. In
particular, the SGD/QAT frozen-replay screens are benchmark evidence rather than analyses of playing strength, and
the corrected TensorRT equal-scale refit finding lives in the
[2026-09-21 benchmark record](../benchmarks/int8-template-staleness-rtx4070super-20260921/README.md).

The root [documentation index](../README.md) gives the complete reader path and distinguishes these analyses from
current operations and committed benchmark results.
