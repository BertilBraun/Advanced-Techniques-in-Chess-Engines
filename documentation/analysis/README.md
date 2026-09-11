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

The root [documentation index](../README.md) gives the complete reader path and distinguishes these analyses from
current operations and committed benchmark results.
