# Technical report

This report explains how the project built and studied a compute-constrained AlphaZero-style chess system. It is
repository-native: claims link to the benchmark, analysis, configuration, or frozen evidence that supports them.
Chess is the research result. Go 7x7 and 9x9 appear only where they explain the shared platform or an early design
decision, including why small-board Go was not retained as a proxy for chess hyperparameter optimization. Systems
work is presented as the throughput foundation for self-play learning rather than as a separate algorithmic claim.

> **Writing pass ready.** The topic-first source dossiers and final strength evidence are complete enough to begin the
> narrative rewrite. The chapters remain evidence-bearing drafts rather than final prose. Use the
> [owner review guide](source-notes/owner-review-guide.md) for the small set of decisions that need project-owner
> judgment; the full dossiers are not an owner review assignment.

The final chess training lineage and its reported teacher/student evaluations are complete. Terminal strength, cost,
and training-volume fields remain centralized in [Final-run results](07-final-run-results.md). Cost and volume claims
that still require archive reconciliation remain explicitly marked rather than inferred from live dashboards.

## Reader path

1. [Motivation and scope](01-motivation-and-scope.md)
2. [Methodology and evidence](02-methodology-and-evidence.md)
3. [System and training method](03-system-and-methods.md)
4. [Research investigations](04-research-investigations.md), with detailed topic chapters on
   [search](04a-search.md), [data and replay](04b-data-and-replay.md),
   [networks and training](04c-networks-and-training.md), and
   [lineage and decision chronology](04d-lineage-and-decisions.md)
5. [Systems optimization](05-systems-optimization.md)
6. [Final chess recipe](06-final-chess-recipe.md)
7. [Final-run results](07-final-run-results.md) — completed teacher, search, parallelism, and distillation results
8. [Limitations](08-limitations.md)
9. [Reproducibility](09-reproducibility.md)
10. [Conclusion](10-conclusion.md)
11. [Bibliography and citation plan](bibliography.md)
12. [Research coverage matrix](coverage-matrix.md)
13. [Publication plan](publication-plan.md)

## Report status

| Area | Status | Authority |
| --- | --- | --- |
| Problem, system, methods, and investigations | Topic-first narrative rewrite ready | [Source dossiers](source-notes/README.md) |
| Final recipe | Current, living recipe | [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) |
| Exact run identity and terminal strength | Complete and checksum-covered | [Chapter 7](07-final-run-results.md) |
| External bibliography | Working list | [Bibliography](bibliography.md); entries marked “verify” need publication-pass checks |
| Research-question and artifact coverage | Audited | [Coverage matrix](coverage-matrix.md) |
| Final publication workflow | Planned | [Publication plan](publication-plan.md) |

The root [project README](../../README.md) is the short showcase. This report is the long-form account. Operational
instructions remain under [`documentation/operations/`](../operations/README.md); this report is not a runbook and
does not authorize compute, deployment, stopping, or deletion.
