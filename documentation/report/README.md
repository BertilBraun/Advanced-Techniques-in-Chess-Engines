# Technical report

This report explains how the project built and studied a compute-constrained AlphaZero-style chess system. It is
repository-native: claims link to the benchmark, analysis, configuration, or frozen evidence that supports them.
Chess is the research result. Go 7x7 and 9x9 appear only where they explain the shared platform or an early design
decision, including why small-board Go was not retained as a proxy for chess hyperparameter optimization. Systems
work is presented as the throughput foundation for self-play learning rather than as a separate algorithmic claim.

> **Owner-review edition.** The complete narrative is typeset as a two-column PDF for one consolidated owner pass.
> The [narrative outline](narrative-outline.md) and [completion plan](publication-plan.md) remain editorial working
> records, not chapters of the review edition. The owner has already accepted the opening, the three failure-study
> directions, and the reader path.

The final chess training lineage and its reported teacher/student evaluations are complete. Terminal strength,
selected-checkpoint training volume, and cost fields remain centralized in [Final-run results](07-final-run-results.md).
Wider node throughput remains open; total project spend is intentionally not a report claim.

## Reader path

1. [Abstract](00-abstract.md) and [motivation and scope](01-motivation-and-scope.md)
2. [Methodology and evidence](02-methodology-and-evidence.md)
3. [System and training method](03-system-and-methods.md)
4. Investigations of [search](04a-search.md), [data and replay](04b-data-and-replay.md), and
   [networks and training](04c-networks-and-training.md)
5. [Systems optimization](05-systems-optimization.md)
6. [Three failures that changed the method](05a-three-failures.md)
7. [Final chess recipe](06-final-chess-recipe.md)
8. [Final-run results](07-final-run-results.md) — completed teacher, search, parallelism, and distillation results
9. [Limitations](08-limitations.md)
10. [Reproducibility](09-reproducibility.md)
11. [Conclusion](10-conclusion.md)
12. [Publication references](references-publication.md) and [working citation plan](bibliography.md)
13. [Research coverage matrix](coverage-matrix.md)
14. [Publication plan](publication-plan.md)
15. [Narrative outline and visualization strategy](narrative-outline.md)

## Report status

| Area | Status | Authority |
| --- | --- | --- |
| Narrative chapters and report figures | Complete owner-review edition; owner pass outstanding | [Opening](01-motivation-and-scope.md), [reader path](#reader-path) |
| Archive-derived training dynamics and throughput | Selected-checkpoint counters and figures complete; wider throughput open | [Completion plan](publication-plan.md) |
| Final recipe | Current, living recipe | [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) |
| Exact run identity and terminal strength | Complete and checksum-covered | [Chapter 7](07-final-run-results.md) |
| External bibliography | Clean review references plus a separate working source plan | [References](references-publication.md), [source plan](bibliography.md) |
| Research-question and artifact coverage | Audited | [Coverage matrix](coverage-matrix.md) |
| Final publication workflow | Planned | [Publication plan](publication-plan.md) |

The root [project README](../../README.md) is the short showcase. This report is the long-form account. Operational
instructions remain under [`documentation/operations/`](../operations/README.md); this report is not a runbook and
does not authorize compute, deployment, stopping, or deletion.

Build the PDF from the repository root with
`uv run --group publication python .\py\tools\render_technical_report.py`. The generated review copy is
`output/pdf/technical-report-review.pdf`; Markdown remains the editable source. Figure SVGs remain vector artwork in
the PDF. Relative evidence links are converted to repository links and will resolve against the default branch after
the documentation changes are merged.
