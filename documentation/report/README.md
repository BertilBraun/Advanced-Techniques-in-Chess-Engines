# Technical report

This report explains how the project built and studied a compute-constrained AlphaZero-style chess system. The
publication manuscript is self-contained: research claims are explained in the body or appendices, and external
works use numbered references. Repository benchmarks remain the audit trail, not required reading for the paper.
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
9. [Limitations](08-limitations.md) and [conclusion](10-conclusion.md)
10. [Publication references](references-publication.md) and [working citation plan](bibliography.md)
11. Appendices: [training diagnostics](appendix-a-training-diagnostics.md),
    [evaluation tables](appendix-b-evaluation-tables.md),
    [supporting comparisons](appendix-c-supporting-comparisons.md), and
    [reproducibility](appendix-d-reproducibility.md)
12. [Research coverage matrix](coverage-matrix.md)
13. [Publication plan](publication-plan.md)
14. [Narrative outline and visualization strategy](narrative-outline.md)

## Report status

| Area | Status | Authority |
| --- | --- | --- |
| Narrative chapters and report figures | Complete owner-review edition; owner pass outstanding | [Opening](01-motivation-and-scope.md), [reader path](#reader-path) |
| Archive-derived training dynamics and throughput | Selected-checkpoint counters and figures complete; wider throughput open | [Completion plan](publication-plan.md) |
| Final recipe | Current, living recipe | [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) |
| Exact run identity and terminal strength | Complete and checksum-covered | [Chapter 8](07-final-run-results.md) |
| External bibliography | Clean review references plus a separate working source plan | [References](references-publication.md), [source plan](bibliography.md) |
| Research-question and artifact coverage | Audited | [Coverage matrix](coverage-matrix.md) |
| Final publication workflow | Planned | [Publication plan](publication-plan.md) |

The root [project README](../../README.md) is the short showcase. This report is the long-form account. Operational
instructions remain under [`documentation/operations/`](../operations/README.md); this report is not a runbook and
does not authorize compute, deployment, stopping, or deletion.

Build the PDF from the repository root with
`uv run --group publication python .\py\tools\render_technical_report.py` after installing
[Tectonic](https://tectonic-typesetting.github.io/en-US/install.html). The generated review copy is
`output/pdf/technical-report-review.pdf`; Markdown remains the editable source. The PDF uses the same two-column
LaTeX geometry and font as the Voice-Light report, with clickable numbered citations and vector figures. The paper
does not use repository-file hyperlinks as substitutes for methods or results.

To regenerate the title-free paper plots while preserving the original showcase and evidence figures, run
`uv run --group publication python .\py\tools\render_ladder_progression.py --paper-only`,
`uv run --group publication python .\py\tools\render_final_search_curve.py --paper-only`, and
`uv run --group publication python .\py\tools\render_final_training_dynamics.py --paper-only` before building the
PDF. The appendices switch to one column so the diagnostics and dense tables remain readable.
