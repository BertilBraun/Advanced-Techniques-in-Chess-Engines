# Documentation index

The directory name determines a document's authority. `research/` holds the current plan and its evidence,
`architecture/` the accepted designs, `operations/` re-executed procedures. Material under `history/` and
`benchmarks/` is evidence about a particular earlier revision or run and must not be used as current guidance.
This index is updated in the same commit as any document it lists.

## Start here

- [Current state](CURRENT-STATE.md) — what the system is today, in two pages.
- [Onboarding](ONBOARDING.md) — read-in-this-order path for a new contributor or agent.
- [Chess recovery plan](plan/chess-recovery-plan-20260820.md) — **the current plan**; work is referenced by
  WP number and its per-generation yardstick is the pass/fail reference for new runs.

## Current authority

- [Chess recovery plan](plan/chess-recovery-plan-20260820.md), grounded in the
  [post-four-day regression analysis](plan/chess-post-four-day-regression-analysis-20260820.md).
- [Python runtime architecture](architecture/python-runtime-rework.md) — accepted Python ownership and process
  model (read its as-built banner: the ingestion contract is being reworked under WP2).
- [Progressive model sizing](architecture/progressive-model-sizing.md) — accepted training policy.
- [Run control](operations/run-control.md) — the only supported way to start, stop, inspect and archive runs.
- [Python runtime README](../py/README.md) and [C++ README](../cpp/README.md) for entry points and builds.

The R1–R12 ledger in [platform-rework.md](architecture/platform-rework.md) is **closed** and authorises nothing;
it is retained as the historical record of the platform rework.

## Where a new document goes

| Kind | Destination |
| --- | --- |
| Current plan or its evidence | `research/` |
| Accepted design | `architecture/` |
| Procedure that is re-executed | `operations/` |
| Measurement | `benchmarks/<topic>-<hardware>-<date>/` |
| Per-run or per-node record | dated file, non-normative (see `operations/` node notes) |
| Superseded document | stays in place with a supersession banner, never silently deleted |

## Evidence, not guidance

- [`benchmarks/`](benchmarks/README.md) — measured artifacts, each scoped to its recorded revision, hardware,
  configuration and date.
- [`history/`](history/README.md) — pre-rework designs and optimization notes, non-normative.

## Research

- [Experiment backlog](../THINGS_TO_TRY.md) — candidate experiments; ideas, not authorized runs.
- [Research references](references.md) — papers and external resources.
