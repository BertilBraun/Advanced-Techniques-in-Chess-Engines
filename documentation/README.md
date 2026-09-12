# Documentation

This index is the reader path through the project. Documents are grouped by purpose because the repository retains
failed experiments and superseded designs as research evidence. A date or confident tone in an old document does
not make it current guidance.

## Start here

1. Read the [root README](../README.md) for the result and system overview.
2. Read [Current state](CURRENT-STATE.md) for what is final, provisional, active, and still missing.
3. Read the [Python](../py/README.md) or [C++](../cpp/README.md) guide before changing that runtime.
4. Read [Run control](operations/run-control.md) and the
   [experiment platform](operations/experiment-platform.md) before touching a run or rented node.

## Results

The retained v34 three-day checkpoint is the project’s main chess result. It reached 3,037 benchmark Elo at 10,000
searches and 3,167 at 80,000 searches on the project’s SSDF-derived Stockfish 13 fixed-node ladder.

| Result | Status | Evidence |
| --- | --- | --- |
| v34 training dynamics and scaling | **Final through generation 1702** | [Hourly curves, throughput, and outscaling playbook](benchmarks/chess-v34-training-dynamics-rtx4070s-20260912/README.md) |
| v34 generation 1465 replay compression | **Final** | [13.20x smaller student and match artifacts](benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md) |
| v34 generation 1465 terminal strength | **Final** | [3,037 at 10k and 3,167 at 80k](benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md) |
| v29 generation 936 deep match | **Final** | [2,844.8 Elo at 10,000 searches](benchmarks/deep-match-generation936-50k-nodes-rtx4070s-20260906/README.md) |
| v29 strength over wall-clock | **Final** | [generations 100–1000 ladder](benchmarks/ladder-elo-vs-generation-rtx4070s-20260906/README.md) |
| Four-day historical baseline | **Final, older runtime** | [frozen evidence](evidence/chess-four-day-freeze-20260817/) |

“Final” means the measurement and compact evidence are committed. It does not mean that the absolute Elo scale is
equivalent to FIDE, online-server, CCRL, or current Stockfish ratings. The rating-scale analysis is separate work.
See [What the v34 Elo numbers mean](analysis/chess-elo-scale-and-reporting-20260911.md) for the calibrated scale and
recommended public language.

## Directory guide

| Directory | Purpose | Authority |
| --- | --- | --- |
| [`analysis/`](analysis/) | Investigations, literature reviews, and conclusions | Evidence and reasoning; check date and status |
| [`architecture/`](architecture/README.md) | Accepted designs and implementation records | Current only where the file’s banner says so |
| [`benchmarks/`](benchmarks/README.md) | Raw measurements and run-specific reports | Authoritative for that exact revision and configuration |
| [`evidence/`](evidence/README.md) | Frozen run and node records | Immutable evidence, never operating guidance |
| [`operations/`](operations/README.md) | Procedures intended to be run again | Current operational guidance |
| [`plan/`](plan/README.md) | Active and completed experiment plans | Planning record; does not itself authorize compute |
| [`history/`](history/README.md) | Pre-rework and superseded material | Archival and non-normative |

The [benchmark template](benchmarks/TEMPLATE.md) defines the evidence expected for new measurements. Large fetched
archives live under the gitignored `.codex-diagnostics/`; compact results and hashes belong in `benchmarks/` or
`evidence/`.

## Current technical guides

- [Python runtime architecture](architecture/python-runtime-rework.md)
- [Replay pipeline](architecture/replay-pipeline-rework.md)
- [Progressive model sizing](architecture/progressive-model-sizing.md)
- [Run control](operations/run-control.md)
- [Evaluation engines](operations/evaluation-engines.md)
- [Stockfish gauntlet](operations/stockfish-gauntlet.md)
- [Experiment result export](operations/experiment-result-export.md)
- [Web play](operations/web-play.md)

Read supersession banners inside these documents. Some architecture files preserve the reasoning for components
that were subsequently replaced.

## Research narrative

For the shortest path through the compute-poor chess work:

1. [Post-four-day regression analysis](plan/chess-post-four-day-regression-analysis-20260820.md) identifies why
   earlier rework stopped learning.
2. [Chess recovery plan](plan/chess-recovery-plan-20260820.md) records the recovery campaign.
3. [Search findings](analysis/chess-search-findings-20260827.md) and the
   [adaptive-search conclusion](analysis/adaptive-search-conclusion-20260904.md) record why adaptive allocation
   and learned stopping were removed.
4. [v29 handoff](plan/next-run-handoff-20260906.md) connects throughput collapse, replay reuse, and data scarcity.
5. [Reference recipes for a compute-poor run](analysis/reference-recipes-for-a-compute-poor-run.md) compares the
   design with AlphaZero, KataGo, lc0, and later work.
6. [v34 final evaluation and distillation](plan/v34-final-evaluation-and-distillation.md) defines the closing
   measurement protocol; the compression branch of that plan is complete.

## Document lifecycle

- Put reproducible procedures in `operations/`.
- Put accepted component designs in `architecture/`.
- Put dated measurements in `benchmarks/<topic>-<hardware>-<date>/`.
- Put investigations and literature synthesis in `analysis/`.
- Put experiment decisions in `plan/`, with a status at the top.
- Keep superseded material only when it explains a decision or preserves evidence; mark it clearly and index it as
  historical.

Research plans and operational guides never authorize spending, launch, stop, or deletion by themselves. The user
owns those decisions.
