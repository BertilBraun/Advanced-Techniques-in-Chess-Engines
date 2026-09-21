# Deployment and operations

Procedures that are re-executed. Dated node and run records are evidence, listed separately below.

The active campaign is the final chess run. Its readable recipe is
[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml), and its archive/evaluation contract
is [Final chess run](../results/final-chess-run.md). The recipe does not authorize a lifecycle action: use Run control
and the current node record for every operation.

- [Run control](run-control.md) — **the only supported way to start, stop, inspect and archive a production or
  test run** (`deployment/run_control.sh`).
- [Experiment platform](experiment-platform.md) — current platform ownership plus explicitly marked historical Go
  screening-queue, ingestion, monitoring, and export procedures.
- [Evaluation engines](evaluation-engines.md) — pinned Stockfish and KataGo installation for WSL and fresh nodes.
- [Stockfish gauntlet](stockfish-gauntlet.md) — strength-calibration gauntlet runbook.
- [Experiment result export](experiment-result-export.md) — packaging terminal queue runs with reproducibility
  metadata; direct-run preservation and fetch remain part of Run control.
- [Public web play](web-play.md) — FastAPI/Modal backend and static browser client.
- [Lichess and Vast](lichess-vast-evaluation.md) — UCI/Lichess deployment and evidence collection.
- [`deployment/setup_remote.sh`](../../deployment/setup_remote.sh) — authoritative fresh training-node bootstrap.
- [`deployment/lichess/README.md`](../../deployment/lichess/README.md) — executable Lichess deployment runbook.

## Node and run records (evidence, dated)

- [Current node](current-node.md) — the one place that describes the currently rented node.
- [Final chess run](../results/final-chess-run.md) — current result status, required archive identity, and pending
  terminal measurements.
- [Phase A test-node provisioning, 2026-08-21](../evidence/node-phase-a-20260821.md)
- [Four-node comparison benchmark, 2026-08-21](../benchmarks/node-comparison-vast-4nodes-20260821/README.md).
- [R11 Vast integrated validation](../evidence/node-r11-vast-20260810.md) — frozen evidence from a destroyed node.
- [Second four-day run readiness](../plan/archive/chess-second-four-day-run-readiness.md) — superseded; the run was not launched.

Operational guides do not authorize external mutations, rentals, provisioning, or experiments by themselves.
