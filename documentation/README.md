# Documentation index

The directory name determines a document's authority. Current plans and architecture live under `architecture/`;
deployment instructions live under `operations/`. Material under `history/` and `benchmarks/` is evidence about a
particular earlier revision or run and must not be used as current implementation guidance.

## Current authority

- [Platform plan and execution ledger](architecture/platform-rework.md) is the authoritative task/status ledger.
- [Python runtime architecture](architecture/python-runtime-rework.md) defines the accepted Python ownership and
  process model plus the remaining authorized cleanup boundary.
- [Python runtime README](../py/README.md) documents current entry points, setup, and validation.
- [C++ README](../cpp/README.md) documents the native build and runtime boundary.

Only the user accepts a phase or authorizes another. R1 through R9 are accepted. The post-R9 Python Phase 4 cleanup
slice is awaiting review; R10, remaining R11/hardware validation, and R12 are pending and unauthorized.

## Deployment and operations

- [Public web play](operations/web-play.md)
- [Lichess/Vast deployment and calibration](operations/lichess-vast-evaluation.md)
- [Fresh compute-node bootstrap](../deployment/setup_remote.sh)
- [Lichess deployment runbook](../deployment/lichess/README.md)

The fresh-node bootstrap installs the locked project environment and native extension. It does not provision
Stockfish, KataGo, their models, or their configurations.

## Research

- [Experiment backlog](../THINGS_TO_TRY.md) is the single current backlog of candidate experiments.
- [Research references](research/references.md) collects papers and external resources.

Backlog entries are ideas, not authorized implementation tasks or approved experiment runs.

## Historical implementation notes

[`history/`](history/README.md) preserves selected earlier designs, inventories, and optimization notes. They describe
superseded architectures and are non-normative. The obsolete trainer roadmap, clean-run plan, and duplicate future
work list were deleted because their replay, evaluation, resignation, and iteration-based assumptions no longer
match the repository.

## Benchmark and result evidence

[`benchmarks/`](benchmarks/README.md) preserves measured artifacts and reports. Each result applies only to its named
revision, hardware, configuration, and date. Historical throughput or strength figures are not current acceptance
criteria unless the platform ledger explicitly adopts them.
