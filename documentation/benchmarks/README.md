# Benchmark and result evidence

This directory contains historical measurements and artifacts, not current architecture or operational guidance.
Every result is scoped to its recorded source revision, hardware, configuration, and date. Commands in an artifact
README may reproduce that historical run but may reference paths or interfaces removed later.

Current acceptance criteria come from the per-generation yardstick in
[the chess recovery plan](../research/chess-recovery-plan-20260820.md); deployment commands from the
[operations guides](../operations/README.md). Historical figures here are not acceptance criteria.

## Active harnesses

The [Chess progressive-model inference benchmark](chess-progressive-inference.md) is the sole active harness for
measuring the retained production Chess models. It loads those models directly from the production configuration.
The [final RTX 4070 SUPER acceptance result](chess-direct-policy-final-progressive-rtx4070s-20260818/README.md)
records the exact `6x96 -> 10x160 -> 15x192` production-model throughput and parameter counts.

## Notable results

- [Four-node Vast comparison (2026-08-21)](node-comparison-vast-4nodes-20260821/README.md) — the reference for
  node selection; also a model README for new benchmarks.
- [Four-RTX-3060 self-play throughput baseline](self-play-throughput-rtx3060.md) — chess and Go 7x7 capacity
  topologies with the matched pre-rework comparison.
- [Naive Python chess MCTS baseline](naive-python-mcts-rtx3060-20260816/README.md) — deliberately unoptimized
  batch-one PUCT reference (~81 sims/s) as an order-of-magnitude comparison for the native search.
- [Two-GPU Go 7x7 training baseline](go-7x7-two-gpu-training-baseline.md) — proposed comparison baseline.
- `chess-results/` — the original 2024 trained model, games, plots, and logs as historical strength evidence
  (large binaries; scheduled to move to the `pre-rework` GitHub release).

## Reading notes

- `self-play-cpp-baseline-4x8x3x96-20260720T071550Z`, `…T073130Z` and
  `self-play-cpp-batching-timeout5000us-20260720T073957Z` are three snapshots of **one** inference-batching study;
  do not read them as independent results.
- `ddp-model-throughput-20260720` defers to `ddp-production-training-20260720` as the authoritative production
  measurement for the same figures.
- Directories dated 2026-07 predate the current evidence rules (config SHA, full source SHA, hardware segment in
  the name) and are intentionally not retro-fitted.
