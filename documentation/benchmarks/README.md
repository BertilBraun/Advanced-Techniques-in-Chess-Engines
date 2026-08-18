# Benchmark and result evidence

This directory contains historical measurements and artifacts, not current architecture or operational guidance.
Every result is scoped to its recorded source revision, hardware, configuration, and date. Commands in an artifact
README may reproduce that historical run but may reference paths or interfaces removed later.

The [Chess progressive-model inference benchmark](chess-progressive-inference.md) is the sole active harness for
measuring the retained production Chess models. It loads those models directly from the production configuration.
The [final RTX 4070 SUPER acceptance result](chess-direct-policy-final-progressive-rtx4070s-20260818/README.md)
records the exact `6x96 -> 10x160 -> 15x192` production-model throughput and parameter counts.

Use the [platform ledger](../architecture/platform-rework.md) for current acceptance evidence and the
[operations guides](../operations/README.md) for current deployment commands.

The [four-RTX-3060 self-play throughput baseline](self-play-throughput-rtx3060.md) records the current chess and Go
7x7 capacity topologies, the matched pre-rework comparison, optimization rationale, and reproduction evidence.

The [naive Python chess MCTS baseline](naive-python-mcts-rtx3060-20260816/README.md) records a deliberately
unoptimized, single-game, batch-one PUCT reference measured at approximately 81 simulations/s on one training-loaded
RTX 3060. It provides an order-of-magnitude comparison point for the native batched search path.

The [proposed two-GPU Go 7x7 training baseline](go-7x7-two-gpu-training-baseline.md) records the complete two-hour
experiment configuration and the measurements required before it becomes the comparison baseline.

`chess-results/` preserves the original trained model, games, plots, and logs as historical strength evidence.
