# Results

This directory contains result-level summaries for complete training runs. A result summary is the narrow bridge
between immutable benchmark evidence and reader-facing claims in the root README and technical report.

A result is publishable only when it records the exact source revision, resolved configuration hash, archived run
identity, selected checkpoint, evaluation protocol, raw result locations, and limitations. The configuration at
[`py/configs/production/chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) is the living
entry point for the project's settled chess recipe; a completed result additionally pins the exact revision and
resolved configuration that actually ran.

## Result summaries

- [Final chess run](final-chess-run.md) — active training lineage; terminal measurements pending.
- [v34 generation 1465](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md) — previous
  completed public benchmark.
