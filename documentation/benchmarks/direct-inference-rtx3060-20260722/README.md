# RTX 3060 direct inference matrix

This directory contains the 2026-07-22 CUDA matrix described in
`../direct-evaluation-inference.md`.

- `results.json` is the structured report with provenance and all 133 configurations.
- `results.jsonl` is the append-order native output captured during the run.

The run used a seeded synthetic 12-layer, 112-channel network. It measures inference throughput,
not chess strength. Every configuration completed; there were no OOM skips. GPU utilization is
a coarse 100 ms process-level sample that includes model loading and warmup.
