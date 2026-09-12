# Benchmark and result evidence

This directory contains historical measurements and artifacts, not current architecture or operational guidance.
Every result is scoped to its recorded source revision, hardware, configuration, and date. Commands in an artifact
README may reproduce that historical run but may reference paths or interfaces removed later.

Current acceptance criteria come from the per-generation yardstick in
[the chess recovery plan](../plan/chess-recovery-plan-20260820.md); deployment commands from the
[operations guides](../operations/README.md). Historical figures here are not acceptance criteria.

New benchmarks follow [TEMPLATE.md](TEMPLATE.md).

## Active harnesses

Harness documentation lives under [harnesses/](harnesses/). The [Chess progressive-model inference benchmark](harnesses/chess-progressive-inference.md) is the sole active harness for
measuring the retained production Chess models. It loads those models directly from the production configuration.
The [final RTX 4070 SUPER acceptance result](chess-direct-policy-final-progressive-rtx4070s-20260818/README.md)
records the exact `6x96 -> 10x160 -> 15x192` production-model throughput and parameter counts.

## Notable results

- [TensorRT INT8 salvage investigation (2026-09-12)](tensorrt-int8-salvage-rtx4070s-20260912/README.md) — the
  approximately 3x full-trunk INT8 core speed is real but invalid; calibration sweeps, all-layer sensitivity,
  mixed precision, autotuning, QAT, SmoothQuant, weight-only INT8, per-channel activation quantization, FP8,
  and a 32-channel value head did not produce a fidelity-valid material speedup.
- [v34 training dynamics (2026-09-12)](chess-v34-training-dynamics-rtx4070s-20260912/README.md) — 732,500
  optimizer steps, 2.93 million games, 187.5 million fresh positions, hourly strength curves, throughput by model
  and visit phase, compute-doubling returns, and a multi-node scaling playbook.
- [v34 terminal strength (2026-09-11)](chess-terminal-v34-generation1465-rtx4070s-20260911/README.md) — generation
  1465 scored 3,037 benchmark Elo [3,012, 3,061] at 10,000 searches and 3,167 [3,143, 3,193] at 80,000 searches,
  with four complete 400-game result files and the SSDF-anchor caveat.
- [v34 replay compression (2026-09-11)](chess-replay-distillation-v34-rtx4070s-20260911/README.md) — a published
  474,069-parameter student, 13.20x smaller than its teacher; -166 Elo under the measured saturated equal-time
  workload, with model files and compact raw evidence.
- [v29 generation-936 deep match (2026-09-06)](deep-match-generation936-50k-nodes-rtx4070s-20260906/README.md) —
  the latest fully documented absolute chess rating: 2,844.8 Elo [2,806.6, 2,882.3] at 10,000 searches.
- [v29 Elo versus generation (2026-09-06)](ladder-elo-vs-generation-rtx4070s-20260906/README.md) — wall-clock and
  generation trajectory through generation 1,000, including the late throughput collapse.
- [Four-node Vast comparison (2026-08-21)](node-comparison-vast-4nodes-20260821/README.md) — the reference for
  node selection; also a model README for new benchmarks.
- [Four-RTX-3060 self-play throughput baseline](self-play-throughput-4xrtx3060-20260809/README.md) — chess and Go 7x7 capacity
  topologies with the matched pre-rework comparison.
- [Naive Python chess MCTS baseline](naive-python-mcts-rtx3060-20260816/README.md) — deliberately unoptimized
  batch-one PUCT reference (~81 sims/s) as an order-of-magnitude comparison for the native search.
- [Two-GPU Go 7x7 training baseline](go-7x7-training-baseline-2xrtx3060-20260810/README.md) — proposed comparison baseline.
- [Chess attention viability](chess-attention-viability-rtx3060-20260827/README.md) — the from-to policy head is
  worth 0.032 nats on the production convolutional trunk and 0.157 on an attention one; the attention trunk
  itself is 0.006 nats behind convolution at matched parameters. Also records that the generation-0 attention
  prior really was near-uniform on real positions, and that the calibration fix now covers both trunks.
- [Stockfish ladder on the four-day checkpoints](chess-stockfish-ladder-8xrtx3060-20260816/README.md) — the
  generation-445 ladder report and match records. The unrelated 2024 legacy artifacts that shared the old
  `chess-results/` directory live in [evidence/chess-legacy-a10-2024/](../evidence/chess-legacy-a10-2024/README.md)
  and the `pre-rework` GitHub release.

## Reading notes

- `self-play-cpp-baseline-4x8x3x96-20260720T071550Z`, `…T073130Z` and
  `self-play-cpp-batching-timeout5000us-20260720T073957Z` are three snapshots of **one** inference-batching study;
  do not read them as independent results.
- `ddp-model-throughput-20260720` defers to `ddp-production-training-20260720` as the authoritative production
  measurement for the same figures.
- Directories dated 2026-07 predate the current evidence rules (config SHA, full source SHA, hardware segment in
  the name) and are intentionally not retro-fitted.
