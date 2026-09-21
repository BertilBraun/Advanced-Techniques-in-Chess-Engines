# Benchmark and result evidence

This directory contains historical measurements and artifacts, not current architecture or operational guidance.
Every result is scoped to its recorded source revision, hardware, configuration, and date. Commands in an artifact
README may reproduce that historical run but may reference paths or interfaces removed later.

The current recipe is [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml), and current
deployment commands live in the [operations guides](../operations/README.md). Historical figures here are not
acceptance criteria. The [experiment catalog](../experiments/README.md) classifies every technique and the
[technical report](../report/README.md) synthesizes the evidence into the project narrative.

New benchmarks follow [TEMPLATE.md](TEMPLATE.md).

## Harnesses

Harness documentation lives under [harnesses/](harnesses/). The
[Chess progressive-model inference benchmark](harnesses/chess-progressive-inference.md) loads the historical
`6x96 -> 10x160 -> 15x192` attention ladder from `vast-chess-8gpu-optimal.yaml`; its
[RTX 4070 SUPER result](chess-direct-policy-final-progressive-rtx4070s-20260818/README.md) is retained as historical
architecture evidence, not as the harness or model ladder for the current convolutional final recipe.

## Notable results

- [TensorRT equal-scale refit failure and fix (2026-09-21)](int8-template-staleness-rtx4070super-20260921/README.md)
  — the V90 collapse was **not** ordinary template staleness. TensorRT 10.14 optimization levels 4–5 incorrectly
  optimized equal Q/DQ scales that later became unequal during refit. Separating scales before template build and
  defaulting to optimization level 3 fixed the reproduced failure; periodic rebuild is insurance, not the fix.
- [V76/V35 medium-model pre-fold backend comparison (2026-09-18)](v76-v35-medium-prefold-backend-20260918/README.md)
  — the 14x160 scaled-post model gained 39.1% search throughput from TensorRT INT8 in the production actor topology.
- [V76/V35 small-model pre-fold backend comparison (2026-09-18)](v76-v35-small-prefold-backend-20260918/README.md)
  — the 12x128 model gained 14.4% search throughput in the same topology; this throughput result is not an Elo
  ablation.
- [QAT post-fold learning-rate sweep (2026-09-14)](chess-sgd-postfold-lr-rtx4070s-20260914/README.md) — a target
  rate of 0.08 gave the best 10,000-step frozen-replay loss among 0.02–0.08. This selected a candidate for online
  testing; it did not measure playing strength, and the final recipe later adopted a different long pre-fold
  lifecycle.
- [QAT pre-fold schedule factorial (2026-09-14)](chess-sgd-prefold-factorial-rtx4070s-20260914/README.md) — the
  historical warmup with folding delayed to step 3,000 led the four controlled frozen-replay arms. All arms learned
  after folding; the result is a training proxy, not an Elo claim.
- [SGD QAT replay screen (2026-09-13)](chess-sgd-replay-screen-rtx4070s-20260913/README.md) — Nesterov SGD was
  stable on the preserved v34 replay, and a warmed 0.06 deployment target led the three short arms. Its one seed,
  stationary replay, and primary-only objective limit transfer to online self-play.
- [V39 INT8 self-play decomposition (2026-09-13)](v39-selfplay-throughput-rtx4070s-20260913/README.md) — INT8
  delivered 1.861x matched exclusive search throughput, while live admitted positions improved 23.3% over the cited
  V35 period. Game length, completion gating, trainer overlap, and replay reuse explain why core speed did not
  translate one-for-one into training cadence.
- [V35-code/V42-initialization controlled A/B protocol (2026-09-13)](v35-code-v42-generation0-controlled-ab-20260913/README.md)
  — preserves the reproducible endpoint-control setup used by the
  [source/configuration audit](../analysis/v35-v42-regression-audit-20260913.md) and
  [executable bisect](../analysis/v35-v42-executable-bisect-20260913.md). The benchmark README records the protocol,
  not a completed broad claim about SGD, QAT, or playing strength.
- [TensorRT INT8 salvage investigation (2026-09-12)](tensorrt-int8-salvage-rtx4070s-20260912/README.md) — the
  approximately 3x full-trunk INT8 core speed is real but invalid; calibration sweeps, all-layer sensitivity,
  mixed precision, autotuning, QAT, SmoothQuant, weight-only INT8, per-channel activation quantization, FP8,
  and a 32-channel value head did not produce a fidelity-valid material speedup.
- [TensorRT INT8 frozen-replay screen (2026-09-12)](tensorrt-int8-replay-screen-rtx4070s-20260912/README.md) —
  scaled post-activation QAT recovered useful fidelity and measured deployment throughput; folding only after
  training broke fidelity. Frozen-replay target loss and output agreement remain proxies for chess strength.
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
  2,844.8 benchmark Elo [2,806.6, 2,882.3] at 10,000 searches, superseded as the strongest completed result by v34.
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

- The exhaustive [benchmark coverage ledger](../experiments/benchmark-coverage.md) maps every benchmark README to
  its topic, experiment status, and appropriate technical-report use.
- Frozen-replay loss, top-action agreement, output fidelity, isolated model throughput, and exclusive search
  throughput are screening measurements. None alone establishes self-play Elo or end-to-end learning efficiency.
- The September TensorRT template investigation contains an explicit correction: refitted scales do reach the
  engine, and ordinary age was not the V90 failure. Cite the final equal-scale optimization finding, not the initial
  staleness hypothesis.
- `self-play-cpp-baseline-4x8x3x96-20260720T071550Z`, `…T073130Z` and
  `self-play-cpp-batching-timeout5000us-20260720T073957Z` are three snapshots of **one** inference-batching study;
  do not read them as independent results.
- `ddp-model-throughput-20260720` defers to `ddp-production-training-20260720` as the authoritative production
  measurement for the same figures.
- Directories dated 2026-07 predate the current evidence rules (config SHA, full source SHA, hardware segment in
  the name) and are intentionally not retro-fitted.
