# Chess throughput history — self-play and trainer, 2026-08-21

Cross-era reference assembled during Phase A for future hardware or software throughput work. Numbers from
different hardware are shown side by side deliberately; every row names its hardware and net, and cross-row
ratios are only meaningful with those columns in mind.

## Self-play search throughput (per GPU)

| era / source | hardware | stack | net | mix | searches/s per GPU | node total |
|---|---|---|---|---|---:|---:|
| Four-day run r3/r4, in-run (regression analysis §3.2) | 8×3060 | old (pre-rework) | 3.1 M dense-head CNN 12×112, ps=1 | production | ≈20k evals/s | ≈160k evals/s |
| 12 Aug self-play latency bench (`chess-self-play-latency-rtx3060-20260812.md`) | 8×3060 | new | attention, ps=4 | production, self-play only | ≈25–27k | 201–213k |
| 12 Aug mixed-contention bench (`chess-training-throughput-rtx3060-20260812/`) | 8×3060 (8 active workers) | new | attention | production, trainer active | ≈21k | 169k |
| Failed runs 18–19 Aug (v11 credit telemetry, 93 quanta, `.codex-diagnostics/v11-credit-training-telemetry.jsonl`) | 8×3090 | new, bounded ingestion | attention | production | ≈18.7k median | 150k median, 178k p90 |
| 18 Aug node comparison (analysis §3.2) | 3090 | new | 0.47 M attention | **8-visit micro-mix — not comparable** | ≈61k evals/s | — |
| 21 Aug node comparison (operations note) | 4070 | new | attention 8×128, best ppg grid | benchmark mix | 104.6k | — |
| 21 Aug WP5 (`search-throughput-rtx4070-20260821/`) | 1×4070, single process | phase-a | 1.16 M plane-head CNN 12×112 | 25 % full, 500 visits, ps=4 | 93.5k | — |
| 21 Aug WP5 | same | phase-a | same | 25 % full, 800 visits, ps=4 | 116.4k | — |

Readings:

- The failed 3090 runs achieved 3060-class search throughput: the bounded-ingestion coordinator kept ~8 of 24
  workers paused (analysis §2.3). Median unique positions/s was **405** on the whole node versus 250–750 for the
  four-day run on 3060s with a 2.7× larger net. The era's throughput loss was coordination, not compute.
- One 4070 at `parallel_searches` 4 ≈ 75 % of the entire failed 8×3090 node's median search rate.
- `parallel_searches` 1→4 is worth +35–38 % on a 4070 via inference batch fill (18–20 → 41–44 avg batch).

## Trainer throughput

| era / source | hardware | mode | net | samples/s | 500-step quantum (batch 2048) |
|---|---|---|---|---:|---:|
| Four-day run (analysis §3.2) | 8×3060 DDP | eager | 3.1 M CNN | ≈12k | ≈85 s |
| Failed-era bench (`training-final-15x192.json`) | 8×3090 DDP | compiled | 15×192 | 8.6k | ≈119 s |
| Failed v11 run (credit telemetry) | 8×3090 DDP | compiled | attention | ≈12k effective | 84 s median optimizer wall |
| 21 Aug WP4 testbed cell | **1×4070** | eager | 1.16 M plane-head CNN 12×112 | 8.6k | ≈119 s single-GPU |

Readings:

- One 4070 matches the failed-era 8×3090 compiled trainer rate and reaches ≈70 % of the four-day 8×3060 DDP
  rate on the smaller net. An 8×4070-class node should push quanta well under the old 85 s (per-GPU rate ×
  world size, minus DDP scaling losses — not yet measured; do not assume linear ×8).
- Consequence for Phase B (8×4070 S class): the trainer stops being the bottleneck; wall-clock lives in
  self-play throughput and ingestion freshness. Measure DDP scaling on the real node before revising
  `optimizer_steps_per_quantum` expectations.

Provenance: WP5 raw JSON in `.codex-diagnostics/wp5-throughput-20260821/`; v11 telemetry in
`.codex-diagnostics/v11-credit-training-telemetry.jsonl`; four-day numbers from
`documentation/plan/chess-post-four-day-regression-analysis-20260820.md` §3.2; WP5/WP4 configs resolved from
`vast-chess-8gpu-run1.yaml` (stand-in topology, SHA `f52e87ae…`).
