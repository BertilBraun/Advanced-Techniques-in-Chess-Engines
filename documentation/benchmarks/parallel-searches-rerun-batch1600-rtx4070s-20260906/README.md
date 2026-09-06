# `parallel_searches` 1 to 8 in the regime where the cap actually binds (batch 1600)

| | |
| --- | --- |
| `experiment_configuration_sha256` | `8f5c82e5ad316328d2ed7dd6703eec866704b95f7b6aef29627ea9e087e08ab4` |
| Source revision | `2b6e7186d0e0994a161f04e2b3b357c34302969b` (clean) |
| Node | `82.225.150.130:15801`, 1x RTX 4070 SUPER 12 GB (GPU-4d95807c), driver 595.84, AMD Threadripper PRO 5965WX, 6.58 effective CPUs of 48 visible, 188 GiB RAM |
| Date | 2026-09-06 |

Rerun of `parallel-searches-sweep-rtx4070s-20260906` in a configuration where the flag can take
effect. That sweep is uninformative and was marked so in its CORRECTION section; this is the
follow-up it asks for.

## Method

`parallel_searches` is an **upper cap on in-flight leaves per tree**, enforced at
`cpp/src/search/SearchExecutor.hpp:719` as `task.in_flight < task.parallel_searches`, with trees
served round-robin from `m_nextTask`. A tree therefore gets a second concurrent leaf only after the
scheduler has cycled every other schedulable tree. With ~200 trees on turn and an inference batch of
64, the batch fills from 64 distinct trees at one leaf each and **the cap is never reached**. That is
why the batch-64 sweep measured essentially nothing.

This rerun raises `inference_batch_size` to **1600** — 8 x 200 trees — so that the scheduler can
cycle every tree and come back around, and the cap binds.

Four arms, `parallel_searches` 1 / 2 / 4 / 8, **200 games each** (100 opening pairs played both
colours), 800 searches per move, against Stockfish 13 at 10,000 fixed nodes, identical openings and
match seed across arms. A 10-game feasibility probe at ps=8 / batch 1600 ran first and passed
(102 s), so the arms proceeded.

Note: the briefing for this run said 400 games per arm. The driver passed `--opening-pairs 100`,
which is 200 games. All numbers below are 200-game arms and the intervals are sized accordingly.

## Acceptance test: divergence rate — PASSED

The test is whether the arms actually play different chess, not whether the Elo moves. Reference
points from the batch-64 sweep: 100% divergence at 10 and 50 games (cap binding), 24% at 400 games
(cap largely inert).

| Contrast (batch 1600) | Games differing in `played_action_ids` |
|---|---|
| ps=2 vs ps=1 | **200 / 200 (100%)** |
| ps=4 vs ps=1 | **200 / 200 (100%)** |
| ps=8 vs ps=1 | **200 / 200 (100%)** |

Every game diverges. The cap binds, and unlike the batch-64 sweep **the Elo numbers below are
measuring something.**

### Control, and the limit of this test

At ps=1 the cap cannot bind at any batch size — `in_flight < 1` permits exactly one leaf per tree.
Yet comparing the ps=1 arm at batch 64 (from the earlier sweep) against the ps=1 arm at batch 1600
here, matched on opening and colour, **200 / 200 games also diverge**.

So divergence is not a clean instrument for "the cap was reached". Tree search is chaotic: any
perturbation of batch composition or ordering flips whole games. What divergence does measure is
whether the flag perturbs the search at all, and the informative contrast is that the *identical* ps
contrast produces 24% divergence at batch 64 and 100% at batch 1600. That is the evidence the cap is
now in play. It is not a direct measurement of in-flight depth; that would need instrumentation of
`task.in_flight`, which does not exist.

## Results

Elo is quoted against the 10,000-node rung anchor of **2470**.

| `parallel_searches` | W/D/L | Score | 95% score interval | Elo | Δ vs ps=1 | Divergence | Wall |
|---|---|---|---|---|---|---|---|
| 1 | 102/46/52 | 0.6250 | [0.5650, 0.6850] | 2558.7 | — | — | 1045 s |
| 2 | 107/41/52 | 0.6375 | [0.5825, 0.6925] | 2568.1 | **+9.4** | 100% | 692 s |
| 4 | 90/61/49 | 0.6025 | [0.5500, 0.6550] | 2542.2 | **−16.3** | 100% | 580 s |
| 8 | 96/51/53 | 0.6075 | [0.5550, 0.6600] | 2545.9 | **−12.7** | 100% | 498 s |

Per-arm 95% interval is about ±45 Elo. Each Δ carries a 95% interval of roughly **±60 Elo** from a
paired-by-opening comparison against the ps=1 arm (pairing buys almost nothing here, precisely
because 100% of games diverge).

Linear fit of Elo on log2(`parallel_searches`):

**−6.4 ± 4.7 Elo per doubling** (t = −1.38), i.e. **−19.3 Elo across 1 → 8**.

## Interpretation

- **The acceptance test passed and the sweep's headline is now superseded.** The batch-64 sweep's
  −0.76 ± 0.37 Elo per doubling was measured where the flag is inert. In the binding regime the
  slope is −6.4 ± 4.7, about **8x steeper**. The earlier number must not be quoted as the cost of
  leaf parallelism.
- **The effect is not statistically significant at this sample size.** t = −1.38 on the slope; the
  ps=8 Δ of −12.7 has a 95% interval of about [−73, +48]. What this run establishes is a
  *direction and a scale*, not a demonstrated effect.
- **It does not resolve `chess-search-findings-20260827.md` §3.1 either way.** That table puts ps=8
  at −45 Elo against ps=1. Our point estimate is −12.7 and our interval comfortably contains −45,
  so §3.1 is not contradicted; it is also not confirmed. Ruling in or out a −45 effect at 800
  searches needs roughly 2,000 games per arm, not 200.
- **The small-population estimates in the sweep's CORRECTION are not reproduced at full sample.**
  Those read −73 (10 games) and −49 (50 games) with per-arm SE of 112 and 50. At 200 games with the
  cap binding we read −12.7 with SE ~30. Consistent with all of them, and a reminder that the −73
  and −49 were noise-dominated point estimates that should never have been read as measurements.
- **Consequence for the 2799.1 / 2844.8 generation-936 numbers: the caveat narrows but does not
  close.** Both were measured at `parallel_searches` 1 and are undistorted. If the historical
  ~2,800 yardstick was measured at the native default of 16, this run bounds the plausible
  depression at roughly 4 doublings x 6.4 = ~26 Elo, with a wide interval — smaller than the "tens
  of Elo, possibly −45" the sweep's correction left open, but not negligible and not measured at 16.
- **Do not read the wall times as throughput.** They fall monotonically (1045 → 498 s, 2.1x from
  1 to 8), but batch 1600 against ~200 trees is a diagnostic configuration chosen to make the cap
  bind, not a configuration anyone would run. The inference pipeline is deliberately oversized here.
  For a throughput statement use a realistic batch; see `ladder-batching-rtx4070s-20260906`.

## Reproduce

Per arm, with `PS` in 1 2 4 8, from `py/`:

```bash
python -m tools.run_stockfish_gauntlet \
  --experiment configs/validation/vast-chess-4day-production-v29-resume.yaml \
  --run-directory /workspace/eval/run --checkpoint-generation 936 \
  --opening-manifest /workspace/evaluation-artifacts/chess/elite-200-openings.json \
  --stockfish-executable /workspace/alphazero-engine/engines/stockfish-13 \
  --stockfish-nodes 10000 --opening-pairs 100 --opening-selection prefix \
  --match-random-seed 20260816 --devices 0 --model-searches 800 --parallel-searches $PS \
  --inference-batch-size 1600 \
  --output-directory /workspace/eval/psrerun/ps$PS
```

Divergence is computed by matching games on `(opening_id, candidate_player)` and comparing
`played_action_ids` element-wise against the ps=1 arm.

## Provenance

- Tool: `py/tools/run_stockfish_gauntlet.py`, sha256
  `69d93a9ff22bb6396fab743cde918ba8b72b761a9dc5e54152b61459bca969d6`
- Checkpoint: v29 generation 936, inference only, `model_936.jit.pt` sha256
  `6b1d0bc7ba5f45199e44b85f0378098037803af52fca2ef47de673fbefc531c8`
- Opponent: Stockfish 13, sha256
  `ec56cd6ad04eecd38885912321ee2dfc76aee68415668ba433fa76eb7180ac5a`, 1 thread, 1024 MiB hash,
  10,000 fixed nodes
- Openings: `elite-200-openings.json`, sha256
  `40582c4f753e90d8ebd170498c37e3f4812bff38287e6bfa6db376e9eec70d39`, first 100 pairs by `prefix`.
  Committed as `py/reference/chess-elite-2025-11-balanced-4moves-200-v1-openings.json`; its
  `selection_sha256` is the sha256 of `py/reference/chess-elite-2025-11-balanced-4moves-200-v1.tsv`.
- Seeds: match 20260816, identical across all four arms and the probe
- Search budget: fixed 800 searches/move, `inference_workers` 1, **`inference_batch_size` 1600**,
  `outstanding_batches_per_worker` 1, `exploration_constant` 1.0
- Anchor: `melonimarco-ssdf-20260820`, Stockfish 13 fixed-node curve, 10,000 nodes = 2470
- Started 2026-09-06 14:26:14 UTC, all arms sequential on device 0, finished 15:13 UTC

## Files

- `probe-summary.json` — 10-game ps=8 / batch-1600 feasibility probe (2/1/7, 102 s)
- `ps1-summary.json`, `ps2-summary.json`, `ps4-summary.json`, `ps8-summary.json` — per-arm results
  with per-game action-id lists and duplicated per-shard game records stripped

Full results including `played_action_ids`, which the divergence numbers depend on, are under
`.codex-diagnostics/nodeB-evaluations/psrerun/`.
