# Supervised testbed on the converted four-day replay — 1× RTX 4070, 2026-08-21 (WP4, trimmed)

Node and provenance: Phase A test node (`node-provisioning-phase-a-test-20260821.md`), source `master` @
`631d4b53` (run at `phase-a` 0d548d6f-era tree, identical py/ content), tool
`py/tools/benchmark_supervised_testbed.py`, configuration `vast-chess-8gpu-run1.yaml` with stand-in topology
(r3 objective: policy + WDL + next_policy 0.1 + remaining_game_length 0.1). Raw report:
`.codex-diagnostics/wp4-testbed-20260821/testbed-fourday-report.json`.

Data: the four-day freeze store and the three-day full-state store, converted from the 703-byte r3 row format
to the current 1,577-byte layout (`convert_r3_replay_store.py`, session scratchpad — legal-action lists did not
exist in the old format and were filled with the policy target support; identical approximation for every cell).
Train: 1.5 M rows, uniform sampling, mirror augmentation on, batch 2048. Held-out: 4,096 fixed rows from the
three-day store. ~2,500 optimizer steps per cell (10-minute cap; the plan's 8-epoch point is ≈5,860 steps).

| cell | params | steps | held-out policy CE init → best | grad-norm max after 200 | samples/s |
|---|---:|---:|---|---:|---:|
| dense-cnn 12×112 @5e-3 (r3 control) | 5.11 M | 2,518 | 773.73 → 2.0824 | 0.383 | 8,593 |
| plane-cnn 12×112 @5e-3 (WP1 head) | 2.68 M | 2,505 | 142.60 → 2.1828 | 0.354 | 8,548 |
| attention 8×128 @1e-3, warm-up 1k | 1.14 M | 2,505 | 2.5235 → 2.2158 | 2.651 | 8,545 |
| attention 8×128 @2e-3, warm-up 1k | 1.14 M | 2,495 | 2.5235 → 2.2089 | 2.302 | 8,513 |
| attention 8×128 @5e-3, warm-up 1k | 1.14 M | 2,494 | 2.5235 → 2.2124 | 1.235 | 8,508 |

Readings (assessment; the run decision is the user's):

1. **Initialisation.** The attention cells start at CE 2.52 from step 0 — the final-LayerNorm + 0.02/small-init
   scheme lands initialised, no warm-up cliff. The CNN initial values (774 dense, 143 plane) are an eval-mode
   BatchNorm artifact (identity running stats at init); the WP1 generation-0 guard measures train-mode, where
   plane-head logit std is 0.1 (`init_probe`). Even under the artifact the plane head starts 5.4× lower than dense.
2. **Plane vs dense head:** 2.1828 vs 2.0824 at ~2,500 equal steps = **4.8 % gap**, outside the plan's 2 % —
   but at half the plan's step budget and still closing (plane fell 0.012 over the last 500 steps, dense 0.005).
   The dense control also carries 2.4 M extra head parameters (its aux next_policy head is dense too).
   Extend to the full 5,900 steps before a final verdict.
3. **Attention:** all three LRs converge stably; no divergence anywhere. Grad-norm peaks above 2 (2.65 @1e-3,
   2.30 @2e-3) occur in the first ~250 steps *inside the 1,000-step warm-up ramp* and settle to ≤0.7 after;
   @5e-3 never exceeds 1.24. At 1.14 M params the best attention cell (2e-3: 2.2089) is within **1.2 %** of the
   2.68 M plane CNN — comfortably inside the plan's "within 5 % of the equal-parameter CNN" trajectory.
4. **Trainer loader ceiling:** all five cells train at ≈8.5 k samples/s regardless of model size (1.1 M vs
   5.1 M identical) — single-process batch building, not the GPU, is the cap on this 16-CPU node. Treat the
   4070 trainer numbers in `throughput-history-chess-20260821` as a floor, and re-measure DDP samples/s on the
   8-GPU node before concluding anything about trainer scaling.
5. Caveats: converted-data legal mask = target support (symmetric across cells); held-out is the three-day
   store only (the Stockfish fixed dataset no longer exists locally); 2,500 of 5,860 planned steps.
