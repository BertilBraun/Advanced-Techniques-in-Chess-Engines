# Stockfish ladder batching on one RTX 4070 SUPER

Date: 2026-09-06. Measures what running every ladder rung in one batched population buys, and what the
inference and parallelism knobs cost, on the checkpoint already measured at
`ladder-elo-generation936-rtx4070s-20260906`.

## What changed

`py/src/evaluation/match.py` gained `run_concurrent_matches`: several match groups share one candidate
selector, so a single native search request spans every rung. `py/tools/run_stockfish_ladder.py` hands all
five rungs to `run_gauntlets` at once instead of looping over them. Timed budgets stay sequential.

## Arms

Five rungs (1,000 / 2,000 / 5,000 / 10,000 / 20,000 Stockfish 13 nodes), 10 games each, 800 searches per
move, opening seed 20260815, match seed 20260816, one GPU. Wall time is the whole `run_stockfish_ladder`
command; the sequential baseline is the sum of its five per-rung `duration_seconds`, which each already
include shard spawn and model load.

| Arm | parallel | batch / outstanding | Wall | vs sequential | GPU power mean | GPU memory | Ladder Elo |
|---|---|---|---|---|---|---|---|
| sequential (pre-change) | 1 | 64 / 1 | **604.5 s** | 1.00x | ~102 W | 429 MiB | 2564.8 [2387.9, 2725.4] |
| concurrent | 1 | 64 / 1 | **222.8 s** | **2.71x** | 127.8 W | 429 MiB | 2626.5 [2518.1, 2780.6] |
| concurrent (repeat) | 1 | 64 / 1 | 224.3 s | 2.70x | 126.4 W | 429 MiB | 2626.5 [2518.1, 2780.6] |
| concurrent | 1 | 320 / 2 | 308.8 s | 1.96x | 149.1 W | 499 MiB | 2642.6 [2495.4, 2814.6] |
| concurrent | 4 | 64 / 1 | **123.4 s** | **4.90x** | 148.1 W | 429 MiB | 2535.2 [2381.3, 2683.7] |
| concurrent | 4 | 320 / 2 | 161.7 s | 3.74x | 147.3 W | 499 MiB | 2535.2 [2275.5, 2770.5] |

Power limit on this card is 170 W, not 200 W. Per-game mean wall drops from 60.5 s (sequential, 50 games in
604.5 s) to 22.3 s at parallel 1 and 12.3 s at parallel 4.

## The scheduling change is exact

The 2,000-node rung run alone under the new code, at the sequential arm's inference settings, reproduces the
pre-change games **bit for bit** — identical `played_action_ids` in all ten games, identical 9/0/1, 104.4 s
against 105.0 s. The concurrent arm is also deterministic: the repeat run reproduces every ply count and the
Elo to all printed digits.

## Batch shape reshuffles outcomes

The Elo differences above are *not* caused by the scheduling. They are caused by the inference batch shape,
which the network's output depends on in the last bits. Isolated, with no concurrency involved at all:

| 2,000-node rung, alone, parallel 1 | Ply counts | Result |
|---|---|---|
| batch 64 / 1 | 89, 56, 87, 98, 147, 110, 115, 78, 52, 104 | 9/0/1 |
| batch 320 / 2 | 87, 84, 87, 70, 91, 134, 67, 86, 91, 90 | 10/0/0 |

Same seeds, same openings, same 800 searches, same parallel 1 — only `--inference-batch-size` differs. Batching
five rungs together does the same thing for the same reason: it changes how many rows go through the network
per call. Search per game is unchanged; the sample of game outcomes is not. Two ladder runs are comparable
game-for-game only when their batch shapes match, and a 10-game rung carries a ±150-200 Elo interval anyway.

## Batch size 320 is the wrong default for evaluation

Raising the cap to the self-play value costs wall time in every arm tried: +38.6% at parallel 1 (222.8 → 308.8 s),
+31.0% at parallel 4 (123.4 → 161.7 s), +34.4% on a single rung alone (102.3 → 137.5 s). Power rises with it, so
the extra watts are padding, not work: evaluation keeps only a few dozen leaves in flight, and the graph capture
buckets are `maximumBatchSize * k / 16`, so a cap of 320 charges at least 20 rows for a batch of 5. Batch 64 with
one outstanding batch stays the tool default.

## CUDA graph capture is active in the evaluation path

`cpp/src/search/InferencePipeline.cpp` captures unconditionally on CUDA at `InferenceRunner` construction, and the
evaluation search uses the same pipeline as self-play. Confirmed by disabling it on the same rung:

| 2,000-node rung, alone, parallel 1, batch 64 | Wall | GPU power | GPU memory | GPU util |
|---|---|---|---|---|
| graphs captured | 102.3 s | 101.8 W | 429 MiB | 90% |
| `ALPHAZERO_DISABLE_INFERENCE_GRAPHS=1` | 262.1 s | 60.5 W | 341 MiB | 42% |

Capture is worth **2.56x** here and costs 88 MiB. No `Inference graph capture unavailable` line appeared in any
run log. Both arms produced identical ply counts, so graph replay is numerically identical to the eager path at
the same batch shape — the variable batch is handled by bucket rounding, not by falling back to eager.

## Parallel searches

Parallel 4 is 1.81x faster than parallel 1 at batch 64 (123.4 s against 222.8 s) and shifts the measurement:
2535.2 against 2626.5, with the 20,000-node rung dropping from 3/3/4 to 2/4/4. That direction agrees with
`documentation/analysis/chess-search-findings-20260827.md` §3.1, which measures roughly −36 Elo for parallel 4
against parallel 1 at 600 visits. **No measurement in this repository supports the idea that the cost vanishes
above 1,000 searches per move**; §3.1 was taken at 600 visits, the effective parallelism is governed by
`min(parallel_searches, inference_capacity / active_trees)` rather than by the visit budget, and the self-play
heuristic in `cpp/src/search/SearchTypes.hpp` *raises* parallelism to 8 at 1,000 visits and above. The tool
default of 4 is a throughput decision taken with that cost known, not a claim that the cost is absent.

## Provenance

- Source revision: `43c4638725362c551171f144825c806a51d7324e`
- Tools: `py/tools/run_stockfish_ladder.py` sha256
  `dae719cf07e477260efabdf4a72d1486b1cd5604c478230f725905664b69dff7`;
  `py/tools/run_stockfish_gauntlet.py` sha256
  `69d93a9ff22bb6396fab743cde918ba8b72b761a9dc5e54152b61459bca969d6`
- Experiment: `py/configs/validation/vast-chess-4day-production-v29-resume.yaml`
- Checkpoint: v29 generation 936, inference only
- Opponent: Stockfish 13, sha256 `ec56cd6ad04eecd38885912321ee2dfc76aee68415668ba433fa76eb7180ac5a`
- Openings: `chess-stockfish-8moves-v3-openings-v1.json`, sha256
  `4bfdf0d97b38499d38f51e0044c06ad977e5fef9c9c8244ffa79f92e1ffd56cb`
- Node: 82.225.150.130:15801, 1x RTX 4070 SUPER 12,282 MiB (170 W cap), driver 595.84, 48 CPUs, 188 GiB RAM
- Ladder results for every arm are stored beside this file as `ladder-result-<arm>.json`. Run logs and the 2 s
  `nvidia-smi` samples stay on the node under `/workspace/eval/bench-*`; that filesystem is ephemeral.
