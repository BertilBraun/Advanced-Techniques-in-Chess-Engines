# Deep 200-game match, generation 936 at 10,000 searches vs Stockfish 13 / 50,000 nodes

| | |
| --- | --- |
| `experiment_configuration_sha256` | `8f5c82e5ad316328d2ed7dd6703eec866704b95f7b6aef29627ea9e087e08ab4` |
| Source revision | `2b6e7186d0e0994a161f04e2b3b357c34302969b` (clean) |
| Node | `82.225.150.130:15801`, 1x RTX 4070 SUPER 12 GB, driver 595.84, AMD Threadripper PRO 5965WX, 6.58 effective CPUs of 48 visible, 188 GiB RAM |
| Date | 2026-09-06 |

Confirmation match at one rung, to replace the ±170 Elo ladder interval with a
tight one and to settle the rung where the anchor curve and the ladder probe disagreed most.

## Why this exists

`ladder-elo-generation936-rtx4070s-20260906` fits **2799.1, 95% [2644.7, 2979.7]** from four rungs
of 10 games each. Two problems with that number:

- The interval is ±170 Elo, which is too wide to answer whether v29 has reached the four-day
  yardstick of ~2,800.
- The rungs disagree internally. The 50,000-node rung scored 0.25 in 10 games, implying ~2,770
  against its 2960 anchor, while the 20,000-node rung scored 0.50, implying 2700, and the
  200,000-node rung implied ~2,990. Ten games per rung cannot separate these.

This match plays the 50,000-node rung 200 times.

## Result

200 games, 100 opening pairs played both colours, 10,000 searches/move, `parallel_searches` 1,
against Stockfish 13 at 50,000 fixed nodes (anchor **2960**).

| W/D/L | Score | 95% score interval | Elo vs 2960 anchor | Wall |
|---|---|---|---|---|
| 32/72/96 | 0.3400 | [0.2925, 0.3900] | **2844.8, [2806.6, 2882.3]** | 7407 s (2.06 h) |

Split by colour: 0.33 as first player, 0.35 as second. Mean game length 126 plies. The interval is
the tool's pair-clustered bootstrap; a normal approximation over the 100 pair scores
(SE 0.0245) gives [2806.1, 2880.8], so the two agree to about 1 Elo.

**±38 Elo**, against ±170 for the ladder fit. The target of ±30 was not quite reached — 200 games at
a 0.34 score is worth about ±38 — but the reduction is a factor of 4.5.

## Reading

- **The two estimates are compatible.** 2844.8 sits comfortably inside the ladder's
  [2644.7, 2979.7]. The ladder's 2799.1 falls 7 Elo below this match's lower bound of 2806.6, which
  is not evidence of a conflict: the match interval covers only its own sampling error, not the
  ladder's, and the ladder interval is the wide one. The honest statement is that the ladder point
  estimate was ~45 Elo low and the two measurements cannot be separated.
- **These are different estimators and must be labelled as such.** The ladder fit interpolates a
  logistic across four anchor rungs and inherits the error of every rung and of the anchor curve
  itself. This match is a single-rung point estimate that assumes the 2960 anchor is correct and
  measures only the score against it. A ladder fit is robust to one bad anchor; a single-rung match
  is not. Do not present 2844.8 as "a more accurate 2799.1" — it is a different quantity with a
  narrower sampling error and a fully undiagnosed anchor error.
- **The opening books differ too.** The ladder used
  `chess-stockfish-8moves-v3-openings-v1.json` (8-move openings); this match used
  `elite-200-openings.json` (Lichess Elite 2025-11, balanced 4-move). A shallower book leaves more
  of the game to the engines and is not guaranteed to be the same difficulty. Some of the 45 Elo gap
  may be book, not estimator.
- **The 50,000-node rung is not anomalous after all.** The 10-game probe read 0.25 there; 200 games
  read 0.34. 0.25 is inside the sampling noise of a 10-game match at a true 0.34 (SE ≈ 0.11), so the
  apparent disagreement with the anchor curve was thin data, not a broken rung.
- The `parallel_searches` caveat on the ladder number applies unchanged here: this match ran at 1,
  so it is undistorted, but if the historical ~2,800 yardstick was measured at the native default of
  16 at a small population, that comparison is still biased in our favour. See
  `parallel-searches-sweep-rtx4070s-20260906`, CORRECTION section.

## Provenance

- Source revision: `2b6e7186d0e0994a161f04e2b3b357c34302969b`
- Tool: `py/tools/run_stockfish_gauntlet.py`, sha256
  `69d93a9ff22bb6396fab743cde918ba8b72b761a9dc5e54152b61459bca969d6`
- Experiment configuration: `py/configs/validation/vast-chess-4day-production-v29-resume.yaml`,
  resolved sha256 `8f5c82e5ad316328d2ed7dd6703eec866704b95f7b6aef29627ea9e087e08ab4`
- Checkpoint: v29 generation 936, inference only,
  `model_936.jit.pt` sha256
  `6b1d0bc7ba5f45199e44b85f0378098037803af52fca2ef47de673fbefc531c8`
- Opponent: Stockfish 13, sha256
  `ec56cd6ad04eecd38885912321ee2dfc76aee68415668ba433fa76eb7180ac5a`, 1 thread, 1024 MiB hash,
  50,000 fixed nodes
- Openings: `elite-200-openings.json`, sha256
  `40582c4f753e90d8ebd170498c37e3f4812bff38287e6bfa6db376e9eec70d39`, 200 available, first 100 pairs
  taken by `prefix` selection. Committed as
  `py/reference/chess-elite-2025-11-balanced-4moves-200-v1-openings.json`; its `selection_sha256`
  is the sha256 of `py/reference/chess-elite-2025-11-balanced-4moves-200-v1.tsv`.
- Seeds: match 20260816
- Search budget: fixed 10,000 searches/move, `parallel_searches` 1, `inference_workers` 1,
  `inference_batch_size` 64, `outstanding_batches_per_worker` 1, `exploration_constant` 1.0
- Anchor: `melonimarco-ssdf-20260820`, Stockfish 13 fixed-node curve, 50,000 nodes = 2960
- Node: 1× RTX 4070 SUPER 12 GB, driver 595.84, AMD Threadripper PRO 5965WX (6.58 effective CPUs of
  48 visible), 188 GiB RAM
- Started 2026-09-06 12:19:54 UTC, one shard on device 0
- Raw result (per-game action ids stripped): `deep50k-summary.json`; full result under
  `.codex-diagnostics/nodeB-evaluations/deep50k/`

## Reproduce

```bash
python -m tools.run_stockfish_gauntlet \
  --experiment configs/validation/vast-chess-4day-production-v29-resume.yaml \
  --run-directory /workspace/eval/run --checkpoint-generation 936 \
  --opening-manifest /workspace/evaluation-artifacts/chess/elite-200-openings.json \
  --stockfish-executable /workspace/alphazero-engine/engines/stockfish-13 \
  --stockfish-nodes 50000 --opening-pairs 100 --opening-selection prefix \
  --match-random-seed 20260816 --devices 0 --model-searches 10000 --parallel-searches 1 \
  --output-directory /workspace/eval/deep50k
```

## Files

- `deep50k-summary.json` — the tool's `result.json` with per-game action-id lists and the duplicated
  per-shard game records stripped; every provenance field and the aggregate are intact.

## Cost

7407 s for 200 games at 10,000 searches on one GPU, i.e. ~37 s of wall per game with the population
overlapping. Longest single game 6006 s of tree time. Budget ~2 h for a 200-game confirmation at
this search budget.
