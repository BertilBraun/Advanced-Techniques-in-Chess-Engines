# Ladder Elo of v29 generation 936 at 800 searches

Date: 2026-09-06. First strength measurement of the v29 model taken outside the training loop, at a
search budget that makes the number meaningful.

## Why this exists

The in-run evaluation ladder plays at **64 searches per move**. That budget was chosen as a cheap
per-generation regression signal, and it is fine for that. It is not a strength measurement: it reads
~2,160 for a model that measures **2,565** at 800 searches. Every absolute Elo quoted from the
training logs before this benchmark is roughly 400 Elo low. Trends measured *within* the 64-search
budget remain valid; levels do not.

## Result

| Stockfish 13 rung (nodes) | Anchor Elo | W/D/L | Score |
|---|---|---|---|
| 1,000 | 1700 | 10/0/0 | 1.00 |
| 2,000 | 1890 | 9/0/1 | 0.90 |
| 5,000 | 2220 | 9/1/0 | 0.95 |
| 10,000 | 2470 | 5/3/2 | 0.65 |
| 20,000 | 2700 | 1/4/5 | 0.30 |

**Ladder Elo 2564.8, bootstrap 95% interval [2387.9, 2725.4]** over 20,000 resamples.
Score bracket 10,000–20,000 nodes; closest rung 10,000.

The interval is wide because each rung is 10 games. This is the probe stage: locate cheaply, then
confirm with a 200-game gauntlet at the bracketing rung.

The 20,000-node rung was added to the anchor table in `e856ba74` for this measurement. It was
load-bearing: the 10,000 rung scored 0.65, so the crossing lies above 2470 and the previous table,
which stopped at 10,000 = 2470, could not have produced a bracket at all.

## Not comparable to the four-day yardstick

The ≈2,800 ladder Elo of the four-day run (r3/r4) was measured **at 10k visits**
(`chess-recovery-plan-20260820.md`, line 10). This benchmark is at 800 searches. Do not place the two
numbers side by side. A comparison against that yardstick requires re-running this ladder at 10,000
searches.

## Provenance

- Source revision: `e856ba749f8bf97bd6683e06cbe75800c61cedf2`
- Tool: `py/tools/run_stockfish_ladder.py`, sha256 `0f0a84f1bdf865cd6c6edd54a347904fff51ce4db1f873d0888fb2205424d844`
- Checkpoint: v29 generation 936 (retained; `model_936.jit.pt`, inference only)
- Opponent: Stockfish 13, sha256 `ec56cd6ad04eecd38885912321ee2dfc76aee68415668ba433fa76eb7180ac5a`
- Openings: `chess-stockfish-8moves-v3-openings-v1.json`, sha256 `4bfdf0d97b38499d38f51e0044c06ad977e5fef9c9c8244ffa79f92e1ffd56cb`
- Seeds: opening selection 20260815, match 20260816; 10 games per rung
- Anchors: `melonimarco-ssdf-20260820`, Stockfish 13 fixed-node curve
- Search budget: fixed 800 searches/move, `parallel_searches` 1, inference batch 64,
  **`exploration_constant` 1.0** — the tool default, not the 1.5 used in v29 self-play
- Node: 1x RTX 4070 SUPER 12 GB, driver 595.84, Threadripper PRO 5965WX (6.6 effective CPUs),
  188 GiB RAM; torch 2.12.1+cu126, cuDNN 91002
- Raw result: `ladder-result-800-searches.json`

## Cost

~40 s per game at 800 searches, games overlapping on one GPU. The five-rung, 50-game probe took
~25 minutes. A 200-game gauntlet at one rung is ~2 h at this budget and scales roughly with search.
