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

---

# Same checkpoint at 10,000 searches

Run after the ladder batching rework (`2b6e7186`), all four rungs played in one concurrent
population, `parallel_searches` 1, `inference_batch_size` 64.

| Stockfish 13 rung (nodes) | Anchor Elo | W/D/L | Score |
|---|---|---|---|
| 20,000 | 2700 | 3/4/3 | 0.50 |
| 50,000 | 2960 | 0/5/5 | 0.25 |
| 100,000 | 3100 | 2/0/8 | 0.20 |
| 200,000 | 3230 | 0/4/6 | 0.20 |

**Ladder Elo 2799.1, bootstrap 95% interval [2644.7, 2979.7]**. Bracket 20,000-50,000,
closest rung 20,000. Raw result: `ladder-result-10000-searches.json`.

## Comparison to the four-day yardstick

The four-day run (r3/r4) reached ~2,800 ladder Elo at 10k visits
(`chess-recovery-plan-20260820.md` line 10). This measurement is at the same search budget and
reads 2799.1 at 69.3 h of cumulative run time, against roughly 96 h for the four-day run.

Do not treat that as a settled win. Four caveats, largest first:

1. **`parallel_searches` — STILL OPEN, and possibly larger than first thought.** A five-arm sweep
   at 400 concurrent games measured only -0.76 +/- 0.37 Elo per doubling, but that population is
   large enough that the per-tree in-flight cap is never reached (24% of games diverge). At 10 and
   50 games, where it binds on every move (100% divergence), the cost measures -73 and -49 Elo. This
   ladder's population was 40 games, inside the binding regime. It ran at 1 so it is undistorted,
   but if the historical ~2800 was measured at the native default of 16, that number may be
   depressed by tens of Elo and the comparison is biased in our favour. See
   `parallel-searches-sweep-rtx4070s-20260906`, CORRECTION section. Original text follows.

   This ladder ran at 1. The native default derives
   parallelism from the visit budget: `searchParallelism(10000)` yields targetRounds 50 and
   doubles to the cap of **16**. If the historical number was measured at the default, it was
   measured at 16 while this one was at 1, and `chess-search-findings-20260827.md` §3.1 puts the
   cost of parallel 8 at -45 Elo against parallel 1. The sweep at 800 searches across
   `parallel_searches` 1/2/4/8/16 quantifies this directly.
2. **The interval is +/-170** and the rungs disagree internally: the 20,000 rung alone implies
   2700, the 200,000 rung implies ~2990. Ten games per rung is thin.
3. **This lineage is not from-scratch.** v29 has been through checkpoint resumes, the adaptive
   stopping fork, and a mid-run visit-schedule change from 1000 to 600. The four-day run is
   described in the plan as a clean from-scratch result.
4. **Budget dependence is large.** The same weights read 2157 on the in-run 64-search ladder and
   2565 at 800 searches. Every quoted Elo must carry its search budget.

## Elo against search budget (this checkpoint, `parallel_searches` 1)

| Searches/move | Ladder Elo | Interval |
|---|---|---|
| 64 (in-run index) | ~2157 | n/a, 3-rung fit |
| 800 | 2564.8 | [2387.9, 2725.4] |
| 10,000 | 2799.1 | [2644.7, 2979.7] |

Roughly +64 Elo per doubling from 800 to 10,000, flattening from the 64 -> 800 segment.
