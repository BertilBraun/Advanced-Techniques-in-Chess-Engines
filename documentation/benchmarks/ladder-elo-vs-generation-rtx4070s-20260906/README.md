# v29 strength against wall-clock, measured at 10,000 searches

Date: 2026-09-06. Ten checkpoints of the v29 run re-evaluated outside the training loop at a search
budget that makes the numbers mean something. Run on the production node's 8 idle GPUs immediately
after the run was stopped, before release.

## Why this exists

The in-run evaluation ladder plays at 64 searches per move. It read ~2,150 and flat for the last two
days, and that reading drove the decision to stop the run. This benchmark tests whether the plateau
was real or an artefact of a saturating instrument.

It was real. The instrument was also wrong.

## Result

| Generation | Wall-clock (h) | Ladder Elo | 95% interval | 64-search index |
|---|---|---|---|---|
| 100 | 2.7 | 2033.6 | [1800, 2353] | 1491 (4 rungs) |
| 200 | 6.0 | 2298.2 | [1877, 2579] | 1714 (3 rungs) |
| 300 | 9.7 | 2407.8 | [2002, 2628] | 1821 (4 rungs) |
| 400 | 14.0 | 2433.9 | [2180, one-sided] | 1852 (4 rungs) |
| 500 | 20.0 | 2614.0 | [2352, 2825] | 2028 (3 rungs) |
| 600 | 28.0 | 2693.1 | [2262, 2880] | 2091 (4 rungs) |
| 700 | 38.3 | 2693.1 | [2515, one-sided] | 2072 (4 rungs) |
| 800 | 53.3 | 2773.6 | [2630, one-sided] | 2137 (4 rungs) |
| 900 | 62.3 | **2827.5** | [2716, one-sided] | 2173 (4 rungs) |
| 1000 | 71.3 | 2800.5 | [2619, one-sided] | 2116 (3 rungs) |

Where the 50,000-node rung scores at or below ~0.35 the bootstrap has no constraint from above and
the upper bound runs to the clamp; those intervals are reported as one-sided. The lower bounds are
sound and carry the comparisons below.

## The plateau, quantified

- **First 28 h: 2034 -> 2693, +660 Elo, 26.1 Elo/h**
- **Last 43 h: 2693 -> 2800, +107 Elo, 2.5 Elo/h**

A **10x collapse** in Elo per wall-clock hour, breaking between generation 600 and 700. The run was
about 90% finished, in strength terms, in its first 28 hours. The remaining 60% of the wall-clock
bought the last 4% of the strength.

Generation 900 (2827.5) measures above generation 1000 (2800.5). The 27-Elo gap is inside noise at
40 games per point; the two are statistically indistinguishable.

## A second, independent cause: the generation rate collapsed

From the boundary summaries, seconds per generation over the run:

| Generation span | s/gen |
|---|---|
| 100 -> 200 | 118 |
| 300 -> 400 | 154 |
| 500 -> 600 | 282 |
| 700 -> 800 | 535 |
| 900 -> 1000 | 324 |

A 4.5x slowdown by generation 800, from the 19x176 model promotion and 1000-visit self-play. The
first 28 h bought 600 generations; the next 43 h bought 400. The drop back to ~325 s/gen at the end
is the mid-run visit cut from 1000 to 600 taking effect.

So "Elo per hour collapsed" and "generations per hour collapsed" are largely the same fact. Elo per
*generation* held up far better than Elo per hour, which points the next run at throughput rather
than at the training recipe.

## What this says about the 64-search in-run ladder

It understated the model by **550-600 Elo at every point** (1491 vs 2034 at generation 100; 2116 vs
2800 at generation 1000). Its *shape* was roughly right — it also flattens after generation 600 —
so it was serviceable as a relative regression signal and useless as a level. Every absolute Elo
quoted from the in-run logs during v29 is wrong by that margin.

It also saturated: rungs retire as the model outgrows them, and the fit was down to 3 rungs against a
table topping out at 10,000 nodes = 2470 anchor, while the model measured 2800.

## Resolution warning

Generation 400 gains +26 Elo and generation 700 gains exactly 0 — local flat spots inside a rising
curve. At 40 games per point this ladder cannot resolve steps below roughly 80 Elo. Most changes
proposed in `documentation/analysis/reference-recipes-for-a-compute-poor-run.md` have plausible
effect sizes below that, so measurement resolution is a binding constraint on improving the recipe,
not only on reporting it.

## Provenance

- Source revision: `555ffb44`; checkpoints from run `vast-chess-4day-production-v29`
- Ladder: `py/tools/run_stockfish_ladder.py`, rungs 300 / 2,000 / 10,000 / 50,000 nodes
  (anchors 1400 / 1890 / 2470 / 2960), 10 games per rung, 40 games per checkpoint
- Budget: 10,000 searches/move, `parallel_searches` 1, `inference_batch_size` 64
- Opponent: Stockfish 13 fixed nodes; openings `chess-stockfish-8moves-v3-openings-v1.json`
- Anchors: `melonimarco-ssdf-20260820`
- Node: 8x RTX 4070 SUPER, all ten ladders run concurrently, round-robin over the GPUs
- Raw results: `gen*/ladder-result.json`

**Not comparable to** `ladder-elo-generation936-rtx4070s-20260906`, which used rungs
20,000-200,000. Fits over different rung sets are not comparable; these ten points are internally
consistent with each other only. Generation 936 read 2799.1 on that other rung set, and generation
1000 reads 2800.5 here — the agreement is reassuring but coincidental in its precision.
