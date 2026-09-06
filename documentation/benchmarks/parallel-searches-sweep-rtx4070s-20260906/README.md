# Elo cost of `parallel_searches` in evaluation, 1 to 16

Date: 2026-09-06. Measures what leaf parallelism costs in playing strength, and what it buys in
throughput, at an evaluation-sized match population.

## Why this exists

`chess-search-findings-20260827.md` §3.1 reports −20 / −36 / −45 Elo for `parallel_searches`
2 / 4 / 8 against 1, measured at 600 visits in self-play conditions and explicitly flagged as
"point estimates, not significant at 200 games". Those numbers were the largest caveat against the
generation-936 ladder Elo of 2799.1, which ran at `parallel_searches` 1 while the native default at
a 10k visit budget is 16 (`searchParallelism` doubles to the cap). This sweep settles the size of
that correction.

## Result

Five arms, 400 games each, identical 200 openings and match seed, 800 searches per move, against
Stockfish 13 at 10,000 fixed nodes.

| `parallel_searches` | W/D/L | Score | Elo vs rung | Δ vs 1 | Wall | Speedup |
|---|---|---|---|---|---|---|
| 1 | 211/99/90 | 0.6512 | +108.5 | — | 1122 s | 1.00x |
| 2 | 210/100/90 | 0.6500 | +107.5 | −1.0 | 1079 s | 1.04x |
| 4 | 208/101/91 | 0.6462 | +104.7 | −3.8 | 1065 s | 1.05x |
| 8 | 209/100/91 | 0.6475 | +105.6 | −2.9 | 1048 s | 1.07x |
| 16 | 209/100/91 | 0.6475 | +105.6 | −2.9 | 1040 s | 1.08x |

**Slope −0.76 ± 0.37 Elo per doubling** (t = −2.05), i.e. **−3.0 Elo across the whole 1 → 16 range**
against −45 in the older table. Per-arm SE is 18.2 Elo; the entire five-arm spread is 3.8 Elo.

The flag is genuinely taking effect: 94 of 400 games differ in move sequence between the 1 and 2
arms. The near-identical scorelines are 94 different games landing on the same result, not a no-op.

## Reading

- **The effect is flat, not monotone-declining.** The 8 arm sits above the 4 arm, so there is no
  clean ordering; the honest statement is a small downward tilt inside noise.
- **This does not refute §3.1.** That was 600 visits in self-play with a different contention
  regime. The documented mechanism is `min(parallel_searches, inference_capacity / active_trees)`,
  which predicts the cost depends on pipeline contention — and 400 concurrent games is far more
  contended than the setting §3.1 measured.
- **Arms 8 and 16 produced identical aggregates** (209/100/91). Effective parallelism plausibly
  saturates below 16 once 400 trees compete for inference capacity, so 16 may not differ from 8 in
  practice here.
- **Throughput gain is 8% across a 16x range**, because 400 concurrent games already fill the
  inference batches. Leaf parallelism only matters when the game population cannot. Compare
  `ladder-batching-rtx4070s-20260906`, which measured 4.9x from parallel 4 at 50 games.

## CORRECTION: this result is regime-specific and does NOT close the 2799 caveat

The sweep above ran 400 concurrent games. `parallel_searches` is an upper cap on in-flight leaves
per tree, enforced at `cpp/src/search/SearchExecutor.hpp:719` with trees served round-robin, so a
tree gets a second concurrent leaf only after the scheduler has cycled every other schedulable tree.
With ~200 trees on turn and a batch of 64, the batch fills from 64 distinct trees at one leaf each
and **the cap is never reached**, except in the tail of each move's search.

Divergence rate proves it, and the Elo cost tracks it:

| Games | ps=1 Elo | ps=8 Elo | delta | games differing | per-arm SE |
|---|---|---|---|---|---|
| 10 | +107.5 | +34.9 | **-72.7** | **10/10 (100%)** | 112 |
| 50 | +34.9 | -13.9 | **-48.8** | **50/50 (100%)** | 50 |
| 400 | +108.5 | +105.6 | -2.8 | 94/400 (24%) | 18 |

At 10 and 50 games every game diverges: the cap binds on every move. At 400 games only 24% diverge.
So the -0.76 Elo/doubling slope measures a regime where the flag is largely inert, and must not be
generalised. The small-population point estimates (-73, -49) are individually inside their noise but
**corroborate** the -45 in `chess-search-findings-20260827.md` rather than contradicting it.

Consequences:

- **Do not run small probe ladders at high `parallel_searches` on the strength of this benchmark.**
  That is the regime where the cost is largest, not smallest.
- **The 2799 caveat is NOT closed.** The generation-936 ladder ran at `parallel_searches` 1 and is
  itself undistorted, but its population was 40 games — inside the binding regime. If the historical
  ~2800 was measured at the native default of 16 at a similarly small population, it could be
  depressed by tens of Elo, biasing the comparison in our favour.
- Closing it properly needs Elo measured at **small concurrency with a large total game count**
  (many sequential small matches), which no run so far provides.

### SUPERSEDED by the batch-1600 rerun

`parallel-searches-rerun-batch1600-rtx4070s-20260906` reruns arms 1/2/4/8 with
`inference_batch_size` 1600 (8 x ~200 trees) so the cap binds. **All three contrasts against ps=1
diverge in 200/200 games**, and the slope there is **−6.4 ± 4.7 Elo per doubling**, about 8x the
−0.76 measured above. Do not quote the −0.76 slope as the cost of leaf parallelism; it measures an
inert flag. The rerun's own slope is not significant (t = −1.38), so the honest summary is
"direction and scale, not a demonstrated effect".

That rerun also shows divergence is a blunter instrument than assumed: at ps=1, where the cap cannot
bind at any batch size, changing the batch from 64 to 1600 still flips 200/200 games.

## Provenance

- Source revision: `2b6e7186`
- Openings: `elite-200-openings.json`, built from
  `py/reference/chess-elite-2025-11-balanced-4moves-200-v1.tsv` (Lichess Elite 2025-11, balanced
  4-move), sha256 `40582c4f753e...`, 200 openings, 100 pairs played both colours = 400 games.
  The built manifest is committed as
  `py/reference/chess-elite-2025-11-balanced-4moves-200-v1-openings.json`; the build needs a
  specific Stockfish binary, so the manifest is the reproducible artifact, not the command.
- Opponent: Stockfish 13 at 10,000 fixed nodes; match seed 20260816; prefix opening selection
- Checkpoint: v29 generation 936, inference only
- `inference_batch_size` 64, `outstanding_batches_per_worker` 1
- Node: 1x RTX 4070 SUPER 12 GB, driver 595.84, Threadripper PRO 5965WX (6.6 effective CPUs)
- Per-arm summaries: `ps{1,2,4,8,16}-summary.json`
