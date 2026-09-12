# v34 generation 1465 terminal chess strength

The retained three-day v34 checkpoint reached **3,037 Stockfish-13 ladder Elo at 10,000 searches per move** and
**3,167 at 80,000 searches per move**. These are conditional benchmark ratings on this project’s historical
SSDF-derived fixed-node scale. They are not FIDE ratings and cannot be compared point-for-point with a human or
another engine list.

Both headline results use 400 games from 200 balanced opening pairs. The paired-bootstrap 95% intervals are
[3,012, 3,061] and [3,143, 3,193], conditional on treating the Stockfish-node anchors as exact.

## Provenance

| Field | Value |
| --- | --- |
| Checkpoint | v34 generation 1465, 14x160 convolutional network |
| Inference model SHA-256 | `402efb61146b5a0f569c960e1f7a7e7714ba1bb28982fcdfb7f10e8ec5a98ad6` |
| Original final-suite source revision | `a87a186e72862d6f5ff845258007b2ff4246d521` |
| Original final-suite experiment SHA-256 | `5a1194975e225d4c5ebc329ca7641b205e7cd17b88d5b4aa24679d9e6a939c9a` |
| 100,000-node confirmation source revision | `92c9f487d38467ac8c34bb88a28604e3a539e15f` |
| 100,000-node confirmation experiment SHA-256 | `23e391f81cba4f4b8463a0867f13d3555f073e2152aa91c50bdd4eed09c4c3d1` |
| Opening manifest | 200 paired four-move openings |
| Opening manifest SHA-256 | `40582c4f753e90d8ebd170498c37e3f4812bff38287e6bfa6db376e9eec70d39` |
| Opponent | Stockfish 13, one thread, 1,024 MiB hash, fixed nodes per move |
| Stockfish executable SHA-256 | `ec56cd6ad04eecd38885912321ee2dfc76aee68415668ba433fa76eb7180ac5a` |
| Hardware | 8x NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB each; driver 595.71.05 |
| Match seed | 20260816 |

The node's billed rate was $17.36 per day. The three-day checkpoint therefore cost **$52** in rounded
training-node rental. That figure excludes separate evaluation nodes and abandoned or restarted run segments; it
is not a complete project invoice.

## Final matches

The candidate W/D/L is listed from v34’s perspective. Each row used the same 200 paired openings. All four matches
were launched concurrently and sharded over all eight GPUs.

| Candidate mode | Parallel searches | Stockfish nodes | W/D/L | Score [95% CI] | Benchmark Elo [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Policy only | none | 1,000 | 154/62/184 | 0.4625 [0.4175, 0.5075] | 1,674 [1,642, 1,705] |
| 64 searches | 1 | 5,000 | 186/80/134 | 0.5650 [0.5225, 0.6088] | 2,265 [2,236, 2,297] |
| **10,000 searches** | **4** | **100,000** | **67/194/139** | **0.4100 [0.3763, 0.4438]** | **3,037 [3,012, 3,061]** |
| **80,000 searches** | **8** | **100,000** | **143/190/67** | **0.5950 [0.5612, 0.6313]** | **3,167 [3,143, 3,193]** |
| 80,000 searches, earlier match | 8 | 50,000 | 252/115/33 | 0.7738 [0.7425, 0.8050] | 3,174 [3,144, 3,206] |

Policy-only Elo uses the 1,000-node anchor of 1,700 and the 64-search row the 5,000-node anchor of 2,220. The two
headline rows use 100,000 nodes = 3,100. Ratings are calculated as

`anchor + 400 * log10(score / (1 - score))`.

The first 80,000-search match used the probe-selected 50,000-node opponent and scored 77.4%, far from the most
informative 50% region. The subsequent 100,000-node confirmation scored 59.5%, tightened the conditional interval,
and moved the central estimate only seven Elo. It is therefore the headline result.

## Opponent-selection probes

Ten games per rung were used only to locate a plausible final opponent. They are not strength measurements.

| Candidate budget | Stockfish nodes | W/D/L | Score |
| ---: | ---: | ---: | ---: |
| 10,000 | 50,000 | 6/3/1 | 0.750 |
| 10,000 | 100,000 | 1/3/6 | 0.250 |
| 80,000 | 50,000 | 3/5/2 | 0.550 |
| 80,000 | 100,000 | 3/6/1 | 0.600 |
| 80,000 | 200,000 | 1/6/3 | 0.400 |

The 10,000-search rung was an exact distance tie and the stronger opponent was selected. For 80,000 searches, the
closest observed score selected 50,000 nodes.

## What the Elo scale means

The node anchors are readings from
[Marco Meloni’s Stockfish 13 curve](https://www.melonimarco.it/en/2021/03/08/stockfish-and-lc0-test-at-different-number-of-nodes/).
That curve fixed Fruit 2.2.1 near 2,830 from the historical Swedish Chess Computer Association list, then connected
Stockfish node levels through engine matches. The reproducible claim is v34’s score against the named Stockfish
binary at the named node limit. The absolute offset inherits assumptions from Meloni’s openings and protocol,
Fruit’s SSDF rating, and an old, approximate bridge between the SSDF engine pool and humans.

This supports “superhuman strength under the benchmark’s calibration” and “strong-engine territory.” It does not
support “3,167 FIDE Elo,” a predicted score against a specific grandmaster, or proximity to current full-strength
Stockfish. See [the rating-scale audit](../../analysis/chess-elo-scale-and-reporting-20260911.md) for sources and
public-reporting rules.

## Search time

The exact 80,000-search evaluation configuration was also measured without Stockfish, training, or self-play
processes present. Eight independent RTX 4070 SUPER workers each searched 50 opening positions after a 1,024-search
warm-up. Each worker used `parallel_searches: 8`, one inference worker, batch cap 64, and one outstanding batch.

Across the 400 searched positions, the per-GPU amortized time averaged **5.31 seconds per position**, with a
5.19-second median and a 5.06--5.87-second range across workers. This is saturated batched service throughput with
50 concurrent positions on each GPU. It is not the response latency of one isolated game.

## Search configuration and comparability

The 10,000-search result used `parallel_searches: 4`; the 80,000-search result used 8. Parallel search improves GPU
utilisation by allowing several in-flight leaves per tree, but its playing-strength cost is not precisely resolved.
The values should therefore retain their parallelism whenever quoted and should not be silently compared with
single-leaf evaluations.

All searched matches used one inference worker, batch cap 64, one outstanding batch, and exploration constant 1.0.
The full result files record every opening, colour assignment, move, outcome, termination, shard, device, model
hash, and configuration field.

## Evidence

- [`artifacts/final/`](artifacts/final/) contains the four complete 400-game result JSON files.
- [`artifacts/confirmation/search-80000-vs-sf100000.json`](artifacts/confirmation/search-80000-vs-sf100000.json)
  contains the closer-opponent 80,000-search confirmation match.
- [`artifacts/timing/`](artifacts/timing/) contains the isolated timing summary, eight worker records, and exact
  measurement scripts.
- [`artifacts/ladders/`](artifacts/ladders/) contains both ladder summaries.
- [`artifacts/ladder-selection.json`](artifacts/ladder-selection.json) records the final opponent decisions.
- [`artifacts/run-ladders.sh`](artifacts/run-ladders.sh) and
  [`artifacts/run-final-evaluations.sh`](artifacts/run-final-evaluations.sh) are the exact launch commands.
- [`SHA256SUMS`](SHA256SUMS) covers every committed artifact.

The complete fetched archive remains under
`.codex-diagnostics/v34-terminal-g1465-evaluations-20260911T143000Z`. All 79 entries in its original SHA-256
manifest passed verification before this compact benchmark was prepared.

The confirmation match and isolated timing evidence were fetched under
`.codex-diagnostics/v34-g1465-80k-vs-sf100k-20260912T032123Z`; all entries in its SHA-256 manifest passed
verification.
