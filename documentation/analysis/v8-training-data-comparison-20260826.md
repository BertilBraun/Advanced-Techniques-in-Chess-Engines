# v8 vs converting runs: what the trainer actually consumed

Date: 2026-08-26. Scope: the materialized sample stream, not the configuration. Configurations were diffed
elsewhere; this note only reports what was measured in replay stores and per-generation telemetry.

## Verdict

**Nothing material differs in the sample stream that v8's trainer is consuming today.** On every measure that
was testable, v8's training data is equal to or better than the runs that convert:

- v8's samples are drawn *closer* to the end of the game than any reference run's (12.8 % of rows within
  10 plies of the terminal position, versus 11.0 % and 11.6 % in the two four-day-lineage stores that convert).
- v8's WDL targets are *sharper*, not more drawish (72.3 % decisive versus 24.1 % and 22.8 %).
- The resignation-continuation share is 9.7-9.9 % in every run and every generation band, exactly the
  configured 0.1.

The three hypotheses in the brief are therefore all negative. The differences that do exist are confined to
generations 0-100, which left the replay buffer around generation 116 (2.0 M live rows at 128 k samples per
generation is a 15.6-generation window; v8 is at generation 314). They can only persist as weight damage.

## What became a sample

`py/src/replay/materialization.py::materialize_completed_game`: an observation becomes a sample **only if
`observation.full_search`**. Every sample in a game carries the same `game.final_wdl` (sign-flipped for the
side to move), blurred by `value_discount_per_ply ** remaining_plies`. Two consequences drive everything below:

1. Plies played after `force_fast_search_after_ply` produce **zero** training samples. Whatever happens in
   that region of the game is invisible to the trainer.
2. The value a cut game is assigned propagates to **all** of that game's samples, not just the last one.

## Direct replay-store measurements

Read straight from the memory-mapped stores (schema 1 for the four-day lineage, schema 4 for v8; ring-buffer
order reconstructed from `head`/`size`).

| store | generations | rows | mean remaining plies | ≤5 | ≤10 | ≤20 | ≤40 | decisive \|w−l\|>0.9 | drawish <0.1 |
|---|---|---|---|---|---|---|---|---|---|
| v8 live (`replay.bin`, node) | 290-314 | 2 000 000 | 51.0 | 6.1 % | **12.8 %** | 25.8 % | 48.8 % | **72.3 %** | 26.7 % |
| four-day freeze run-state | 601-624 | 1 500 000 | 56.0 | 5.4 % | 11.0 % | 22.3 % | 42.8 % | 22.8 % | 38.6 % |
| three-day full state | 517-546 | 1 500 000 | 54.5 | 5.7 % | 11.6 % | 23.3 % | 44.1 % | 24.1 % | 37.2 % |

Decisiveness by distance from the terminal position (the joint the brief asked for):

| remaining plies | v8 decisive | four-day decisive | three-day decisive |
|---|---|---|---|
| 0-5 | 0.850 | 0.824 | 0.835 |
| 5-10 | 0.817 | 0.769 | 0.783 |
| 10-20 | 0.807 | 0.751 | 0.764 |
| 20-40 | 0.791 | 0.269 | 0.276 |
| 40-80 | 0.724 | 0.000 | 0.000 |
| 80-160 | 0.598 | 0.000 | 0.000 |

The four-day and three-day collapse to zero decisive targets beyond ~26 plies because those runs ran
`value_discount_per_ply` 0.9985 → 0.996 (0.996^26 = 0.90), which uniformly blurs the WDL target with distance.
v8, r3-replica and v4 all run 1.0, so their targets stay at the raw ±1. **Every run in the v-series is sharper
than the four-day reference, so target blurring cannot explain non-conversion** — and r3-replica converts with
1.0 as well, so the discount is not the discriminator either. It is worth recording only as a setting no
v-series run has ever reproduced.

`sample_weight` is uniformly 1.0 in all three stores. v8's `auxiliary_1_eligible` is 0.9785, i.e. 2.15 % of
current rows have a censored remaining-game-length target (`censor_remaining_game_length_target: true` acting on
cut games); the reference stores are 1.000 because the setting did not exist then.

## Matched-generation comparison, v8 vs r3-replica

r3-replica's archive holds no replay store, so this half rests on per-generation TensorBoard telemetry
(histograms are 256-sample subsamples per generation; ~7.4 k samples pooled over generations 100-129, so a
14 % share carries roughly ±0.8 pp).

| measure, generations 100-129 | r3-replica | v8 |
|---|---|---|
| remaining plies at sample, mean / p50 | 46.5 / 37.1 | 47.8 / 39.1 |
| samples within 10 plies of the end | 14.5 % | 14.0 % |
| samples within 20 plies of the end | 29.6 % | 28.1 % |
| decisive WDL target | 73.2 % | 79.3 % |
| resignation-continuation share of games | 9.66 % | 9.93 % |
| samples per completed game | 17.4 | 18.2 |
| termination: natural / resignation / cut | 27.9 / 66.5 / 4.0 % | 30.2 / 68.9 / 1.0 % |
| replay generation age of trained rows, mean | 8.8 | 8.9 |

Materially identical. v8 sits ~1.5 pp lower on the near-terminal share and 6 pp higher on decisiveness; neither
is a plausible cause of a 20 % abandonment rate against Stockfish level 0.

Continuation share across all bands (per-generation deltas of the cumulative counters, over summed completed
games):

| generations | r3-replica | v4 | v8 |
|---|---|---|---|
| 1-50 | 0.0992 | 0.0997 | 0.0993 |
| 51-100 | 0.0981 | 0.0967 | 0.0983 |
| 101-129 | 0.0966 | — | 0.0993 |
| 130-200 | — | — | 0.0999 |
| 201-313 | — | — | 0.0965 |

This is structural: `SelfPlayWorker._active_game` draws `is_resignation_continuation` at game start for
*every* game with probability `continuation_game_probability`, independently of whether the game would ever
resign. The share cannot drift. v8 also resigns *later* than r3-replica (average trigger ply 78.8 versus 71.3
at generations 121-250; resigned-game mean length 80.8 versus 73.7), so its continuation games are if anything
longer.

## What does differ — generations 0-100 only

v8's `force_fast_search_after_ply` schedule is 110 (gen 0), 120 (20), 140 (40), 160 (60), 200 (100+).
r3-replica and v4 are a flat 200 throughout. Combined with the ply-cap schedule (v8: 150/160/180/200/**250**
from gen 100; r3-replica: 150/160/180/200 from gen 110), and weighted by the measured game-length distribution:

| generations | threshold T | cap | share of plies played that could not become samples | | |
|---|---|---|---|---|---|
| | r3-replica / v8 | r3-replica / v8 | r3-replica | v4 | v8 |
| 1-19 | 200 / 110 | 150 / 150 | 0.00 % | 0.00 % | **14.45 %** |
| 20-39 | 200 / 120 | 160 / 160 | 0.00 % | 0.00 % | **14.25 %** |
| 40-59 | 200 / 140 | 180 / 180 | 0.00 % | 0.00 % | 8.35 % |
| 60-99 | 200 / 160 | 180 / 200 | 0.00 % | 0.00 % | 3.07 % |
| 100-129 | 200 / 200 | 200 / 250 | 0.01 % | — | 1.01 % |
| 130-313 | — / 200 | — / 250 | — | — | 1.25-1.34 % |

The lost plies are always the *last* plies of the longest games — the conversion phase. For roughly the first
40 generations, one ply in seven of everything v8 played was structurally excluded from the training set, and
it was always the endgame ply. r3-replica and v4 lost none.

Two knock-on effects in the same window:

**Cut-game rate doubled.** Games past the threshold were played with 81-136 fast simulations instead of
326-544 full ones, so they wandered and hit the cap. Game-weighted `MAXIMUM_PLIES` fraction:

| generations | r3-replica | v4 | v8 |
|---|---|---|---|
| 1-19 | 0.158 | 0.110 | **0.268** |
| 20-39 | 0.162 | 0.162 | **0.316** |
| 40-59 | 0.106 | 0.125 | 0.181 |
| 60-99 | 0.067 | 0.130 | 0.043 |
| 100-129 | 0.040 | — | 0.010 |
| 130-313 | — | — | 0.011-0.013 |

**Cut values came from a fast search and were stamped on the whole game.** v8 sets
`early_termination.value_target: search_root_value`, which resolves to `bootstrap_cut_game_value=True`
(`py/src/games/chess/training.py:76`), so a cut game's `final_wdl` is `−(last observation's root value)`.
r3-replica and v4 have `early_termination: null`, so `bootstrap_cut_game_value=False` and a cut game gets the
material-based `adjudicated_wdl`. Because v8's cap sat *above* its fast-search threshold in every band, that
last observation is always a **fast** search (81-136 simulations); r3-replica's threshold sat above its cap, so
its last observation would always have been a full search. Estimated share of training samples carrying a
cut-game value (`cut_fraction × min(cap,T) / E[min(L,T)]`):

| generations | r3-replica (material) | v8 (fast-search bootstrap) |
|---|---|---|
| 1-19 | 27.2 % | 36.1 % |
| 20-39 | 23.9 % | 38.4 % |
| 40-59 | 17.4 % | 23.7 % |
| 60-99 | 12.9 % | 7.5 % |
| 100-129 | 8.9 % | 2.2 % |
| 130-313 | — | 2.5 % |

So at generations 1-39, close to 40 % of v8's samples carried a value bootstrapped from an 81-136-simulation
search of a position the run had already stopped searching properly — and they carried it uniformly, including
the opening plies of those games.

## What still differs today, and it is small

v8's self-play ply cap is 250 while `force_fast_search_after_ply` stays at 200. Plies 200-250 of every long
game generate no samples (1.25-1.34 % of all plies played, versus 0.01 % for r3-replica, whose cap and
threshold were both 200). Games reaching the cap are 1.1-1.3 % and still get a fast-search-bootstrapped WDL
stamped on every one of their samples — about 2.5 % of rows, matching the 2.15 % censored
remaining-game-length targets measured in the live store. This contradicts the working assumption that v8's
self-play cap fraction is 0.00; it is small but not zero.

Note that the four-day converting reference also ran `force_fast_search_after_ply` 160 under a cap of 200, i.e.
a 40-ply dead zone of its own, and converts 99/100. A dead zone at the current magnitude is therefore not by
itself sufficient to prevent conversion.

## Conclusion

The sample stream v8 is training on now is not defective, and by the two measures the brief singled out — the
ply distribution of samples and the decisiveness of their WDL targets — it is better than the data the
converting reference runs used. The defect is historical: for the first ~100 generations v8 discarded the
endgame plies of its longest games and replaced a third of its value targets with fast-search bootstraps. That
data has long since been evicted. The evidence points at residual weight damage from generations 0-100 rather
than an ongoing data defect.

## What could not be measured

- **No replay store or completed-games in the r3-replica, v4, v2 or v7 archives.** They carry only configs,
  logs, checkpoints, evaluations and TensorBoard. The direct sample-level read was possible only for v8 (live,
  on the node) and for the four-day/three-day freeze archives, which are a different lineage and a much later
  generation (517-624). The v8-versus-r3-replica half of this note therefore rests on per-generation
  TensorBoard telemetry, not on rows.
- **The joint distribution of sample ply and target decisiveness is not logged**, only the two marginals. It
  was recovered for v8 and the two freeze archives by reading their stores; it cannot be recovered for
  r3-replica.
- **v8's remaining-game-length histogram is censored on cut games**, so in bands where the cut fraction is high
  (generations 1-59, 18-32 %) v8's remaining-ply distribution is biased low and is *not* comparable to
  r3-replica's. At generations 100+ the cut fraction is ~1 %, so the comparison there is sound.
- **No run-level data exists for the four-day r3 phase at 12 h itself** (the W99/D0/L1 reference). The freeze
  archive's `run-state` and `three-day-full-state` are generations 517-624 and 517-546 of the r3/r4
  continuation, not the 12-hour point.
- TensorBoard histograms are 256-sample-per-generation subsamples, so pooled band statistics carry roughly
  ±0.8 pp on a 14 % share.

## Sources

- v8 live: `/workspace/alphazero-engine/py/training_data/validation/vast-chess-4day-production-v8/replay.bin`
  and `/workspace/tensorboard/vast-chess-4day-production-v8/coordinator` on 38.49.42.120:53893, read-only.
- `.codex-diagnostics/vast-chess-4day-r3replica-20260824T075654Z/{tensorboard,run}`
- `.codex-diagnostics/vast-chess-4day-production-v4-20260825T070451Z/{tensorboard,run}`
- `.codex-diagnostics/chess-baseline-four-day-freeze-20260817/vast-chess-8gpu-1d-r4-four-day-freeze-20260817/run-state/replay.bin`
  (source revision d39d5c85, schema 1, layout digest d6b4fed4ac251ab2…)
- `.codex-diagnostics/chess-baseline-four-day-freeze-20260817/vast-chess-8gpu-1d-r4-three-day-full-state/replay.bin`
