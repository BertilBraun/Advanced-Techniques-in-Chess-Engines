# Why v7 and v8 could not convert won games

2026-08-26. Runs v6 through v9 on the 8×RTX 4070 SUPER node.

## Symptom

v8 reached overwhelmingly winning positions against Stockfish level 0 and could not finish them. At
11.3 h it scored 74/25/1 — one loss in a hundred — while **19 of 100 evaluation games were abandoned
at the 300-ply cap**, which the scorer counts as draws. Replaying those 21 abandoned games from
their recorded `played_action_ids` and computing material at the final position:

    candidate material at cap: mean 19.67 pawns, median 20.00
    winning >= +3: 21 | balanced: 0 | losing <= -3: 0

Every abandoned game was overwhelmingly won. The deficit was entirely unconverted wins, not weakness:
v8 lost 1 game per 100 where the four-day reference lost 3.

The tax ran 12–20 games per 100 across every Stockfish rung and fell to 3–7 against v8's own past
checkpoints — the signature of failing only when far stronger than the opponent.

## Reference behaviour

| run | wall-clock | level-0 W/D/L | abandoned |
| --- | --- | --- | --- |
| four-day r3 phase | 12 h | 99 / 0 / 1 | **0** |
| r3-replica | 8 h | 86 / 9 / 5 | 1–5 |
| v2 | 2 h | 2 / 17 / 81 | 0–3 |
| v4 | 6 h | 45 / 21 / 34 | 0–2 |
| v7 | 4.3 h | — | **21** |
| v8 | 11.3 h | 74 / 25 / 1 | **19** |

v2 and v4 lose heavily (81 and 34 per 100) and still finish their games. Weakness and
non-conversion are independent failures. Abandonment appears exactly at v7.

## Hypotheses eliminated, with the evidence

Five were proposed and rejected in order. Recording them because each looked plausible.

1. **`remaining_game_length` censoring removed the urgency signal.** The four-day reference has no
   `early_termination` block at all, so it censored nothing and converted 99/100.
2. **Search backup discount.** The four-day r3 phase ran `value_discount_per_ply: 1.0` — no discount
   anywhere — and converted. v7 ran 1.0 and abandoned 21; v8 ran 0.98, a stronger discount than any
   reference, and abandoned 19. Mechanism: with no terminal inside the tree, every line evaluates
   ≈1.0 and a backup discount scales them all alike, so it provides no gradient.
3. **Syzygy removal.** r3-replica has `maximum_ply_syzygy_paths: None` too.
4. **Resignation rate.** v8 fires at 0.68–0.72, r3-replica at 0.66–0.70; the resignation config
   blocks are character-identical. `continuation_game_probability` is drawn per game at start,
   independent of whether the game resigns, so the continuation share cannot drift — measured at
   9.65–9.99 % in every run.
5. **`force_fast_search_after_ply` starving endgame training.** Weakened twice: the four-day
   reference itself ran threshold 160 under a cap of 200 — a 40-ply dead zone — and converted
   99/100; and by v8's generation 100 the threshold sat at 200 against a p90 game length of 165, so
   it touched only the top few percent of games while abandonment stayed at 19–21.

A sample-stream comparison against the archived replay stores also ruled out any *current* data
defect. v8's training samples sit **closer** to terminal positions than the converting references
(12.8 % within ten plies of the end against 11.0 %) and carry **sharper** targets (72.3 % decisive
against 22.8 %).

## What actually differed

Two things, neither of which any earlier analysis had isolated.

**Generations 0–39 were poisoned.** `force_fast_search_after_ply` ran 110–160 while the ply cap was
150–200, and only `full_search` observations become training samples, so **14.4 % of all plies
played were structurally excluded from the training set** (r3-replica: 0.00 %) — always the last
plies of the longest games. Worse, because the cap sat *above* the fast-search threshold, every cut
game's value was bootstrapped from an 81–136 simulation fast search and then stamped on **every
sample in that game**, roughly 36–38 % of samples in that window. That data left the replay buffer
around generation 116, so by generation 314 v8 carried residual weight damage rather than an ongoing
defect.

**No v-series run ever had a training-side ply discount.** The four-day and three-day references ran
`value_discount_per_ply` 0.9985 → 0.996, visible in their stored targets as decisiveness collapsing
beyond ~26 plies remaining. Every v-series run used 1.0. This was wrongly dismissed early because
r3-replica converts at 1.0 — but r3-replica was 86/9/5, weak enough that it rarely reached
overwhelming positions at all.

## The v9 change set

Five changes, applied together:

| change | from (v8) | to (v9) |
| --- | --- | --- |
| `value_discount_per_ply` (training) | 1.0 | **0.998** |
| `force_fast_search_after_ply` | staged 110→200 | flat 200 |
| `maximum_game_plies` | 160@20, 180@40, 200@60 | r3 schedule (160@50, 180@80, 200@110) |
| `continuation_game_probability` | 0.1 | 0.2 |
| `false_nonloss_rate_ceiling` | 0.03 | 0.025 |

Plus a code change: a game reaching the ply cap now defers one batch round and runs a **forced full
search at the cut position**, whose root value becomes the final WDL. This removes the
cap-above-threshold hazard entirely and contributes one full-search policy target at a deep position
per cut game.

## Result

Comparing at matched **strength** rather than matched wall-clock — the control that matters, because
abandonment only appears once a model is strong enough to reach won positions:

| | level-0 score | draws | abandoned |
| --- | --- | --- | --- |
| **v9 @7200s** | 0.700 | 8 | **0** |
| v8 @3.3 h | 0.690 | 18 | 13 |
| v8 @4.0 h | 0.725 | 23 | 15 |
| v8 @5.3 h | 0.745 | 39 | 22 |

v9 records **zero abandoned games in every evaluation from 6000 s onward** (one exception at
10800 s). At 15600 s it scored 97/1/2 — essentially the four-day reference's twelve-hour result,
reached in 4.3 hours. Ladder Elo 1502 at 4.7 h against v8's 1236; fixed-nodes-1000 0.340 against
0.150.

## Limitations

- **Five changes were applied at once.** Which one fixed conversion is not isolated. The ply discount
  is the best-supported candidate on mechanism and on being the one thing the converting references
  had, but the cut-position search and the doubled continuation games are untested alternatives.
- Matching on level-0 score is mildly circular: v8's score was itself depressed by its abandonment,
  so v8's true strength at 0.690 was somewhat higher than the number implies.
- One run each side. The four-day reference did not show its own late-run behaviour until well past
  this point.

## Method notes worth keeping

- **Compare at matched strength, not matched wall-clock.** Low abandonment early means the model is
  too weak to reach won positions, not that it converts. This confound invalidated two intermediate
  conclusions during the investigation.
- **Evaluation records support forensics.** `evaluations/*.json` stores `played_action_ids` per game,
  so any position can be replayed and measured — that is how the +20-pawn figure was obtained.
- **Check the time base before comparing archives.** The four-day archive's `run-state/evaluations`
  timestamps are elapsed seconds of the r4 *phase*, while `tensorboard/composed-r3-r4` is the
  from-scratch timeline. Mixing them compares a mature model against a young one.
- **The skill ladder and the fixed-nodes ladder are not equally comparable across runs.** Both use
  `engines/stockfish` (Stockfish 18) for skill levels, but the `stockfish_fixed_nodes` rungs are
  pinned to `stockfish-13` in the v-series and were not in the four-day run.
- **`test_self_play_worker.py` skips without the native extension.** Two validators
  (`CompletedSelfPlayGame` and the replay shard manifest) rejected the trailing unplayed observation
  the cut-position search produces; both were caught only by running the suite on a node, and either
  would have failed every cut game within minutes of launch. Local green is not green for the
  self-play worker.
