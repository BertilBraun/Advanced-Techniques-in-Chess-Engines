# Late-training evidence supporting Appendix C.5

These checks do not change the reported training totals or the plotted training window.

## Checkpoint matches

Local archive: `C:\Projects\AZ\.codex-diagnostics\final-2026-09-23\evidence-plateau-probe.tgz`.
The six records are `plateau-probe/g{900,960,1020}-s64-vs10k/result.json` and
`plateau-probe/g{900,960,1020}-s400-vs20k/result.json`.

Each record identifies the evaluated checkpoint explicitly. The diagnostic configuration filename contains
`v99-grown-promotion`, but the evaluated weights are the earlier checkpoints 900, 960, and 1020, not the
failed v99 continuation. No v99 training decline is used as plateau evidence.

The counts below were recomputed from individual `games[].outcome` entries. Ratings use
`anchor + 400 * log10(score / (1 - score))`, with score `(wins + draws / 2) / 100` and anchors
2,470 at 10,000 Stockfish nodes and 2,700 at 20,000 nodes.

| Checkpoint | Searches | W/D/L | Derived benchmark Elo |
| ---: | ---: | --- | ---: |
| 900 | 64 | 18/26/56 | 2331.0 |
| 960 | 64 | 16/31/53 | 2335.0 |
| 1020 | 64 | 16/34/50 | 2347.0 |
| 900 | 400 | 31/41/28 | 2710.4 |
| 960 | 400 | 33/38/29 | 2713.9 |
| 1020 | 400 | 26/36/38 | 2658.1 |

## Late continuation

Local archive: `C:\Projects\AZ\.codex-diagnostics\final-2026-09-23\evidence-tensorboard.tgz`.
Read `evaluation/ladder_elo_64` from all event files under
`tensorboard/vast-chess-8gpu-v97-revert-promotion/coordinator/`, ordered by event step.

There are 28 observations, spanning event steps 214800–247200 and wall times
1790079269.6784463–1790112508.43303 (9.232987 hours). The arithmetic means of the first and last
14 observations are 2373.6946847098216 and 2376.742257254464, respectively.
Rounded at the displayed precision, these are 2373.7 and 2376.7, a difference of 3.0 Elo.
This is a descriptive comparison of successive ladder observations, not independent training seeds or a
confidence interval. It uses the ongoing three-rung estimator, separately from the single-rung checkpoint
matches above. It excludes v99 and subsequent changed-recipe experiments.
