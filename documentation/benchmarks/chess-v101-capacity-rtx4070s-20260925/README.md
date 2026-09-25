# V101 — does capacity keep paying after catch-up?

The last experiment of the chess programme. It asks one question: a 19x176 trained from scratch
reaches a given strength later than a 12x128, but does it keep climbing where the smaller model
flattens? Every cheaper candidate for the plateau had already been eliminated, so this was the
remaining one worth a full run.

**It does not.** V101 closed to within 13 Elo of V89 by hour 17 and then went flat while V89 kept
climbing past it.

## Provenance

| | |
|---|---|
| run name | `vast-chess-8gpu-v101-fp16-big-wide-value` |
| source revision | `75f0c00c` |
| configuration | `py/configs/production/vast-chess-8gpu-v101-fp16-big-wide-value.yaml` |
| `experiment_configuration_sha256` | `e8ee62766177771fab6a723fbb676051fc6a5a258c0fbfb8d20deebda40626c5` |
| node | Vast.ai offer 48571853, 8x RTX 4070 SUPER, driver 595.71.05 |
| runtime | PyTorch 2.12.1+cu126, CUDA 12.6, cuDNN 9.10.2 |
| elapsed | 86,727 s (24.1 h), generation 240, 120,000 optimizer steps |
| cost | $17.35 |
| baseline | `vast-chess-8gpu-v89-v35-progressive-int8-sgd-reuse4-plateau`, same node and hardware |

Evidence archive `.codex-diagnostics/final-2026-09-23/evidence-v101.tgz`, sha256
`e6faa44f56d8a462be6fd6b7ba5dac9632ec526e3c88b73b1ee4f1a01ff55e41` — TensorBoard, the preserve
archive, evaluation results, run state and run-control logs. Weights and the replay store are not
retained.

## What differed from the baseline

V101 changed five things at once; it was a go/no-go run, not an ablation, so nothing here attributes
an effect to any single change.

| | V89 | V101 |
|---|---|---|
| network | 12x128 -> 14x160 progressive | **19x176 fixed, from scratch** |
| value head | 2 channels / 48 fc | **32 / 128** |
| quantization | INT8 QAT | **disabled (FP16 TensorRT)** |
| exploration constant | fixed 1.5 | **`auto`** (AlphaZero schedule) |
| auxiliary targets | next_policy, remaining_game_length | **+ 3x future_search_value, irreversible_progress, value_residual** |
| learning rate | 0.1 -> 0.01 over 1000 generations | **0.2 -> 0.01** |

Unchanged: replay ratio 4, 20M buffer, batch 2048, 500 steps per quantum, self-play visits
300 -> 800, the opening suite, and the evaluation definitions.

## Result

Both runs measured on `evaluation/ladder_elo_64`, the three-rung bracketed fit at 64 searches a
move, against the same Stockfish 13 anchors. Raw series in `ladder-elo.csv`. Six-point means:

| hours | V101 | V89 | gap |
|---|---|---|---|
| 5 | 1705 | 1972 | -267 |
| 9 | 1983 | 2078 | -95 |
| 13 | 2058 | 2101 | -43 |
| 17 | 2130 | 2143 | **-13** |
| 19 | 2114 | 2169 | -55 |
| 21 | 2114 | 2172 | -58 |
| 23.33 (last shared) | 2146 | 2187 | -41 |

Per-hour slopes from a least-squares fit over each window:

| window | V101 | V89 |
|---|---|---|
| 8-14 h | **+18.8 Elo/h** | +5.3 |
| 14-21 h | **+2.2 Elo/h** | +9.4 |
| 16-24 h | **+1.0 Elo/h** | +5.5 |
| 21-24 h | **-2.5 Elo/h** | +1.5 |

Over the final six evaluations V101 averaged 2132 against 2129 for the six before it: **+3 Elo across
the last two hours.** The run ended flat, not climbing.

The closing between hours 5 and 17 is not evidence for the hypothesis: V89 was saturating over that
window at +5.3 Elo/h while V101 was still in the steep part of its own curve. When V101 reached the
same region it saturated too, about four hours later and roughly 40 Elo lower, and its slope fell to
zero while V89's stayed positive. V89 goes on from 2187 to a stitched plateau of 2358 over the
following days.

The single fit that most cleanly separates the two readings is the recent slope rather than the gap.
A bigger model that had further to go would show a slope above the baseline's once it arrived; V101
shows one below it at every window past hour 14.

Policy-only (`evaluation/ladder_elo_1`, 1 search a move, the raw network without a tree) tells the
same story one step earlier: the gap tightened to -31 by hour 19 and then held at -33, flat from
hour 15.

Time to 2000 Elo: **V89 4.0 h, V101 8.67 h.**

## Reading

Capacity was not the binding constraint, and the bigger network reached its ceiling sooner rather
than later. That completes the elimination started in the post-mortem: the learning-rate floor,
self-play label quality, replay diversity and now capacity have each been measured and found not to
be what holds this system at ~2360 at 64 searches.

The honest limit of this result is that V101 bundled five changes. A flat outcome does not show the
capacity hypothesis is wrong in general — only that this recipe, on this budget, on this hardware,
did not beat a 12x128 that had a four-hour head start and kept it.
