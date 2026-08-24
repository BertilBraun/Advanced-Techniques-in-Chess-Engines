# Showcase plots

Generated figures used by the repository [README](../../README.md). Every file here is produced by
[`py/tools/render_showcase_plots.py`](../../py/tools/render_showcase_plots.py) from a fetched run archive —
nothing is drawn by hand, and the source archive names are printed in each figure's footer.

## Current contents

| File | Content | Source |
| --- | --- | --- |
| `chess-strength-vs-wall-clock.svg` | fitted Stockfish-ladder Elo and score vs Stockfish level 0, both against wall-clock hours | `vast-chess-comp2-adaptive-20260823T204328Z` |
| `chess-training-loss.svg` | training policy/WDL loss vs optimizer steps, and fixed-dataset policy cross-entropy vs wall-clock | `vast-chess-comp2-adaptive-20260823T204328Z` |
| `chess-experiment-ladder-comparison.svg` | fixed-dataset top-1 accuracy for the four 2026-08-23 ladder runs | `vast-chess-4day-cnn`, `-attention`, `-comp1`, `-comp2` archives |

The dashed grey reference curve in all three figures is the four-day r3/r4 run, read from the tracked
[`yardstick_wall_h.csv`](../evidence/chess-four-day-freeze-20260817/yardstick_wall_h.csv).

**These are interim figures.** They are rendered from the 2026-08-23 experiment-ladder archives, which are the
best complete evidence available while `vast-chess-4day-production-v2` is still running (ends ~2026-08-28).
Regenerate them from the production archive once it is fetched; the run labels and the footer will then name the
production archive instead.

## Regenerating

From `py/`, with the fetched archive under `.codex-diagnostics/`:

```bash
python -m tools.render_showcase_plots --primary 'production-v2=../.codex-diagnostics/<fetched-archive>' --reference ../documentation/evidence/chess-four-day-freeze-20260817/yardstick_wall_h.csv --output-directory ../documentation/showcase
```

Add one `--comparison label=path` per run to redraw the comparison figure. The renderer reads
`tensorboard/coordinator/` when the archive has it and falls back to the per-boundary JSON under
`run/evaluations/` when it does not. It needs `matplotlib`, which is **not** part of the locked training
environment — install it into a local environment (`pip install matplotlib`) rather than adding it to `uv.lock`,
which the node bootstrap installs verbatim.

## Reading notes

- The x-axis is wall-clock time since run start, so curves from different hardware are not comparable; every run
  plotted so far is 8× RTX 4070 SUPER.
- Ladder Elo is the bisection fit over the Stockfish fixed-node rungs (anchors in
  [`py/src/evaluation/ladder.py`](../../py/src/evaluation/ladder.py)) at 64 evaluation visits. It is not
  comparable with the ≈2,800 figure quoted for the four-day run, which was measured at 10,000 visits.
- Fixed-dataset cross-entropy sits *below* the four-day reference (better) at the same wall-clock hour while the
  level-0 score sits far behind it: fit on the reference positions is not the binding constraint, playing
  strength out of self-play is.
