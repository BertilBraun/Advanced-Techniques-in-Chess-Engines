# v34 final evaluation and small-model distillation plan

Date: 2026-09-10. This plan does not authorize stopping v34 or spending GPU time. It fixes the
measurement protocol so the terminal checkpoint can be evaluated immediately after the user stops
the run.

## Terminal checkpoint and evidence gate

Use one checkpoint for every result: the last fully published generation before the clean stop.
Stop, preserve, and fetch through `deployment/run_control.sh`; verify the fetched archive manifest,
checkpoint hash, resolved configuration hash, source revision, and ZIP integrity before treating the
run as complete. Run the evaluations from the preserved run state only after all eight GPUs are idle.

The four reported model budgets are:

| Label | `--model-searches` |
| --- | ---: |
| policy only | 1 |
| shallow search | 64 |
| deep search | 10,000 |
| very deep search | 80,000 |

All reported matches use `parallel_searches=1`, one inference worker, batch size 64, one outstanding
batch, exploration constant 1.0, Stockfish 13 with one thread and 1,024 MiB hash, and the committed
`py/reference/chess-elite-2025-11-balanced-4moves-200-v1-openings.json`. The opening manifest has 200
pairs and file SHA-256 `40582c4f753e90d8ebd170498c37e3f4812bff38287e6bfa6db376e9eec70d39`.

## Opponent selection

Run 20-game probes at each rung. Twenty games cannot establish strength; they only select a rung for
the 400-game match.

| Model budget | Stockfish node probes | Reason |
| ---: | --- | --- |
| 10,000 | 20k, 50k, 100k, 200k | Brackets v29's 2,800--2,845 result and reaches the 3,230 anchor. |
| 80,000 | 50k, 100k, 200k, 500k, 1M | Allows the deeper search to exceed 3,000 without extrapolating beyond calibrated anchors. |

Select the calibrated rung whose probe score is closest to 0.50; choose the higher rung on an exact
tie. A final score expected between 0.35 and 0.65 gives useful precision. If every probe lies outside
that band, extend the ladder by one calibrated adjacent rung and probe before selecting.

For policy-only and 64-search matches, use the final in-run evaluation metadata to avoid another
ladder. Pool the last four boundaries only when all four used the same Stockfish node rung; use that
pooled score solely to choose the opponent. If the expected score is outside 0.35--0.65, run a
20-game adjacent-rung probe. The TensorBoard games remain trend evidence and are not substituted for
the terminal-checkpoint match because they evaluate four different checkpoints on repeated openings.

## Eight-GPU execution

`tools.run_stockfish_ladder` and `tools.run_stockfish_gauntlet` already shard opening pairs across all
devices, start one process per device, merge the results, and require the merged game indices to cover
the requested games exactly once. With 200 pairs, each GPU receives 25 pairs and plays 50 games.

Use all eight GPUs for one phase at a time. This keeps every result on the same hardware topology and
gives the expensive 80,000-search match the full machine. The order is:

1. 10,000-search ladder.
2. 80,000-search ladder.
3. Four 400-game single-rung matches, cheapest first: 1, 64, 10,000, 80,000 searches.
4. Validate each `result.json`, its eight shard files, W/D/L total, game-index coverage, hashes, and
   paired-bootstrap interval before starting the next phase.

Command form for a ladder, run from `py` in the preserved revision's prepared worktree:

```bash
python -m tools.run_stockfish_ladder \
  --experiment "$EXPERIMENT_CONFIG" \
  --run-directory "$RUN_STATE" --checkpoint-generation "$GENERATION" \
  --opening-manifest "$OPENINGS" --stockfish-executable "$STOCKFISH" \
  --stockfish-node-ladder 20000 50000 100000 200000 \
  --probe-games 20 --opening-selection-seed 20260815 --match-random-seed 20260816 \
  --devices 0 1 2 3 4 5 6 7 --model-searches 10000 --parallel-searches 1 \
  --inference-workers 1 --inference-batch-size 64 --outstanding-batches 1 \
  --exploration-constant 1.0 --output-directory "$OUTPUT/ladder-10k"
```

The 80,000-search ladder changes the node list to `50000 100000 200000 500000 1000000`, the model
budget to 80,000, and the output directory to `ladder-80k`.

Command form for each final match:

```bash
python -m tools.run_stockfish_gauntlet \
  --experiment "$EXPERIMENT_CONFIG" \
  --run-directory "$RUN_STATE" --checkpoint-generation "$GENERATION" \
  --opening-manifest "$OPENINGS" --stockfish-executable "$STOCKFISH" \
  --stockfish-nodes "$SELECTED_NODES" --all-opening-pairs --opening-selection prefix \
  --match-random-seed 20260816 --devices 0 1 2 3 4 5 6 7 \
  --model-searches "$MODEL_SEARCHES" --parallel-searches 1 \
  --inference-workers 1 --inference-batch-size 64 --outstanding-batches 1 \
  --exploration-constant 1.0 --output-directory "$MATCH_OUTPUT"
```

The prior v29 10,000-search match took 2.06 hours for 200 games on one RTX 4070 SUPER. Ideal scaling
therefore puts 400 games on eight cards near 31 minutes. Treat roughly 4--5 hours as the planning
allowance for the 80,000-search match; measure rather than claim linear scaling. The cheap matches and
both probe ladders should fit around those two long phases.

With 200 paired opening clusters, the expected sampling interval is roughly +/-25--30 Elo near the
chosen rung, based on the v29 paired variance. Report the actual paired-bootstrap interval. This does
not include error in the Melonimarco Stockfish anchor scale, so the result must retain the opponent
node count and anchor source beside the Elo estimate.

## Approximately 0.5M-parameter distillation

The repository already supports teacher-head distillation and multi-GPU dataset generation. Under the
current chess representation, useful from-to-head candidates around the target size are:

| Student | Parameters |
| --- | ---: |
| 4x80 | 496,063 |
| 5x72 | 502,243 |
| 6x64 | 470,295 |
| 8x56 | 475,023 |

These counts include the policy and value heads and exclude temporary auxiliary training heads. Do not
pick one from parameter count alone. Generate 16M teacher-labelled positions as eight independent 2M
shards with distinct seeds, then merge them with `tools.distill_merge_datasets`. At 1,409 bytes per
record, budget about 22.5 GB for the merged binary plus temporary shards and checkpoints. Preserve at
least 50 GB of free disk before beginning.

The generator should use independent per-ply retention (`--sample-one-position-in 14`) and a 0.10
random-action perturbation. Before launch, make its opening randomisation match v34's uniform inclusive
0--8-ply rule; the current builder accepts one fixed ply count and therefore needs a small explicit
change and focused validation. It does not need regret restarts: the objective is to imitate the final
teacher over a broad playable distribution, and perturbations already move samples off its principal
trajectories.

Train the four architectures at two seeds each, one arm per GPU, for 100,000 steps on the complete
dataset. Use AdamW at 0.002, batch 1,024, a 1,000-step warmup, a high-rate plateau followed by cosine
annealing over the final 20%, and the from-to policy head. Select by held-out policy cross-entropy gap
above the target-entropy floor; do not rank gaps below 0.005 nats without the second seed.

Evaluate only the winning architecture. First measure teacher and student throughput on an idle GPU at
the match's actual root population and pin that ratio. Then play 400 games against the teacher at:

- equal searches at 64 and 10,000 searches per move;
- equal compute at the same teacher budgets, giving the student the pinned throughput multiple;
- the same Stockfish rung used for the teacher's 64-search final match, at both 64 searches and the
  equal-compute student budget.

Report parameter reduction, inference throughput ratio, held-out policy and WDL losses, student-minus-
teacher Elo at equal searches and equal compute, and calibrated Stockfish Elo. The earlier 0.48M probe
is the baseline: it was -246 Elo at equal shallow searches and -75 Elo at equal compute against a much
weaker teacher, with a 4x measured throughput advantage. A deeper-search gap is likely; retaining most
of v34's 10,000-search strength would be a surprising positive result rather than the default
expectation.

## Completion artifacts

Fetch the full evaluation directories, distillation dataset manifest, all student manifests and logs,
the selected student checkpoints, throughput measurement, and match results. Each benchmark README
must carry the exact source revision, resolved configuration SHA, teacher and student checkpoint hashes,
opening and engine hashes, GPU inventory, commands, wall times, W/D/L, intervals, and limitations.
