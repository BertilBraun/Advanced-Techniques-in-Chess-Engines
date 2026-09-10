# v34 final evaluation and small-model distillation plan

Date: 2026-09-10. This plan does not authorize stopping v34 or spending GPU time. It fixes the
measurement protocol so the terminal checkpoint can be evaluated immediately after the user stops
the run.

## Terminal checkpoint and evidence gate

Use one checkpoint for every result: the last fully published generation before the clean stop.
Stop and preserve through `deployment/run_control.sh`. Start the first ladder from the preserved run
state as soon as all eight GPUs are idle, then fetch the archive concurrently with that GPU work. The
transfer may contend lightly for disk but should not leave rented GPUs idle. Verify the fetched archive
manifest, checkpoint hash, resolved configuration hash, source revision, and ZIP integrity before
treating the run as complete.

The four reported model budgets are:

| Label | `--model-searches` |
| --- | ---: |
| policy only | 1 |
| shallow search | 64 |
| deep search | 10,000 |
| very deep search | 80,000 |

The policy-only match uses one search and `parallel_searches=1`; the searched final matches use
`parallel_searches=4`. All use one inference worker, batch size 64, one outstanding batch, exploration
constant 1.0, Stockfish 13 with one thread and 1,024 MiB hash, and the committed
`py/reference/chess-elite-2025-11-balanced-4moves-200-v1-openings.json`. The opening manifest has 200
pairs and file SHA-256 `40582c4f753e90d8ebd170498c37e3f4812bff38287e6bfa6db376e9eec70d39`.

## Opponent selection

Run 10-game probes at each rung. Ten games cannot establish strength; they only expose the likely
neighbourhood for the user's opponent selection.

| Model budget | Initial Stockfish node probes | Reason |
| ---: | --- | --- |
| 10,000 | 50k, 100k | v29 scored 0.34 at 50k; v34's observed improvement is measured in tens rather than hundreds of Elo. |
| 80,000 | 50k, 100k, 200k | Repeats the useful 10k anchors and adds one stronger bound without assuming a large search-depth gain. |

The calibrated node anchors are 20k = 2700 Elo, 50k = 2960, 100k = 3100, 200k = 3230, 500k = 3350,
and 1M = 3400. These are approximate readings from the Melonimarco Stockfish 13 curve, not precise
engine ratings. The final ladders stop at 100k and 200k respectively; do not add higher rungs without
a new user decision.

Present every probe W/D/L and score to the user, along with the closest-to-0.50 rung and any score
bracket. The user selects the final opponent. A final score expected between 0.35 and 0.65 gives useful
precision, but ten-game noise means the selection must consider all rungs rather than mechanically
accepting the closest point.

For policy-only and 64-search matches, use the final in-run evaluation metadata to avoid another
ladder. Pool the last four boundaries only when all four used the same Stockfish node rung; use that
pooled score solely to choose the opponent. If the expected score is outside 0.35--0.65, run a
10-game adjacent-rung probe. The TensorBoard games remain trend evidence and are not substituted for
the terminal-checkpoint match because they evaluate four different checkpoints on repeated openings.

## Eight-GPU execution

`tools.run_stockfish_ladder` and `tools.run_stockfish_gauntlet` already shard opening pairs across the
selected devices, start one process per used device, merge the results, and require the merged game
indices to cover the requested games exactly once.

Each 400-game gauntlet has 200 opening pairs. An eight-GPU assignment gives each GPU 25 pairs/50 games,
of which roughly 25 positions are candidate turns at any instant. With `parallel_searches=4`, each GPU
can expose about 100 in-flight leaves to the inference path. The configured inference batch itself is
64, so four-way search parallelism is sufficient to keep it fed without creating a 100-position model
batch.

Run all four final matches sequentially on all eight GPUs. This avoids the long idle tail created by a
four/four split: 10k finishes much earlier than 80k, whereas sequential execution gives every phase the
whole machine. Use `parallel_searches=4` for 64, 10k, and 80k. Policy-only uses one search and therefore
`parallel_searches=1`. The measured playing cost of parallel search is unresolved (roughly 6--45 Elo in
existing evidence), so do not describe the searched results as directly interchangeable with v29's
`parallel_searches=1` number.

The order is:

1. 10,000-search ladder.
2. 80,000-search ladder.
3. Run the four 400-game matches sequentially on all eight GPUs: policy-only, 64, 10,000, then 80,000.
4. Validate each `result.json`, its eight shard files, W/D/L total, game-index coverage, hashes, and
   paired-bootstrap interval before accepting the phase.

Command form for a ladder, run from `py` in the preserved revision's prepared worktree:

```bash
python -m tools.run_stockfish_ladder \
  --experiment "$EXPERIMENT_CONFIG" \
  --run-directory "$RUN_STATE" --checkpoint-generation "$GENERATION" \
  --opening-manifest "$OPENINGS" --stockfish-executable "$STOCKFISH" \
  --stockfish-node-ladder 50000 100000 \
  --probe-games 10 --opening-selection-seed 20260815 --match-random-seed 20260816 \
  --devices 0 1 2 3 4 5 6 7 --model-searches 10000 --parallel-searches 4 \
  --inference-workers 1 --inference-batch-size 64 --outstanding-batches 1 \
  --exploration-constant 1.0 --output-directory "$OUTPUT/ladder-10k"
```

The 80,000-search ladder uses `50000 100000 200000`, changes the model budget to 80,000, and writes to
`ladder-80k`.

Command form for each final match:

```bash
python -m tools.run_stockfish_gauntlet \
  --experiment "$EXPERIMENT_CONFIG" \
  --run-directory "$RUN_STATE" --checkpoint-generation "$GENERATION" \
  --opening-manifest "$OPENINGS" --stockfish-executable "$STOCKFISH" \
  --stockfish-nodes "$SELECTED_NODES" --all-opening-pairs --opening-selection prefix \
  --match-random-seed 20260816 --devices $DEVICES \
  --model-searches "$MODEL_SEARCHES" --parallel-searches "$PARALLEL_SEARCHES" \
  --inference-workers 1 --inference-batch-size 64 --outstanding-batches 1 \
  --exploration-constant 1.0 --output-directory "$MATCH_OUTPUT"
```

Set `DEVICES` to `0 1 2 3 4 5 6 7` for every match. The policy-only match sets
`PARALLEL_SEARCHES=1`; every searched match sets it to 4. The prior v29 10,000-search match took 2.06
hours for 200 games on one RTX 4070 SUPER; ideal eight-card scaling puts 400 games near 31 minutes
before the parallel-search throughput gain. Treat roughly 4--5 hours as the planning allowance for
80k on eight cards and measure rather than claim linear scaling.

With 200 paired opening clusters, the expected sampling interval is roughly +/-25--30 Elo near the
chosen rung, based on the v29 paired variance. Report the actual paired-bootstrap interval. This does
not include error in the Melonimarco Stockfish anchor scale, so the result must retain the opponent
node count and anchor source beside the Elo estimate.

## Approximately 0.5M-parameter distillation

Use the final 10M-position replay buffer as the first student dataset. It already contains searched
policy targets, outcome/value targets, legal actions, sample weights, surprise scores, source
generations, and the current encoded representation. That is more valuable for the first experiment
than generating raw final-teacher head labels: the student learns from 600/800-visit improved policy
targets. Scientifically this is fixed-replay compression rather than pure teacher-logit distillation,
and it should be labelled that way.

Compare two target sources over the same frozen replay positions:

1. **Search-target compression:** train from the replay's stored visit policy and value/outcome targets.
2. **Teacher-head distillation:** run the final v34 network over those same encoded positions and train
   from its raw policy and WDL distributions, without search.

Holding positions, split, architecture, optimiser, steps, and seeds fixed makes the comparison answer
whether searched labels transfer capability better than direct function imitation, and how quickly
each target source trains. Independently generated teacher positions would confound the target source
with a different state distribution and are unnecessary for this first comparison.

The existing distillation trainer does not directly read a production `ReplayStore`, so add a typed,
read-only replay input path plus a batched final-teacher relabelling path before launch. Freeze the
replay at the clean stop and record its layout, logical row count, physical ordering/state, and file
hash. Do not fetch or copy the 10M replay before starting evaluation; retain it on the node and train
from the read-only store after final evaluation.

Under the current chess representation, useful from-to-head candidates around the target size are:

| Student | Parameters |
| --- | ---: |
| 4x80 | 496,063 |
| 5x72 | 502,243 |
| 6x64 | 470,295 |
| 8x56 | 475,023 |

These counts include the policy and value heads and exclude temporary auxiliary training heads. Do not
pick one from parameter count alone.

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
