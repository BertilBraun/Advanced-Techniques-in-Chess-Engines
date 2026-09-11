# v34 final evaluation and small-model distillation plan

Date: 2026-09-10. This plan does not authorize stopping v34 or spending GPU time. It fixes the
measurement protocol so the terminal checkpoint can be evaluated immediately after the user stops
the run.

The terminal evaluation and replay-compression phases are complete. Final strength results and compact raw evidence
are recorded in
[`chess-terminal-v34-generation1465-rtx4070s-20260911`](../benchmarks/chess-terminal-v34-generation1465-rtx4070s-20260911/README.md).
Compression results and the published student model are recorded in
[`chess-replay-distillation-v34-rtx4070s-20260911`](../benchmarks/chess-replay-distillation-v34-rtx4070s-20260911/README.md).

## Terminal checkpoint and evidence gate

Use one checkpoint for every result: the last fully published generation before the clean stop.
Stop and preserve through `deployment/run_control.sh`. Start the first ladder from the preserved run
state as soon as all eight GPUs are idle, then fetch the archive concurrently with that GPU work. The
transfer may contend lightly for disk but should not leave rented GPUs idle. Verify the fetched archive
manifest, checkpoint hash, resolved configuration hash, source revision, and ZIP integrity before
treating the run as complete.

The four reported model budgets are:

| Label | Model action selection |
| --- | ---: |
| policy only | direct masked-policy argmax |
| shallow search | 64 searches |
| deep search | 10,000 searches |
| very deep search | 80,000 searches |

Policy-only uses direct inference. The searched final matches use `parallel_searches` of 1 at 64
searches, 4 at 10,000, and 8 at 80,000. All use one inference worker, batch size 64,
one outstanding batch, exploration constant 1.0, Stockfish 13 with one thread and 1,024 MiB hash, and the committed
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

Run the two ladders concurrently on disjoint devices: 10k on GPUs 0--2 and 80k on GPUs 3--7. Use batch
64, parallelism 4 for 10k, and parallelism 8 for 80k. Fetch the preserved archive concurrently with
both ladders.

Each final match has 200 opening pairs. Launch four ordinary gauntlet commands concurrently, each
sharded over GPUs 0--7. Policy-only uses direct inference; 64 uses parallelism 1; 10k uses 4; and 80k
uses 8. Monitor device memory and process health after launch. If four stacks do not fit, stop the
failed phase cleanly and revise the allocation from measured memory.

Keep the inference batch cap at 64. In the matched 800-search evaluation benchmark, parallelism 1 took
219.6 seconds with batch 64 and 305.7 seconds with batch 320; parallelism 4 took 120.2 versus 158.6
seconds. The self-play-sized batch cap was 32--39% slower because the evaluation population padded
underfilled batches. The four commands retain independent 64-slot inference queues.

Parallelism 8 is an accepted throughput/strength trade for the two deep budgets. Its playing cost is
unresolved (roughly 6--45 Elo in existing evidence), so retain it in every result and do not describe
the numbers as directly interchangeable with v29's `parallel_searches=1` result.

The order is:

1. Run the 10,000- and 80,000-search ladders concurrently on their pinned GPU sets.
2. Present both ladders for user selection of the two final opponents.
3. Run all four 400-game matches concurrently as separate commands, each over all eight GPUs.
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
  --devices 0 1 2 --model-searches 10000 --parallel-searches 4 \
  --inference-workers 1 --inference-batch-size 64 --outstanding-batches 1 \
  --exploration-constant 1.0 --output-directory "$OUTPUT/ladder-10k"
```

The concurrently launched 80,000-search ladder uses `50000 100000 200000`, devices `3 4 5 6 7`,
parallelism 8, and output directory `ladder-80k`.

Command form for each final searched match:

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

Use parallelism 1 at 64 searches, 4 at 10k, and 8 at 80k. Policy-only requires a direct policy
selector rather than describing a one-visit tree search as policy-only.
The prior v29 10,000-search match took 2.06 hours for 200 games on one RTX 4070 SUPER. Treat roughly
4--5 hours as the planning allowance for the shared phase's 80k tail and measure rather than claim
linear scaling.

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

The replay does not contain raw policy logits or the source network's raw WDL prediction. Those are
discarded before materialization. It stores sparse MCTS visit counts, the legal-action set, the
discounted terminal-outcome WDL target, and the recorded search root value. The second treatment must
therefore infer fresh policy and WDL distributions from the final checkpoint.

The first experiment is search-target compression only: train from the replay's stored visit policy
and value/outcome targets. Raw policy logits and raw network WDL predictions are not retained in the
replay. A later teacher-head experiment must rerun the final checkpoint over positions to recreate
those labels. When that follow-up is run, reuse the same replay positions and split so the comparison
changes the labels without changing the state distribution.

The existing distillation trainer does not directly read a production `ReplayStore`, so add a typed,
read-only replay input path before launch. Freeze the replay at the clean stop and record its layout,
logical row count, physical ordering/state, and file hash. Do not fetch or copy the 10M replay before
starting evaluation; retain it on the node and train from the read-only store after final evaluation.

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

Evaluate only the winning architecture. First measure teacher and student end-to-end search throughput
on an idle GPU at the match's actual root population and pin the student/teacher searches-per-second
ratio. Parameter count and raw network throughput do not determine equal playing time because tree
work and batching do not scale linearly with model size. Correct `tools.distill_match` to scale the
student budget from `searches_per_second`; it currently records that field but uses
`positions_per_second`.

Play 200 games (100 paired openings) against the teacher at equal searches and again with the student
budget multiplied by the pinned search-throughput ratio. The second condition is equal expected search
time per move; report actual move and match durations because it cannot guarantee exact wall-time
equality. Expand to 400 games only if the first interval leaves the conclusion unresolved.

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

## Prepared orchestration entry points

The executable local preparation is available as three modules, run from `py`:

- `tools.run_stockfish_terminal_ladders` launches both fixed ladders concurrently on disjoint configurable GPU sets. It
  does not fetch archives and rejects an existing output root.
- `tools.run_stockfish_terminal_evaluations` requires the four user-selected Stockfish node counts and launches policy-only,
  64-search, 10,000-search, and 80,000-search matches concurrently. Each child uses every configured GPU and an
  independent batch-64 inference path. Policy-only is direct masked-policy argmax and is recorded as such.
- `tools.run_replay_compression_experiment` hashes the frozen replay once and passes that recorded digest to each read-only
  trainer without eight repeated full-store hashes. It trains the four approximately 0.5M-parameter
  architectures at two seeds on all eight GPUs, selects by mean held-out policy gap above floor, measures the
  end-to-end search-throughput ratio at the match root population, and runs the 200-game equal-search and
  equal-expected-time matches. Valid completed training arms and evaluation results are reused only after the typed
  immutable request manifest exactly matches all resolved inputs and settings; incomplete evidence is never
  overwritten.

Exact commands and arguments are maintained in [`py/README.md`](../../py/README.md#terminal-chess-evaluation-and-replay-compression).
