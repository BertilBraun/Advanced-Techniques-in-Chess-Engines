# Clean self-play training plan

## Purpose

This document separates immediate diagnostics on the existing `complete-training-run-v5`
from the behavioral changes intended for a clean run. The current run remains useful
because it already contains a strong model, but it is no longer suitable for attributing
strength changes to individual training decisions.

The plan has four goals:

1. measure the value head in ways that distinguish calibration, expected-score error,
   and draw prediction;
2. replace epoch-style replay training with an explicit and observable sample-reuse
   budget;
3. implement resignation with a measured false-positive bound;
4. choose self-play worker concurrency from production throughput measurements.

No new clean run should start until the diagnostic metrics, replay-ratio accounting,
and resignation audit semantics are implemented and tested.

## Decisions recommended now

- Keep the three-class win/draw/loss (WDL) head for the diagnostic phase. Its output
  is already normalized correctly with a softmax. Compare it with a scalar projection
  before changing the architecture.
- Treat WDL cross-entropy as the primary proper scoring rule, but always report it
  alongside WDL Brier score, scalar expected-score MSE, calibration, and per-class
  results.
- Do not evaluate value quality on training replay. Build a frozen, independently
  generated outcome holdout.
- Replace full passes over an iteration-indexed replay window with a fixed number of
  randomly sampled optimizer steps. Express the budget as trained positions per newly
  generated retained position.
- Initially target a replay ratio of 4.0 for the clean run. Benchmark 2.0 and 4.0
  before committing; do not continue with the current implicit ratio of approximately
  15.
- Retain a replay window large enough for diversity, tentatively ten generation
  updates, but sample only a fraction of it per update.
- Keep global batch size 2,048 for the initial controlled comparison.
- Remove the one-shot low-material termination from the clean run. Retain the
  250-ply cap and apply the material adjudication only at that cap.
- Keep the MCTS contribution to the value target at 15% after its warm-up for the
  first controlled run. Test strict outcome-only targets separately rather than
  changing both target construction and replay ratio at once.
- Introduce resignation only after its audit path can enforce and report a false
  resignation rate. Use a loss as the target for a game that actually resigns.
- Benchmark two, three, and four self-play processes per GPU. Select by games/hour
  and retained samples/hour, not instantaneous GPU utilization.

## What the current implementation actually does

The production configuration currently uses:

- 5,000 completed games per generation update;
- a fixed 15-iteration replay window;
- one complete pass over that window per update;
- global batch size 2,048 across four DDP ranks;
- 600-search full turns and 150-search fast turns;
- full searches on 25% of turns;
- storage of full-search turns only;
- a further `portion_of_samples_to_keep = 0.75` subsample when a replay shard is
  written;
- eight self-play processes, two per GPU, reduced to one per GPU while DDP trains.

The 0.75 setting discards positions, not entire games. Combined with playout-cap
randomization, approximately 18.75% of raw turns are retained before symmetry and
other filters: 25% full-search turns multiplied by 75% retained samples.

At iteration 261, the replay snapshot contained approximately 2.50 million retained
positions and produced 1,220 global optimizer steps. A newly retained position stays
in the window for 15 updates and is included in every complete pass, so its expected
lifetime reuse is approximately 15 presentations. Calling this "one epoch" obscures
the relevant quantity.

The console history through iteration 275 records 281 training invocations, including
repeated invocations of iterations 76, 144, 191, 196, and 260 after restarts. Those
invocations presented 808,313,853 positions and executed approximately 394,546
complete global batches. Restart-safe accounting is therefore part of the plan.

### Replay accounting definitions

Use these definitions in code and TensorBoard:

```text
new_retained_positions
    Positions first written by self-play since the previous model update.

trained_position_presentations
    Global optimizer steps * global batch size.

instantaneous_replay_ratio
    trained_position_presentations / new_retained_positions.

lifetime_sample_reuse
    Expected number of times a retained position is drawn before eviction.
```

For uniform sampling from a stable window, if every update adds `G` positions, draws
`R * G` positions, and retains `W` updates, each stored position is sampled with
probability `R / W` per update for `W` updates. Its expected lifetime reuse is `R`.
The replay-window size and reuse budget are therefore independent controls.

With the current late-run value `G ~= 2,500,000 / 15 ~= 166,667`:

| Replay ratio | Position presentations per update | Global steps at batch 2,048 |
| ---: | ---: | ---: |
| 2 | about 333,334 | about 163 |
| 4 | about 666,668 | about 326 |
| 10 | about 1,666,670 | about 814 |
| current full pass | about 2,500,000 | about 1,220 |

Increasing games per update while still making a complete pass over the window does
not reduce sample reuse. It increases both new data and optimizer work proportionally.

## Comparison with published systems

Iteration numbers are not comparable across implementations. Optimizer steps,
position presentations, newly generated positions, replay ratio, self-play search
cost, and wall time are the comparable quantities.

### AlphaGo Zero

AlphaGo Zero sampled each mini-batch uniformly from all positions in its most recent
500,000 self-play games. It used a global batch of 2,048 and emitted a checkpoint
every 1,000 training steps. Self-play, optimization, and evaluation ran
asynchronously. A self-play generation iteration contained 25,000 games; it was not
a full pass through the replay buffer.

The large 40-block run generated 29 million games and performed 3.1 million
mini-batches of 2,048 positions. That is 6.35 billion position presentations, but the
paper does not provide enough retained-position counts to derive a precise per-sample
reuse ratio.

Source: [Silver et al., *Mastering the game of Go without human
knowledge*](https://ai6034.mit.edu/wiki/images/Nature24270_AlphaGoZero.pdf).

### AlphaZero chess

AlphaZero removed the gated "best network" iteration and updated a single network
continually. Its chess run trained for 700,000 mini-batches of 4,096 positions.
The published paper describes continual updates rather than epochs over a fixed
buffer. Independent analysis of its training artifacts describes self-play network
refreshes every 1,000 training steps.

At 300,000 optimizer steps, AlphaZero had already crossed its reported Stockfish
strength comparison. At batch 4,096, that corresponds to 1.23 billion position
presentations. The current run has already executed about 394,546 global batches at
batch 2,048, or about 808 million position presentations. It is therefore not an
early or lightly trained run merely because it is around generation 275.

Sources:

- [Silver et al., *Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm*](https://arxiv.org/abs/1712.01815).
- [McGrath et al., *Acquisition of chess knowledge in
  AlphaZero*](https://www.pnas.org/doi/10.1073/pnas.2206625119).

### KataGo

KataGo samples uniformly from a moving window rather than making complete passes.
Its window grew from 250,000 to about 22 million positions. The main run generated
241 million self-play samples and trained on 954 million sample presentations, for
an aggregate replay ratio of approximately 3.96. It used batch size 256 because that
was the maximum fitting on its training GPU.

KataGo also explicitly identifies the tension between policy and value data:
fast searches generate more complete games for value learning, while only
full-search turns become policy targets.

Source: [Wu, *Accelerating Self-Play Learning in
Go*](https://arxiv.org/abs/1902.10565).

### Leela Chess Zero

Lc0 uses random sampling from recent training chunks and treats sampling ratio as a
first-class hyperparameter. Published run summaries report global batch sizes from
256 to 4,096 and estimated sampling ratios near 0.47 to 0.95 for several successful
runs. One historical configuration reached an estimated ratio of 14.22 and is not a
good target for this project.

Sources:

- [Lc0 training-run sampling comparison](https://lczero.org/dev/wiki/training-runs/).
- [Lc0 training repository and example configuration](https://github.com/LeelaChessZero/lczero-training).

### Practical conclusion

There is no universal rule that a sample must never be seen more than ten times, but
the current implicit ratio of about 15 is high relative to AlphaZero-style systems
and KataGo's measured ratio of about 4. A step-budgeted replay ratio makes this
choice explicit and testable.

## Phase A: diagnostics on the current trained run

These changes are observational. They should be deployed to the current run before
any further behavioral experiment.

### A1. Value metrics

Add the following metrics to the dataset evaluator:

- WDL cross-entropy;
- WDL Brier score: mean squared error across all three probabilities;
- scalar MSE and MAE after projecting to `P(win) - P(loss)`;
- WDL argmax accuracy and a 3x3 confusion matrix;
- mean predicted win, draw, and loss probabilities;
- predicted entropy;
- expected calibration error and maximum calibration error;
- reliability bins for win, draw, loss, and non-loss probabilities;
- per-outcome cross-entropy and Brier score;
- metrics split by ply bucket and material bucket.

Report reference baselines beside every result:

- uniform WDL prediction;
- empirical WDL-frequency prediction;
- always-draw prediction where numerically meaningful;
- constant mean expected-score prediction for scalar MSE.

Rename `evaluation/value_mse_loss`; it currently contains WDL cross-entropy.
Preserve the old TensorBoard tag temporarily only as a documented alias if continuity
is required.

### A2. Independent frozen holdout

Do not use replay data as the primary validation set. Generate a frozen holdout from
models that are never trained on those games:

1. play fixed paired openings using selected frozen checkpoints and fixed search
   settings;
2. disable resignation and the low-material shortcut;
3. record every searched position and the final strict game outcome;
4. separate naturally terminal games from 250-ply adjudications;
5. exclude capped games from the primary strict-outcome metrics and report them in a
   separate adjudicated cohort;
6. store the generating checkpoint, search parameters, opening identifier, final
   result, termination reason, and source revision.

Evaluate at least checkpoints 180, 220, 240, 260, and the latest checkpoint on the
same immutable holdout.

### A3. Target-construction audit

The current target pipeline combines three transformations:

1. terminal or heuristic game result;
2. `game_outcome_discount_per_move = 0.005`;
3. a 15% interpolation toward the MCTS root scalar.

The final scalar is converted to WDL using:

```text
win  = max(value, 0)
draw = 1 - abs(value)
loss = max(-value, 0)
```

This means that discounting or interpolating a decisive result toward zero is
represented as increased draw probability. For example, a target of `+0.6` becomes
`(0.6 win, 0.4 draw, 0 loss)`. That is a modeling choice, not a mathematically unique
conversion from expected score to a WDL distribution.

Add separate histograms and scalar summaries for:

- raw terminal outcome;
- outcome after per-move discounting;
- MCTS root scalar;
- final mixed scalar;
- resulting WDL target;
- distance in plies and recorded full-search samples from termination;
- natural, capped, low-material, and resigned termination classes.

Run an offline ablation on the frozen holdout comparing:

- current discounted and mixed targets;
- no per-move discount, 15% MCTS mixture;
- strict final outcome only;
- a scalar tanh head trained on the same expected-score targets.

This comparison determines whether the WDL head is failing or whether its current
targets deliberately teach diffuse draw probabilities.

### A4. Random and model-zero non-win audit

TensorBoard examples are insufficient because they are sampled and do not provide a
denominator. For every draw or loss against random and model zero, persist:

- PGN and opening identifier;
- candidate color;
- final FEN;
- result and termination reason;
- number of plies;
- whether the 400-ply evaluation cap fired;
- repetition count, half-move clock, and insufficient-material status;
- root WDL prediction and scalar value at each candidate move;
- selected move, root visit distribution, entropy, and search depth.

Run at least 400 fixed paired games for each audited checkpoint. Report Wilson or
bootstrap confidence intervals for score and each outcome rate.

The current evaluation protocol has a 400-ply cap despite earlier intent to leave
evaluation uncapped. Capped games must be reported separately rather than silently
counted as ordinary draws.

## Phase B: replay training refactor

### B1. Step-budgeted sampling

Replace `num_epochs` as the production stopping condition with:

```text
training_position_budget =
    replay_ratio * new_retained_positions_since_last_successful_update

optimizer_steps =
    floor(training_position_budget / global_batch_size)
```

Draw every batch uniformly without replacement within a short sampling cycle, then
reshuffle as needed. Do not require a full pass over the buffer. Across DDP ranks,
each global batch must remain disjoint and deterministic from the recorded seed.

Initial experiment values:

- replay ratios: 2.0 and 4.0;
- replay window: ten generation updates;
- global batch: 2,048;
- minimum optimizer steps: enough to avoid an update with only a handful of batches;
- carry fractional unused position budget into the next update.

Choose between 2.0 and 4.0 using held-out value metrics, policy loss, paired strength,
and training stability. KataGo's aggregate ratio of approximately 4.0 is the stronger
starting hypothesis.

### B2. Restart-safe accounting

Persist a completed-training manifest atomically with:

- source iteration;
- replay snapshot identity;
- number of new positions credited;
- position budget consumed;
- optimizer steps completed;
- model and optimizer checkpoint hashes.

A restart must either resume an incomplete update from a known checkpoint or detect
that the update already committed. It must not silently train on the same replay
snapshot twice.

### B3. Model-update cadence

Keep the initial cadence at 5,000 completed games while measuring the new retained
position count. Do not increase games per update merely because self-play is faster:
larger updates make the self-play policy staler.

After worker-concurrency benchmarking, choose cadence by a target number of newly
retained positions and a maximum model age, not only by games:

```text
update when retained positions >= target
or model wall age >= maximum age
```

Track total generated games, retained positions, optimizer steps, position
presentations, and wall time. Do not use total iteration count as the primary run
budget.

## Phase C: resignation experiment

AlphaGo Zero resigned only when both the root value and best-child value were below
the resignation threshold. Resignation was disabled in 10% of self-play games.
Those games were played to termination and used to select a threshold keeping false
positives below 5%.

The existing implementation is incomplete for this experiment:

- it checks only the returned root result;
- it uses a fixed disabled threshold;
- an actual resignation currently passes the MCTS result as the game outcome rather
  than assigning a loss;
- an audited game continues recording post-trigger training samples;
- it records whether a resigned player later won, but does not tune a threshold.

### C1. Required semantics

- Mark 10% of games as no-resignation audit games at game creation.
- For audit games, record the root value and best-child value on every eligible turn.
- Continue audit games to strict termination or the 250-ply cap.
- Once the first hypothetical resignation fires, record a training cutoff. Do not
  store positions after that cutoff in the ordinary policy/value replay.
- For a real resignation, assign `-1` from the resigning player's perspective and
  backpropagate the corresponding alternating result to pre-resignation samples.
- Track recovered wins and recovered draws separately. For chess, the conservative
  false-positive definition should be any non-loss, because resigning a drawable
  position also changes the correct game result to a loss.
- Require both root and best-child values to be below the threshold.

### C2. Threshold calibration

Start with resignation disabled while collecting a minimum audit population. For
each completed audit game, retain the sequence:

```text
(ply, root value, best-child value, final strict outcome)
```

Select the highest threshold whose estimated non-loss rate among hypothetical
resignations is at most 5%. Report both the point estimate and a binomial confidence
interval. Use a rolling completed-game window so the threshold adapts as the network
changes, and rate-limit threshold movement to avoid abrupt distribution shifts.

Candidate safeguards:

- no resignation before a configurable minimum ply;
- no threshold activation before the audit sample minimum;
- threshold clamped to a conservative range;
- automatic disable if the confidence bound or recent false-positive rate exceeds
  the configured limit.

### C3. Resignation outputs

Log:

- current threshold;
- number of audit games and hypothetical triggers;
- real resignations;
- recovered wins, draws, and losses;
- point and upper-confidence false-positive rates;
- plies and inference work saved;
- termination distribution before and after activation;
- value calibration around the threshold.

Run the resignation canary on the current strong model only after Phase A metrics
are present. It should not share an observation window with another target or replay
change.

## Phase D: self-play worker concurrency

Current production uses two self-play processes per GPU and one per GPU during DDP.
The observed utilization sawtooth is consistent with independent processes entering
Python-side game finalization and replay serialization together, but utilization
alone does not prove that more processes improve completed-game throughput.

Benchmark the exact production path at:

- two processes per GPU;
- three processes per GPU;
- four processes per GPU.

For each topology, measure separately outside and during DDP:

- completed games/hour;
- retained samples/hour;
- full and fast searched plies/hour;
- model evaluations/second and model calls/second;
- effective inference batch-size distribution;
- GPU SM and memory-controller utilization;
- p50/p95/p99 inference latency;
- time with GPU utilization below 25%;
- Python finalization and replay-save time;
- process and aggregate peak RSS;
- per-GPU memory and OOM margin.

Use identical stochastic production settings and enough time to finish a meaningful
number of games. Do not use a fixed-opening duplicate benchmark.

If four processes per GPU improve throughput safely, use four outside training and
test two active processes per GPU during training. Also benchmark two-rank DDP
against four-rank DDP under this contention. Choose the topology that minimizes
wall time per complete generate-train cycle.

A shared inference service that multiplexes multiple game producers is a possible
later architecture. It should not be introduced in the first clean run because it
would combine a substantial infrastructure change with the training experiments.

## Phase E: clean run experiment sequence

Change one causal group at a time.

### E0. Frozen baseline

- Save the current strongest candidate and optimizer state.
- Produce the independent value holdout.
- Run the expanded value metrics and random/model-zero audit.
- Record all source, configuration, model, and dataset hashes.

### E1. Replay-ratio pilot

Train short branches from the same initialization and identical generated replay:

- ratio 2.0;
- ratio 4.0.

Keep batch size, learning rate, value targets, and model architecture identical.
Select by held-out value calibration and paired strength, not training loss alone.

### E2. Value-target pilot

Using the selected replay ratio, compare:

- strict outcome;
- 85% outcome plus 15% MCTS scalar;
- current target including per-move outcome discount.

Do not enable low-material termination or resignation in this pilot.

### E3. Resignation canary

Enable calibrated resignation with 10% audit games. Retain the 250-ply cap as a
separate adjudication. Accept resignation only if:

- the observed and confidence-bound false-positive rates remain within the chosen
  limit;
- retained samples/hour improves;
- held-out value calibration and paired strength do not regress.

### E4. Production clean run

Tentative starting configuration, subject to the pilots:

- topology: 10 blocks by 3 convolutions by 96 channels;
- non-caching inference client;
- 600 full searches and 150 fast searches;
- 25% full-search turns, training targets stored only for those turns;
- global training batch 2,048;
- one step-budgeted replay update per generation;
- replay ratio 4.0;
- ten-update replay window;
- 5,000 games or an equivalent retained-position threshold per model refresh;
- 15% MCTS value mixture after warm-up;
- no low-material one-shot termination;
- 250-ply cap with separately tagged material adjudication;
- calibrated resignation only after its audit population is ready;
- worker and DDP topology selected by the Phase D benchmark.

The learning-rate schedule remains undecided. Select it with a short replay-fixed
pilot after the replay ratio is fixed. Do not infer a new learning rate merely from
batch-size scaling because AdamW, target softness, and the large change in optimizer
steps all affect the appropriate value.

## Acceptance criteria

The implementation is ready for a clean run when:

- all value metrics have unit tests against known WDL distributions;
- the fixed holdout is immutable and excluded from training;
- current and projected scalar values are consistent between Python and JIT/C++;
- replay ratio and actual position presentations are logged and restart-safe;
- DDP ranks receive disjoint samples for the configured step budget;
- resignation assigns a loss, excludes post-trigger rollout samples, and reports its
  measured false-positive rate;
- random/model-zero non-wins have complete termination artifacts;
- worker concurrency has a production-path throughput and memory result;
- a run manifest records every chosen parameter and source hash.

## Questions the experiments must answer

1. Is the high fixed-set WDL cross-entropy caused by poor expected-score prediction,
   poor draw calibration, or a mismatch introduced by discounted soft targets?
2. Does a scalar value head materially outperform WDL after both are evaluated with
   proper and projected metrics?
3. Does reducing expected sample reuse from about 15 to 2 or 4 improve held-out
   calibration and playing strength per generated position?
4. Does the 15% MCTS mixture improve strength once replay ratio is controlled?
5. Can calibrated resignation save meaningful self-play compute while keeping the
   upper confidence bound on false non-loss resignations acceptable?
6. Do four self-play processes per GPU improve completed games/hour without causing
   memory instability or inference-latency collapse?
7. Are random/model-zero non-wins genuine conversion failures, legal draw mechanisms,
   or artifacts of the 400-ply evaluation cap?
