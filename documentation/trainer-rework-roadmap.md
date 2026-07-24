# Trainer rework implementation roadmap

## Status and scope

This document is the executable plan for replacing iteration-bound self-play training
with a sample-credit-driven replay trainer. It divides the work into independently
assignable stages with explicit interfaces, tests, deployment boundaries, and rollback
points.

The current `complete-training-run-v5` remains useful for resignation and diagnostic
experiments. The replay, value-target, and scheduling changes require a new run because
they alter the data contract and the meaning of training progress.

This roadmap deliberately does not include:

- a network-topology change;
- WDL-valued MCTS backup;
- a caching inference client;
- a low-material termination heuristic;
- a change to full or fast self-play search counts;
- a change to playout-cap randomization;
- a network-learning-rate retuning experiment.

## Decisions

The initial clean-run design is:

| Control | Initial value |
| --- | ---: |
| Global batch size | 1,024 |
| Replay ratio | 4.0 |
| Optimizer steps per training quantum | 50 |
| Position presentations per quantum | 51,200 |
| Unique new samples required per quantum | 12,800 |
| Model publication cadence | Every completed 50-step quantum |
| Retained checkpoint cadence | Every 1,000 optimizer steps |
| Evaluation cadence | Every 1,000 optimizer steps |
| Maximum optimizer steps | 500,000 |
| Replay capacity | 2,500,000 unique positions, disk-backed |
| DDP ranks | 4 |
| Self-play pause during training | None |
| Initial learning rate | 0.002 |
| Self-play maximum game length | 250 plies |
| Self-play processes | 4 per GPU |
| Search concurrency per process | 2 search threads and 2 inference queues |
| Stored augmentation copies | None |
| Training-time symmetry | Random valid symmetry per sampled position |
| Post-augmentation discard | Removed |
| Outcome discount by distance to termination | Removed |
| MCTS value auxiliary weight | 0.15 |
| Resignation audit fraction | 0.10 |
| Production resignation during initial warm-up | Disabled |

Each unique retained full-search position earns four position-presentation credits.
A global batch consumes 1,024 credits. The trainer does not wake for a partial
quantum:

```text
presentation_credits_per_new_sample = replay_ratio
presentation_credits_per_quantum = global_batch_size * optimizer_steps_per_quantum
new_samples_per_quantum = presentation_credits_per_quantum / replay_ratio

presentation_credits_per_quantum = 1,024 * 50 = 51,200
new_samples_per_quantum = 51,200 / 4 = 12,800
```

Unused credits carry across process restarts. If multiple complete quanta accumulate,
the trainer executes them in order and publishes after each quantum so self-play does
not remain unnecessarily stale.

Every quantum atomically replaces a crash-recovery model and optimizer state. Every
twentieth quantum, at 1,000-step boundaries, additionally creates a retained immutable
historical checkpoint and schedules evaluation. Thus checkpoint retention remains
coarse without risking replay-credit duplication after a crash.

## Architectural invariants

The following invariants apply across all stages:

1. A replay credit is issued once for a unique, persisted, full-search position.
2. Symmetry augmentation never creates replay credit.
3. Credits are committed only after the corresponding replay shard is atomically
   visible and validated.
4. A training quantum either commits its model, optimizer, counters, and manifest
   together or remains retryable.
5. DDP ranks consume disjoint slices of each global batch.
6. Model versions are immutable. A self-play worker acknowledges a version only after
   the new parameters and buffers are active.
7. A model refresh does not flush replay data, destroy the MCTS object, or discard
   retained trees.
8. Weight replacement occurs only while the affected inference pipeline is quiescent;
   no request may observe a partly copied model.
9. Existing tree statistics may straddle adjacent model versions. This is an explicit
   asynchronous-training approximation and is logged, not accidental.
10. Final outcome and MCTS root value remain separate fields with separate losses.
11. Natural, capped, resigned, and diagnostic-audit terminations remain distinguishable.
12. Optimizer steps, trained position presentations, generated unique samples, and
    published model versions are monotonically increasing, restart-safe counters.

## Stage 0: resignation research

### Finding

Resignation is a real AlphaZero-family technique, but hard resignation is not
universally retained by later systems.

#### AlphaGo Zero

AlphaGo Zero:

- automatically selected a resignation threshold;
- targeted fewer than 5% false-positive resignations;
- disabled resignation in 10% of games and played those games to termination;
- resigned only when both the root value and best-child value were below the
  threshold.

The paper defines a false positive as a game that could have been won if resignation
had been disabled. Its binary Go value did not need to distinguish a recovered draw.

Source: [Silver et al., *Mastering the game of Go without human
knowledge*](https://ai6034.mit.edu/wiki/images/Nature24270_AlphaGoZero.pdf).

#### AlphaZero

The AlphaZero chess, shogi, and Go paper states that its training and search parameters
were identical to AlphaGo Zero unless otherwise specified. It does not restate the
complete training-time resignation calibration procedure for chess. It explicitly:

- terminated chess and shogi training games after 512 steps as draws;
- used a fixed AlphaZero value threshold of `-0.9` in reported evaluation matches.

The best-supported interpretation is that training inherited the AlphaGo Zero
procedure, but the paper does not provide a chess-specific false-positive definition,
warm-up schedule, or calibration cadence. Those details must not be attributed to
AlphaZero as published facts.

Source: [Silver et al., *A general reinforcement learning algorithm that masters
chess, shogi, and Go through
self-play*](https://www.davidsilver.uk/wp-content/uploads/2020/03/alphazero-science_compressed.pdf).

#### ELF OpenGo

ELF OpenGo used a sliding-window quantile tracker to target a 5% false-positive rate.
It also documented a serious failure mode: premature resignation caused a pessimistic
feedback loop, shortened games, skewed color results, produced overconfident values,
and destabilized training.

Source: [Tian et al., *ELF OpenGo: An Analysis and Open Reimplementation of
AlphaZero*](https://arxiv.org/abs/1902.04522).

#### Minigo

Minigo found resignation calibration unexpectedly difficult. It reduced the target
false-positive rate from 5% to 3%, automated a previously manual calibration path, and
monitored game length, color win rate, value magnitude, and held-out value error for
the pessimistic-loop signature.

Source: [Lee et al., *Minigo: A Case Study in Reproducing Reinforcement Learning
Research*](https://openreview.net/pdf?id=H1eerhIpLV).

#### KataGo

Modern KataGo describes a soft-resignation scheme: instead of immediately terminating
training games, it finishes positions beyond the would-resign point with cheaper,
lower-weighted searches. The stated motivation is avoiding pathological bias in the
training distribution.

Source: [KataGo methods documentation, “Policy Surprise
Weighting”](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#policy-surprise-weighting).

### Decision

Resignation is viable, but it starts as an audit-only feature on the current trained
model. Hard resignation is admitted to production only after the audit data
demonstrates a safe threshold. Soft resignation remains the fallback if hard
resignation produces target or game-distribution bias.

The initial safety target is stricter than AlphaGo Zero:

- a recovered win or draw is a false resignation in chess;
- the upper 95% confidence bound on the false-non-loss rate must be at most 3%;
- actual resignation remains disabled until at least 100 completed audit triggers are
  available;
- threshold updates use only completed audit games, never real resignations whose
  counterfactual result is unknown.

### Stage 0 deliverable

This research section is the completed deliverable. Stage 1 implements and tests the
audit path.

## Stage 1: resignation audit and canary

This is the only behavioral stage that can first run against the current trained model.
It must be independently deployable and reversible.

The existing code is a partial prototype, not the desired audit:

- it tests only the returned root scalar;
- it chooses the 10% continuation at the moment the threshold fires rather than at
  game creation;
- it uses the threshold-crossing MCTS scalar as the result of a real resignation;
- it counts recovered wins but not recovered draws;
- it has no calibrated threshold, confidence bound, activation gate, or persisted
  audit trajectory.

### Task 1A: typed resignation configuration

Add explicit configuration for:

- `audit_game_probability = 0.10`;
- `minimum_eligible_ply`;
- `minimum_published_model_version`;
- `minimum_completed_audit_triggers = 100`;
- `false_non_loss_upper_bound = 0.03`;
- `confidence_level = 0.95`;
- calibration-window size;
- threshold candidate range and resolution;
- maximum threshold change per calibration;
- production-resignation enable fraction;
- production-resignation fade length.

For the current iteration-based canary, real resignation remains disabled. For the
clean run, publication version 30 is an absolute activation floor, after which the
enable fraction fades in. The statistical audit gate can delay activation further.

Iteration 30 is not treated as equivalent to 30 publications in historical runs. It is
only a conservative minimum for the new model-version sequence.

### Task 1B: correct trigger semantics

At game creation, permanently assign the game as either:

- an audit game, which can never actually resign; or
- a production game, which may resign after all gates are satisfied.

At every eligible full-search turn, record:

```text
game_id
model_version
ply
side_to_move
root_value
best_child_value_from_root_player_perspective
threshold
would_resign
```

The child value must be converted to the root player's perspective before comparison.
If the child node stores the opponent-to-move value, negate its scalar value. Require
both root and best-child values to cross the threshold.

Once an audit game first crosses the threshold:

- record the hypothetical resignation ply;
- retain the subsequent trajectory only for calibration;
- exclude post-trigger positions from ordinary policy and value training;
- play until a natural result or separately tagged length-cap adjudication.

For an actual resignation:

- assign a hard loss to the resigning player;
- backpropagate alternating hard outcomes to eligible pre-resignation samples;
- do not pass the threshold-crossing MCTS scalar as the game result.

### Task 1C: calibration

For every candidate threshold, replay completed audit trajectories and select the most
aggressive candidate satisfying the configured upper confidence bound.

Recalculate after each 100 new completed audit triggers. Use a bounded rolling window
large enough to adapt to current model strength while retaining statistical power.
Persist the calibration input range and selected threshold in a manifest.

Report recovered outcomes separately:

- win;
- draw;
- loss;
- non-loss;
- naturally terminal versus capped/adjudicated.

A capped material adjudication is not valid evidence that a resignation was safe. It
is reported separately and excluded from the primary threshold fit.

### Task 1D: monitoring and automatic disable

Log:

- audit games started and completed;
- hypothetical and actual resignations;
- threshold and threshold-candidate curve;
- point and upper-confidence false-non-loss rates;
- recovered wins and draws;
- game-length distribution;
- color-conditioned results;
- mean absolute root value;
- held-out value loss;
- estimated searches and wall time saved.

Automatically disable real resignation if:

- the upper confidence bound exceeds the limit;
- there are too few recent audit triggers;
- color imbalance or game length crosses a configured alarm;
- value magnitude rises while held-out value quality worsens.

### Stage 1 tests

- Audit selection occurs once at game creation and is stable after copies/reroots.
- Audit games never resign.
- Root and best-child perspective conversion is correct for both colors.
- A real resignation produces a hard loss.
- Post-trigger audit samples never enter ordinary replay.
- Recovered draws count as false resignations.
- Calibration selects the expected threshold on deterministic fixtures.
- Confidence-bound behavior is correct for zero and nonzero failures.
- Threshold movement is rate-limited and restart-safe.
- Automatic disable preserves game generation.

### Stage 1 production validation

1. Deploy audit-only code to the current strong checkpoint.
2. Run until at least 100 hypothetical triggers complete naturally.
3. Compare fixed candidates such as `-0.99`, `-0.97`, `-0.95`, `-0.90`.
4. Inspect every recovered win and a sample of recovered draws.
5. Enable a small production fraction only if the 3% upper-bound criterion passes.
6. Run one observation window with no replay, target, search, or learning-rate change.

Rollback is a configuration-only disable. The audit records remain useful.

### Stage 1 canary result

The compute-node canary at `3bcdeadb3cf8481dd1cba9206ad1adffac7b9596`
validated the audit persistence and exclusion semantics against a trained checkpoint.
Its deliberately aggressive test threshold produced one hypothetical trigger at ply 5,
but that game ended by the diagnostic ply cap and was correctly excluded from safety
evidence. The run did not collect the required 100 naturally completed triggers, so it
provides no statistical basis for enabling hard resignation. Production resignation
therefore remains disabled for the initial clean run while audit collection continues.
The artifact is recorded under
`documentation/benchmarks/resignation-audit-canary-20260723/`.

## Stage 2: value-target contract

This stage changes the replay schema and is used only for the clean run.

### Task 2A: separate target fields

Replace the single mixed scalar target with typed fields:

```text
final_outcome: loss | draw | win
mcts_root_value: float in [-1, 1]
termination_reason: natural | resignation | ply_cap | diagnostic
target_eligible: boolean
```

Store the root value from the sample player's perspective. Do not convert it into a
synthetic WDL distribution in the dataset.

Natural and real-resignation outcomes are ordinary hard WDL targets. A capped material
adjudication remains provenance-tagged and is excluded from the primary outcome loss
until an explicit adjudication experiment decides otherwise.

### Task 2B: loss definition

The initial value objective is:

```text
outcome_loss = cross_entropy(value_logits, final_outcome)
expected_score = softmax(value_logits)[win] - softmax(value_logits)[loss]
mcts_auxiliary_loss = huber(expected_score, mcts_root_value)
value_loss = 0.85 * outcome_loss + 0.15 * mcts_auxiliary_loss
```

Log unweighted components as well as the combined loss because their numeric scales
differ. The 15% coefficient is a loss weight, not a claim that the MCTS scalar defines
15% of a calibrated WDL probability distribution.

Remove:

- scalar interpolation before WDL conversion;
- per-move outcome discount;
- distance-dependent WDL blur;
- low-material heuristic outcomes.

Keep MCTS backup scalar. Carrying full WDL through search is a separate future
experiment.

### Task 2C: metrics

Add:

- WDL cross-entropy and Brier score;
- expected-score MSE and MAE;
- MCTS auxiliary Huber loss;
- WDL calibration by class;
- expected-score calibration;
- metrics split by termination reason;
- target eligibility and exclusion counts.

Ply- and material-sliced value metrics are deferred to Stage 4. The Stage 2 replay
contract deliberately does not add those two sample attributes immediately before
Stage 4 replaces replay serialization and indexing. Stage 4 must persist the required
typed metadata, and Stage 6 must expose the resulting slices in TensorBoard.

### Stage 2 tests

- Outcome perspective alternates correctly by ply.
- WDL one-hot order is stable across Python, Torch, JIT, and C++.
- MCTS scalar remains independent of final outcome.
- Synthetic cases distinguish equal-scalar but different WDL distributions.
- Discounting is absent.
- Capped targets cannot silently enter the ordinary outcome loss.
- Legacy replay is rejected with a clear schema-version error.

Rollback requires returning to a legacy checkpoint and replay set. New and old replay
formats are intentionally not mixed.

## Stage 3: cheap model refresh

This stage makes 50-step publication practical and can be developed independently of
the replay scheduler.

### Current behavior

The native queued inference clients already call `updateInferenceModel`, which loads a
new TorchScript module and copies named parameters and buffers into the existing
module. However, the Python update path still:

- saves or flushes the current replay shard;
- replaces the dataset object;
- drops every retained root;
- runs Python garbage collection;
- updates curriculum settings together with weights.

The direct self-play inference path reconstructs `DirectSelfPlaySearch`, including its
worker pipelines, rather than updating those workers in place.

### Task 3A: separate commands

Split the existing iteration update into:

- `refresh_model(model_version, model_path)`;
- `update_search_schedule(schedule_state)`;
- `flush_replay_shard()`;
- `snapshot_statistics()`.

A model refresh must not imply any of the other three actions.

### Task 3B: in-place direct-inference update

Add a quiescent model-update operation to every direct inference worker:

1. stop accepting new batches;
2. drain and resolve all accepted batches;
3. load and validate the new module;
4. copy parameters and buffers into each live module;
5. publish the new version atomically;
6. resume batch acceptance.

Do not reconstruct `DirectSelfPlaySearch`, its queues, or its worker threads.

If loading fails, every worker continues using the previous complete version. A
partially updated worker set must never resume.

### Task 3C: retain games and trees

Keep:

- unfinished game state;
- the per-process in-memory replay shard;
- the MCTS object;
- retained search trees;
- accumulated generation statistics.

Log the oldest and newest model versions contributing to each completed game. Existing
tree statistics spanning adjacent versions are accepted. Add an optional diagnostic
mode that discards roots at refresh so the strength and throughput effect can be
measured without changing production semantics.

### Stage 3 tests and benchmark

- Model output changes after refresh while object and worker identities remain stable.
- All parameters and buffers match the published checkpoint.
- No request sees mixed weights.
- Failed refresh leaves the previous model usable.
- Dataset length, unfinished games, roots, and statistics survive refresh.
- Search-schedule values do not change during a pure refresh.
- Repeated refresh does not grow CPU or GPU memory.
- Queued and direct non-caching paths have equivalent refresh semantics.

Benchmark on the compute node:

- export and file-publication time;
- per-process refresh latency;
- total acknowledgement latency;
- refresh acknowledgement pause time;
- transient and steady GPU memory;
- games/hour with refresh every 25, 50, 100, and 250 steps;
- tree-retained versus tree-reset strength and throughput.

Acceptance for the initial design is refresh every 50 steps with no statistically
meaningful throughput regression and no memory growth.

## Stage 4: replay shards, FIFO index, and augmentation

This stage preserves coarse file-based writes while making individual samples
randomly addressable.

### Task 4A: atomic replay shards

Continue grouping at least tens of completed games per file. A producer:

1. writes a temporary HDF5 shard;
2. flushes and closes it;
3. computes its sample count and hash;
4. atomically renames it into the replay inbox;
5. atomically publishes a typed shard manifest.

The manifest contains:

- schema version;
- shard ID;
- game count;
- unique sample count;
- producing worker;
- minimum and maximum model versions;
- termination counts;
- content hash;
- creation timestamp.

Each stored sample also carries the typed ply and material metadata required for the
value-quality slices deferred from Stage 2.

The trainer never reads a temporary or unmanifested shard.

### Task 4B: one stored representation

Store each full-search position once in canonical orientation. Remove
`portion_of_samples_to_keep` from self-play replay writing.

At sample decode, independently choose a valid chess symmetry. For the current chess
encoding this is identity or left-right mirror. Transform both board planes and policy
indices together. The final outcome and MCTS scalar are unchanged.

Augmentation randomness derives from the persisted sampler seed, rank, global step,
and position within the global batch so restarts reproduce the same batch.

### Task 4C: bounded FIFO replay index

Maintain a persistent index of live shards and sample ranges. Capacity is measured in
unique stored positions, not iterations or augmented rows.

The initial capacity is 2,500,000 unique positions. Position tensors and targets remain
in HDF5 shards on disk. Memory may contain compact shard manifests, prefix ranges,
leases, sampler state, and aggregate statistics, but never the full replay payload.
Metadata memory must scale with shard count rather than stored position count.

Eviction is FIFO by committed shard. A shard may be deleted only after:

- it is outside the configured position capacity;
- no active batch references it;
- its eviction is recorded atomically.

The buffer-capacity value remains an experiment parameter and is not derived from the
50-step publication cadence.

### Task 4D: idle rolling-buffer compaction

The rolling replay buffer owns ingestion, FIFO indexing, compaction, leases, sampling,
and decode as one consistency boundary. Self-play workers continue publishing small,
atomic producer shards so new positions become visible immediately.

While the trainer is waiting for enough credits to start another quantum, compact
chronologically adjacent producer shards into immutable HDF5 containers. The sole
container target is 100,000 unique positions. Preserve whole producer shards and close
a container when their cumulative sample count first reaches or exceeds the target;
there is no game-count or byte-size limit.

The compactor:

1. streams source payloads into a temporary container without retaining them all in
   memory;
2. flushes, closes, hashes, and atomically renames the container;
3. atomically publishes a typed manifest mapping each logical source shard to its
   physical container range;
4. atomically remaps the rolling index;
5. waits for existing source leases to drain before deleting source files.

Original producer-shard ingestion issues replay credits. Compaction never creates
credits. FIFO eviction retires logical source ranges independently; a physical
container is deleted only after all of its ranges are evicted and unleased. Because
containers preserve chronological adjacency, at most the oldest live container should
be partially retired.

Expose compaction as bounded idle work so the Stage 5 scheduler can stop starting new
compaction as soon as a training quantum is credit-eligible.

Rank 0 is the sole replay-index writer. Other DDP ranks open the replay read-only and
refresh their index snapshot between quanta after rank 0 finishes ingestion, eviction,
or compaction. Cross-process leases are deliberately not persisted, so the Stage 5
scheduler must barrier all rank decodes before entering a replay mutation phase.

### Task 4E: fast random access

The loader must:

- sample uniformly across live unique positions;
- map sampled global indices to logical segments and physical containers without
  scanning every file;
- group a rank quantum by physical container and read all requested container indices
  together;
- vectorize HDF5 decoding;
- decode one complete 12,800-position rank quantum and expose fifty zero-copy
  256-position optimizer-batch views;
- prefetch the next rank quantum rather than individual HDF5 batches;
- issue disjoint samples to DDP ranks;
- reshuffle reproducibly after restart.

Within a 50-step quantum, sample without replacement when the live buffer is large
enough. Across quanta, repeated sampling is expected and is controlled by replay
credit.

### Stage 4 tests and benchmark

- No partial shard becomes visible.
- Every committed unique position issues exactly four credits once.
- Symmetry does not change sample count or credits.
- Mirrored policy indices remain legal and invert correctly.
- Sampling is uniform over unequal shard sizes.
- DDP rank partitions are disjoint.
- FIFO eviction never deletes a referenced shard.
- Index recovery after interruption yields the same live sample set.
- Compaction preserves deterministic sampling and issues no replay credits.
- Interrupted compaction leaves either source shards or the complete container usable.
- Logical FIFO eviction inside a container does not delete a still-live range.
- Rank-quantum decoding opens and coalesces by physical container.
- Loader throughput exceeds trainer consumption with production HDF5 shards.

## Stage 5: persistent credit-driven DDP trainer

### Task 5A: credit ledger

Persist:

```text
credited_unique_samples
earned_position_credits
consumed_position_credits
available_position_credits
completed_optimizer_steps
completed_training_quanta
```

Ingesting a committed shard increments:

```text
earned_position_credits += unique_sample_count * 4
```

Credits never depend on augmented rows, replay capacity, or the number of times a file
is discovered.

### Task 5B: quantum scheduler

Derive the quantum size from configuration. With the initial values, start a 50-step
training quantum only when at least 51,200 presentation credits are available.
Run exactly 50 complete global batches. On success:

1. save model and optimizer;
2. save the credit and sampler state;
3. create the immutable model-version manifest;
4. atomically mark the quantum committed;
5. subtract the calculated 51,200 presentation credits;
6. publish the model version.

On failure, resume from the last committed quantum. Never repeat a committed quantum
or consume its credits twice.

The DDP processes remain alive while waiting for credits. Avoid repeated NCCL
initialization. Use four DDP ranks for the clean run. Stop automatically after 500,000
committed optimizer steps unless the run is cancelled earlier.

### Task 5C: contention control

Do not pause self-play while DDP executes a quantum. The initial production layout is
four self-play processes per GPU, each with two search threads and two inference queues,
co-resident with one DDP rank per GPU. Benchmark this exact layout for generation
throughput, DDP duration, GPU memory headroom, and GPU duty cycle before launch.

The layout remains configurable for diagnosis, but the launch configuration does not
include a two-rank DDP comparison.

### Stage 5 tests

- Exactly 12,800 unique new samples enable one quantum at batch 1,024 and replay
  ratio 4.
- One fewer sample does not.
- Fractional or surplus credit carries forward.
- Duplicate shard discovery creates no credit.
- Restart before and after commit yields the correct optimizer step and credit count.
- Four DDP ranks execute 50 equal, disjoint global batches.
- Self-play remains enabled before, during, and after a training quantum.
- A failed publication does not consume credits.

### Stage 5 implementation status

The persistent credit ledger and runtime are implemented on `master` through
`1c8a83b73da6e7a330f8c766de044fc87fe0cc27`. The runtime keeps four DDP ranks
alive, admits only complete 50-step quanta, reconstructs deterministic disjoint
rank partitions from the disk replay, leaves continuous self-play enabled, and
commits credits only after every self-play process acknowledges the newly saved
model version. A prepared but unacknowledged publication is resumed without
retraining after restart.

Local validation completed with 324 passing and 5 skipped tests. The compute node
completed the focused credit/replay/DDP suite with 99 passing tests. A separate
four-GPU NCCL backend smoke executed 50 optimizer steps at global/local batches
of 1,024/256 in 5.05 optimizer seconds and processed all 51,200 expected
presentations. That smoke validates the production model and collective backend;
the complete credit scheduler, self-play refresh, interruption, and evaluation
sequence remains the Stage 7 end-to-end canary rather than being inferred from
this component test.

## Stage 6: publication, evaluation, and observability

### Task 6A: immutable versions

Identify each published model by:

- monotonically increasing model version;
- completed optimizer step;
- trained position presentations;
- generated unique positions credited;
- model, optimizer, and JIT hashes;
- source revision and run configuration hash.

Self-play workers acknowledge the exact immutable version. Do not use a mutable
`latest` file as the synchronization primitive.

### Task 6B: evaluation cadence

Scheduling uses trained position presentations rather than legacy iterations:

- cheap value/holdout metrics may run every publication;
- the current complete evaluation suite starts every 1,000 optimizer steps;
- only one evaluation suite is active at once;
- when a cadence boundary arrives while evaluation is still active, coalesce pending
  work to the newest eligible immutable checkpoint instead of starting an overlap;
- evaluation runs out-of-band and never gates replay ingestion, self-play, model
  training, checkpointing, or later model publication;
- process crashes, timeouts, CUDA failures, and repeated evaluation failures are
  persisted and visible but never terminate or stall the training loop;
- failed evaluations use bounded retry with backoff, and a permanently failing task is
  marked failed so the next cadence boundary can still run.

The 1,000-step cadence is provisional. Lower it toward 500 only if production timing
shows the entire suite reliably completes before the next boundary. Evaluation always
records the immutable model version and checkpoint hash.

### Task 6C: TensorBoard axes

Use `trained_position_presentations` as the primary training x-axis. Also log:

- optimizer step;
- training quantum;
- model version;
- unique positions generated and credited;
- credit balance;
- instantaneous and cumulative replay ratio;
- replay buffer occupancy and age;
- model staleness of generated samples;
- value metrics sliced by sample ply and material;
- refresh pause and acknowledgement latency;
- loader wait time;
- self-play and DDP throughput;
- evaluation source version.

Legacy iteration is not synthesized.

### Stage 6 tests

- Metrics preserve monotonic axes across restart.
- Evaluation results attach to the correct immutable model.
- Failed or slow evaluation cannot deadlock the trainer.
- Repeatedly crashing evaluation workers do not reduce self-play or training progress.
- A cadence boundary during an active evaluation coalesces to the newest checkpoint.
- A worker cannot acknowledge a model hash it did not load.
- Consolidated TensorBoard logs retain model-version provenance.

### Stage 6 implementation status

Stage 6 is implemented on `master` as of `23b8561`:

- immutable publications bind optimizer progress, credit counters, source and
  configuration revisions, and model/optimizer/JIT hashes;
- self-play refreshes through one atomic publication pointer and acknowledges
  the exact JIT hash it loaded;
- evaluation is a persistent, timeout-bounded, retry-bounded, coalescing
  scheduler that never gates credit training;
- telemetry uses trained position presentations as its primary axis and records
  replay age/staleness, replay I/O amplification, DDP and credited-position
  throughput, publication/acknowledgement latency, and evaluation provenance;
- total and termination value metrics remain exact, while ply/material slices
  deterministically sample one batch in ten to bound diagnostic overhead.

Local validation passed 342 standard tests with 5 skipped. Compute-node focused
validation passed 61 tests. The final four-rank, 50-step diagnostic smoke
reached 9,651.90 optimizer samples/s and 10,414.3 MiB peak host RSS, versus
10,140.71 samples/s and 10,371.4 MiB before Stage 6. Exact outputs and the
short-run limitation are recorded under
`documentation/benchmarks/credit-runtime-stage6-20260724/`.

## Stage 7: end-to-end migration and clean-run canary

### Task 7A: compatibility boundary

Increment the replay schema and run-manifest version. The clean trainer refuses
legacy mixed-scalar replay. The legacy trainer refuses the new schema. Conversion is
not provided because reconstructing separate final and MCTS targets from the existing
mixed scalar is impossible.

### Task 7B: deterministic integration run

Run a CPU/small-model integration loop that covers:

- shard generation;
- credit issuance;
- a 50-step quantum;
- DDP sampling;
- model publication;
- in-place self-play refresh;
- replay eviction;
- checkpoint/restart;
- evaluation scheduling.

Verify every counter and artifact hash.

### Task 7C: compute-node production canary

Run a short stochastic production-path canary with the planned topology. Record:

- unique samples/hour;
- games/hour;
- trainer samples/second;
- DDP active duration;
- refresh latency;
- model age in samples and wall time;
- replay occupancy;
- CPU and GPU utilization;
- process and aggregate RSS;
- GPU memory and OOM margin;
- evaluation completion time.

Compare against the current iteration-bound path using equivalent search and network
settings.

### Task 7D: clean-run launch gate

Do not start the clean run until:

- Stage 1 audit semantics are validated;
- Stage 2 target metrics pass fixed fixtures;
- Stage 3 refresh is memory-stable;
- Stage 4 loader is faster than DDP consumption;
- Stage 5 restart tests pass;
- Stage 6 evaluation cannot block training;
- a complete canary survives forced process interruption;
- every run parameter and source hash is recorded.

## Assignment boundaries

The intended task split is:

| Task | Primary ownership | Depends on |
| --- | --- | --- |
| Resignation audit and calibration | Self-play and telemetry | Stage 0 |
| Value schema and loss | Dataset, model training, evaluation | None |
| Cheap native model refresh | C++ inference and MCTS binding | None |
| Atomic shards and augmentation | Self-play serialization | Value schema |
| FIFO index and sampler | Replay loader | Atomic shards |
| Credit ledger and scheduler | Trainer orchestration | FIFO sampler |
| Publication and evaluation | Commander and telemetry | Credit scheduler, refresh |
| Integration and canary | End-to-end | All stages |

Each task must land as a separately validated commit. Do not combine the resignation
experiment with replay or target changes. Do not combine the native refresh path with
the trainer scheduler until its memory and concurrency tests pass independently.

## Remaining experiment decisions

The following are intentionally deferred:

1. Whether hard resignation graduates after at least 100 valid audit triggers.
2. Whether soft resignation is preferable if hard resignation fails its safety gate.
3. Whether the 1,000-step evaluation cadence can safely be lowered toward 500.
4. Whether retained mixed-version MCTS trees measurably help or hurt.
5. Whether the four-process, two-thread, two-queue self-play layout is optimal after
   production-path utilization measurement.

These are experiment controls, not reasons to keep the architecture iteration-bound.
