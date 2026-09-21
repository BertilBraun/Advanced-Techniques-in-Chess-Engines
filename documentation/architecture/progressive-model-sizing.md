# Progressive model sizing

> Current implementation guide, updated for
> [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). The KataGo comparison and design
> rationale below are retained as historical context. Exact recipe values belong to the configuration, while the
> state transitions described here follow [`py/src/training/progressive.py`](../../py/src/training/progressive.py).

## Purpose and scope

Model sizing is a training policy above the shared chess/Go model, replay, objective, DDP, and checkpoint contracts.
It is independent of the concrete network architecture: fixed sizing supplies one complete model definition, while
progressive sizing supplies an ordered tuple of complete definitions. The runtime does not infer width, depth, or a
relationship between architectures. Progressive sizing also selects one explicit candidate-start policy. The final
chess configuration uses stage-specific plateaus in the primary searched evaluation ladder; historical research
configurations may retain the single-latch Elo policy or elapsed active-run starts.

This policy does not add a transformer path, model-shape adapter, weight transfer, match gate, or checkpoint
averaging. Every later model starts from its own random initialization. All models use the run-fixed input,
policy/WDL, and auxiliary-head layout, so their total losses have the same defined terms and weights.

## KataGo precedent and intentional differences

KataGo's paper says that it began with small residual networks, trained the next larger size concurrently on the
same data, and switched when the larger network's average loss caught up. Its main run moved from 6 blocks x 96
channels to 10x128, 15x192, and 20x256, with switches at roughly 0.75, 1.75, and 7.5 days. The paper also records a
separate training GPU for the concurrently trained next size. See [Accelerating Self-Play Learning in
Go](https://arxiv.org/abs/1902.10565), section 2 and appendix C. KataGo's current training guide describes its normal
official pipeline as asynchronous self-play, shuffling, training, exporting, and optional gating, and explicitly
supports extra or train-only models; see [SelfplayTraining.md](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md).

This platform intentionally differs:

- training is synchronous at the coordinator boundary, so every eligible model trains sequentially within one
  quantum on the same immutable replay snapshot and deterministic sample identity;
- the coordinator waits for all required models before publishing the checkpoint and transitioning self-play;
- the final staged Elo plateau policy persists one latch for the immediate successor and resets it after promotion;
- promotion compares paired exponential moving averages built only from quanta seen by both the active model and its
  immediate successor, rather than an unspecified lifetime average;
- no match gate decides promotion, and evaluation results never publish or promote a candidate;
- no parameters or optimizer moments transfer between sizes, and no checkpoints are averaged;
- only one active checkpoint is atomically published to self-play and evaluation.

These differences trade KataGo's asynchronous throughput for deterministic comparability and a small exact restart
boundary that fits this platform's blocking DDP quantum.

## Configuration and eligibility

`training.progressive_model_sizing` is a discriminated configuration union. The `fixed` variant owns one model and
has no candidate-start or promotion settings. The `progressive` variant owns at least two ordered
`ProgressiveModelDefinition` values, one candidate-start policy, and the promotion policy. Each definition owns a
stable model ID and a complete network definition. This is the only network-configuration owner: the fixed model or
first progressive definition is the day-zero published model, so there is no duplicate `training.network` field.

The legacy `elo_plateau` candidate-start configuration contains one value beyond its discriminator:

- `minimum_worthwhile_gain_per_hour`, a positive threshold for the primary searched ladder's EMA Elo gain rate.

The final chess recipe instead uses `staged_elo_plateau`. It contains one ordered entry for every successor model,
and each entry names that model and its positive gain-rate threshold. The configured stages are:

- `chess-cnn-scaled-post-14x160-fromto-int8`: 15 Elo/hour;
- `chess-cnn-scaled-post-19x176-fromto-int8`: 5 Elo/hour.

The staged model IDs must exactly match the configured successor order. A stage can therefore neither apply the
wrong threshold to a model nor silently omit a successor.

The Elo EMA decay is fixed in code at `0.90`. The runtime stores the bias-corrected EMA, its observation count, and
the latest applied boundary. Its initial baseline is Elo `0` at boundary `0`. Whenever a new primary searched
ladder Elo boundary becomes complete, boundaries are applied in chronological order. If `n` observations have
already been incorporated, the next corrected average is:

```text
previous_weight = 1 - 0.90^n
current_weight = 1 - 0.90^(n + 1)
ema_next = (0.90 * ema_previous * previous_weight + 0.10 * observed_ladder_elo) / current_weight
gain_per_hour = (ema_next - ema_previous) / hours_between_boundaries
```

The elapsed hours are the actual time between the two consecutive EMA boundaries. Failed boundaries add no Elo
observation; duplicate results and results received out of order do not apply an observation twice. A gain at or
above the current stage's threshold resets the consecutive-below-threshold count. Five consecutive gains strictly
below the threshold latch candidate catch-up for the next complete quantum. Later observations continue updating the
EMA and telemetry but cannot clear that stage's completed latch.

After promotion, the staged policy targets the new active model's immediate successor, clears the latch and
confirmation count, and starts a new stage. It seeds that stage at the latest observed raw Elo and its boundary, so
the next gain is measured from the transition point rather than from the preceding stage's complete history. If no
Elo observation exists yet, the new stage starts at the zero baseline. There is no earliest elapsed gate, lookback
window, configurable ladder search budget, or post-latch cancellation within one stage.

The promotion configuration explicitly owns:

- EMA decay `d`, constrained to `0 < d < 1` and normally configured as `0.8`;
- a positive number of paired warmup quanta;
- the maximum candidate-to-active relative loss;
- a positive catch-up learning rate for an eligible candidate that is not yet active.

The primary searched ladder is the highest configured project-model search budget, currently the
`evaluation/ladder_elo_64` series also published as `evaluation/ladder_elo`. Policy-only and lower-search ladder
series never affect candidate training, including when every primary-budget rung at a boundary fails.

The `elapsed` alternative owns one strictly increasing positive `start_days` entry per candidate. It preserves
timed progressive experiments that do not produce a Stockfish ladder, without placing start times on model
definitions or mixing elapsed and Elo fields in one configuration shape. It uses the evaluation manager's persisted
active-run clock, so stopped or preparation time does not advance the schedule.

The final chess configuration uses convolutional stages `12x128`, `14x160`, and `19x176`. Every stage has a
from-to attention policy head, a WDL value head, scaled post-activation residual blocks, and global-pooling context
in every second block. Training adds next-policy and remaining-game-length auxiliary heads; deployment strips those
training-only heads. The complete definitions, including branch scales and head widths, live in the final YAML and
are not duplicated here as another configuration source.

## Quantum and replay semantics

At a progressive training boundary the coordinator pauses the self-play workers selected by the configured topology.
It freezes a typed replay-batch identity containing the canonical `ReplayDescription` and global source optimizer
step. Before the Elo latch, only the active model trains. After the latch, the active model and its immediate
successor receive those same values. The existing
deterministic rank sampler consequently chooses the same rows in the same optimizer-step order for every model.
Models may have different execution times but cannot observe different batches.

Under the elapsed alternative, every model whose configured start has passed trains in model order as before.

Each eligible model owns one persistent `TrainerGroup`. Its DDP ranks remain resident across generations and close
only at run shutdown. Newly eligible candidates start their trainer group once, so ordinary quantum transitions do
not repeatedly pay process startup, checkpoint loading, CUDA-context creation, or compilation costs.

After a promotion, the superseded smaller model stops training and the staged plateau state resets for the new
active model's immediate successor. That successor starts from scratch at model-local generation zero only after its
own stage latches. Only one successor catches up at a time; later configured stages wait until they become the
immediate successor. They do not replay historical batches or receive active-model weights.

The active model's learning-rate schedule uses the run's global generation. An eligible successor trains at the
configured catch-up learning rate until it is promoted, then uses the learning rate for the current global
generation on its next quantum. Each model still owns persisted local optimizer progress for warmup, checkpoint
generation, and recovery. There is no mutable PyTorch scheduler object. All models share the game-owned resolved
objective and auxiliary target layout for a quantum.

## Promotion semantics

For the active model and only its immediate successor, the runtime records total training loss after each shared
quantum and updates paired EMAs:

```text
ema_next = decay * ema_previous + (1 - decay) * observed_total_loss
```

The first shared observation initializes each EMA directly. A successor is promotable only after the configured
number of paired observations. It promotes when:

```text
candidate_ema <= active_ema * maximum_relative_loss
```

The configuration chooses this tolerance. The final chess recipe uses `1.002`, so candidate loss may be at most
0.2% above active loss after ten paired warmup quanta. Comparisons occur strictly in stage order. If the active model
changes, a later candidate's paired comparison resets so the candidate and new active EMA cover exactly the same
quanta.

## Persistence, publication, and recovery

Every model has a private `models/<model-id>` checkpoint namespace. A complete private checkpoint contains the full
training model including auxiliary heads, optimizer state, trimmed policy/WDL inference model, and manifest. Rank
zero writes a random generation-zero checkpoint before a newly eligible model accepts training, making even its
initialization restartable.

`progressive-training.json` atomically persists:

- active model ID;
- the candidate-start policy variant and, for Elo plateau starts, EMA Elo, latest applied boundary, instantaneous
  EMA gain rate, confirmation count, and latch; staged state additionally records its target candidate and latest
  raw observation;
- every model's optimizer progress, latest checkpoint, training-loss EMA, and paired promotion EMA;
- a pending quantum's exact replay identity and ordered required model IDs;
- each completed model result and comparable total loss.

After each model result the pending record is saved. A crash resumes at the first incomplete model without repeating
completed candidates or allowing replay ingestion. A changed replay identity is a fatal restart error.

Once all candidates finish, the selected active private checkpoint is copied to the ordinary global generation
namespace and its manifest is written last. The progression state is then completed and the credit ledger is
committed. This order makes recovery idempotent across crashes before publication, between publication and state
completion, or between state completion and credit commit. Self-play and evaluation receive only the ordinary
published reference.

Private retention keeps the exact latest checkpoint for every candidate and any checkpoint named by a pending
quantum; older private model, optimizer, inference, and manifest files are removed. Ordinary published checkpoint
retention remains unchanged.

The coordinator delegates the complete quantum to a `TrainingSession`. Fixed training and progressive training are
separate implementations with typed result variants; the coordinator pauses the workers selected by the experiment
topology, commits one publication, transitions self-play immediately, and then hands the already-collected outcome
to `TrainingReporter`. Replay ingestion is not part of quantum finalization. The coordinator appends every currently
sealed replay shard at the beginning of each outer-loop iteration, before worker supervision and the decision to
train. Materialization continues in separate long-lived processes, while the replay manager holds the captured
store description immutable for the blocking training quantum.
`ProgressiveTrainingSession` owns
candidate ordering, private checkpoint import, shared-replay training, promotion, publication, and recovery.

Evaluation startup is explicit. On restart, persisted evaluation jobs remain dormant until any pending progressive
quantum has completed and its active checkpoint has been published. The manager then reconstructs the chronological
terminal prefix of primary ladder boundaries from durable result files, and the progressive state applies only
boundaries newer than its persisted EMA. This closes the crash window between evaluation result publication and
candidate-start persistence without repeating an observation.

## Telemetry

Each model writes separate TensorBoard series below `progressive_models/<model-id>/`, including policy, WDL, total
loss, gradient norm, local optimizer steps, and quantum duration. `progressive/active_model_index` records the
published stage. Elo plateau starts require five consecutive evaluation observations below the current stage's gain
threshold. Candidate-start state is persisted, so restart neither clears an in-progress confirmation count nor
repeats an already applied boundary. Candidate-start telemetry records
`progressive/candidate_start/ema_elo`, `instantaneous_ema_gain_per_hour`,
`minimum_worthwhile_gain_per_hour`, `consecutive_below_threshold_observations`, and `latched` at elapsed evaluation
boundaries. The Elo series uses the runtime's bias-corrected 0.90 EMA, and the gain rate is the change between
consecutive corrected EMA values divided by the elapsed boundary interval.
Evaluation and self-play series remain attached only to the globally published model generation.
