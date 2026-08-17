# Progressive model sizing

## Purpose and scope

Progressive sizing is an optional training policy above the shared chess/Go model, replay, objective, DDP, and
checkpoint contracts. It is independent of the concrete network architecture: each of exactly three configured
stages supplies a complete `NetworkParams` value, and the runtime does not infer width, depth, or a relationship
between architectures. The first model starts at elapsed run day `0.0`; later models have strictly increasing
elapsed active-run starts, such as `0.75` and `1.5` days.

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
- the coordinator waits for all required models before it resumes self-play or starts newly due evaluation jobs;
- promotion compares paired exponential moving averages built only from quanta seen by both the active model and its
  immediate successor, rather than an unspecified lifetime average;
- no match gate decides promotion, and evaluation results remain user-facing evidence only;
- no parameters or optimizer moments transfer between sizes, and no checkpoints are averaged;
- only one active checkpoint is atomically published to self-play and evaluation.

These differences trade KataGo's asynchronous throughput for deterministic comparability and a small exact restart
boundary that fits this platform's blocking DDP quantum.

## Configuration and eligibility

`training.progressive_model_sizing` owns three ordered `ProgressiveModelDefinition` values. Each owns a stable model
ID, an elapsed active-run start in days, and a complete network definition. The first definition must exactly equal
`training.network`, which remains the day-zero published model. This prevents the published initial checkpoint and
the progressive state from naming different architectures.

The promotion configuration explicitly owns:

- EMA decay `d`, constrained to `0 < d < 1`;
- a positive number of paired warmup quanta;
- the maximum candidate-to-active relative loss, normally `1.01`.

Elapsed eligibility uses the evaluation manager's persisted active-run clock. It therefore resumes from accumulated
elapsed time after a stopped experiment and does not count preparation or stopped time.

## Quantum and replay semantics

At a training boundary the coordinator pauses every self-play worker. It freezes a typed replay-batch identity with
the replay path, FIFO head and size, capacities, layout digest, and global source optimizer step. The active model
and every eligible larger model, in configured order, receive that same replay description and source optimizer
step. The existing deterministic rank sampler consequently chooses the same rows in the same optimizer-step order
for every model. Models may have different execution times but cannot observe different batches.

After a promotion, superseded smaller models stop training. The new active model and its immediate or later eligible
successors continue. A later stage that becomes eligible after many quanta starts from scratch at model-local
generation zero and joins the next complete quantum; it does not replay historical batches or receive active-model
weights.

Learning-rate schedules use each model's persisted local optimizer progress. There is no mutable PyTorch scheduler
object: the canonical configured schedule plus exact model-local optimizer steps is the complete scheduler state.
All models share the game-owned resolved objective and auxiliary target layout for a quantum.

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

With the normal value `1.01`, candidate loss may be at most one percent above active loss. Comparisons occur strictly
in stage order. If the active model changes, a later candidate's paired comparison resets so the candidate and new
active EMA cover exactly the same quanta.

## Persistence, publication, and recovery

Every model has a private `models/<model-id>` checkpoint namespace. A complete private checkpoint contains the full
training model including auxiliary heads, optimizer state, trimmed policy/WDL inference model, and manifest. Rank
zero writes a random generation-zero checkpoint before a newly eligible model accepts training, making even its
initialization restartable.

`progressive-training.json` atomically persists:

- active model ID;
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

## Telemetry

Each model writes separate TensorBoard series below `progressive_models/<model-id>/`, including policy, WDL, total
loss, gradient norm, local optimizer steps, and quantum duration. `progressive/active_model_index` records the
published stage. Evaluation and self-play series remain attached only to the globally published model generation.
