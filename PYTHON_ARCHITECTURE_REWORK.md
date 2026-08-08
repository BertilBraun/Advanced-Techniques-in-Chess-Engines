# Python architecture rework

## Status and purpose

This document defines the target Python architecture for the experiment runtime. It is an implementation plan, not a compatibility specification for the current Python structure.

Implementation status: Phase 1 is `accepted`. Phase 2 is `in_progress`. Phases 3 and 4 remain `pending` and are not authorized.

The rework should replace the current mixture of file mailboxes, commander-owned component details, trainer-owned replay maintenance, copied Python replay snapshots, and exact recovery machinery with a small synchronous coordinator and explicit process boundaries.

The design priorities, in order, are:

1. simple ownership and readable control flow;
2. one canonical typed configuration for every concept;
3. high-throughput self-play, replay ingestion, and DDP training;
4. straightforward restart from mmap, ledger, and checkpoint files;
5. deterministic behavior where it affects experiments;
6. approximate rather than transactionally exact recovery of replay samples and credits.

Small replay or credit discrepancies after an unclean exit are acceptable. The runtime must not add complicated transactional recovery to prevent the loss or duplication of a negligible number of samples.

Implementation starts from cleanup checkpoint `66a0a8b`.

## Replacement policy

This rework replaces the production runtime in place. It must not add a second mmap replay beside the existing replay, a second trainer beside the existing trainer, or Go-only orchestration beside chess orchestration.

The implementation may temporarily make the end-to-end pipeline unavailable inside an authorized phase, but the phase cannot finish until:

- both chess and Go use the same shared orchestration path;
- the superseded production path is deleted;
- tests target the authoritative implementation rather than duplicate old/new modes;
- configuration exposes only the authoritative design;
- no `legacy`, `native`, `optimized`, `mmap`, or similar implementation selector remains when there is only one implementation.

Feature-sized commits remain required within a large phase. They should move complete ownership slices toward the final design rather than preserve two runnable architectures.

## Final runtime topology

The coordinator is the main process. Cheap synchronous orchestration components are ordinary objects owned by it. Separate processes are reserved for genuinely parallel work.

```text
Coordinator process
├── ReplayManager object
├── CreditLedger object
├── TrainerGroup object
│   ├── persistent DDP rank 0 process
│   ├── persistent DDP rank 1 process
│   └── additional persistent DDP rank processes
├── self-play worker processes
└── short-lived evaluation processes
```

The coordinator sees one `TrainerGroup`, never individual trainer ranks. `TrainerGroup` is a local facade that owns rank processes and their connections.

There is no:

- replay-manager process;
- publisher component;
- persistent evaluation process;
- generic communication abstraction;
- file-mailbox control protocol;
- explicit coordinator lifecycle-state model;
- complete Python replay snapshot sent to trainer ranks;
- non-DDP training implementation.

## Canonical progress

Completed optimizer steps are the only persisted training-progress axis.

```python
class TrainingProgress(FrozenModel):
    completed_optimizer_steps: int
```

Model generation is derived from the configured optimizer steps per quantum:

```text
model_generation = completed_optimizer_steps // optimizer_steps_per_quantum
```

`optimizer_steps_per_quantum` is static for the complete experiment. At every coordinator boundary:

```python
assert completed_optimizer_steps % optimizer_steps_per_quantum == 0
```

For a quantum:

- the source generation is the current derived generation;
- the target generation is the source generation plus one;
- replay, credit, learning-rate, and objective schedules are evaluated for the source generation;
- self-play schedules are evaluated for the target generation while workers load the resulting checkpoint.

Model generation is exposed as a property of the ledger. It is not independently persisted and cannot disagree with optimizer progress.

## Configuration-owned generation schedules

Every schedule is defined in the immutable experiment configuration. Components retain and evaluate their canonical schedule objects directly. The implementation must not introduce a scheduled configuration plus a second fixed or effective configuration with the same fields.

Only genuinely schedulable values use schedule types. Static configuration remains ordinary typed fields.

### Generic schedule forms

Constant and staged schedules are generic over their value type:

```python
ScheduleValueT = TypeVar('ScheduleValueT')


class ConstantSchedule(FrozenModel, Generic[ScheduleValueT]):
    kind: Literal['constant']
    value: ScheduleValueT

    def value_at(self, model_generation: int) -> ScheduleValueT:
        ...


class GenerationStage(FrozenModel, Generic[ScheduleValueT]):
    start_generation: int
    value: ScheduleValueT


class StagedSchedule(FrozenModel, Generic[ScheduleValueT]):
    kind: Literal['staged']
    stages: tuple[GenerationStage[ScheduleValueT], ...]

    def value_at(self, model_generation: int) -> ScheduleValueT:
        ...
```

Staged schedules require:

- at least one stage;
- the first stage at generation zero;
- strictly increasing, unique start generations.

Linear schedules are generic over numeric start and end values and use an explicit rounding policy:

```python
class ScheduleRounding(StrEnum):
    NONE = 'none'
    FLOOR = 'floor'
    NEAREST = 'nearest'
    CEILING = 'ceiling'


NumericScheduleValueT = TypeVar('NumericScheduleValueT', int, float)


class LinearSchedule(FrozenModel, Generic[NumericScheduleValueT]):
    kind: Literal['linear']
    start_generation: int
    end_generation: int
    start_value: NumericScheduleValueT
    end_value: NumericScheduleValueT
    rounding: ScheduleRounding

    def value_at(self, model_generation: int) -> NumericScheduleValueT:
        ...
```

The typed specializations enforce:

- `LinearSchedule[float]` uses `rounding: none` and returns `float`;
- `LinearSchedule[int]` uses `floor`, `nearest`, or `ceiling` and returns `int`;
- the end generation is greater than the start generation;
- the result is clamped to the start value before the range and the end value after it.

The implementation may use precise type aliases such as:

```python
FloatGenerationSchedule: TypeAlias = (
    ConstantSchedule[float]
    | StagedSchedule[float]
    | LinearSchedule[float]
)

IntegerGenerationSchedule: TypeAlias = (
    ConstantSchedule[int]
    | StagedSchedule[int]
    | LinearSchedule[int]
)
```

Closed-set or Boolean values use constant or staged schedules, not linear interpolation.

Example linear integer configuration:

```yaml
maximum_game_plies:
  kind: linear
  start_generation: 0
  end_generation: 20
  start_value: 100
  end_value: 400
  rounding: nearest
```

Example fade-out:

```yaml
early_termination_probability:
  kind: linear
  start_generation: 0
  end_generation: 20
  start_value: 1.0
  end_value: 0.0
  rounding: none
```

Example staged learning rate:

```yaml
learning_rate:
  kind: staged
  stages:
    - start_generation: 0
      value: 0.005
    - start_generation: 60
      value: 0.002
    - start_generation: 250
      value: 0.001
```

### Schedule ownership and evaluation

Schedules live below the experiment component that consumes them:

```text
experiment
├── replay schedules
├── credit schedules
├── training schedules
├── game-specific self-play schedules
├── game-specific objective schedules
└── evaluation schedules
```

Each component evaluates its schedules once at its natural generation boundary:

- `ReplayManager.ingest_available_games(generation)` applies replay capacity before ingestion;
- `CreditLedger.add_samples(count, generation)` applies the credit policy for newly ingested samples;
- `TrainerGroup.train_quantum(replay, progress)` resolves learning rate and objective once before dispatching the quantum;
- self-play workers resolve search and game-policy parameters once while loading the target generation;
- evaluation scheduling resolves the due jobs after a checkpoint completes.

Learning rate, objective weights, and other trainer parameters remain fixed for the entire quantum. Schedule evaluation must not be added to the optimizer hot loop.

Static values that affect artifact or process compatibility cannot be scheduled. These include:

- network architecture and dimensions;
- auxiliary target definitions, head shapes, and mmap layouts;
- game representation layout;
- mmap slot layout and maximum capacity;
- action-ID and visit-count widths;
- optimizer type;
- global and local batch size;
- optimizer steps per quantum;
- DDP topology and devices.

## Coordinator ownership

The coordinator owns:

- the public run loop;
- `ReplayManager`;
- `CreditLedger`;
- the `TrainerGroup` facade;
- self-play process handles and duplex connections;
- evaluation process handles;
- checkpoint activation;
- process startup, failure handling, and shutdown;
- wall-time, cost, disk, memory, and other run limits.

It does not:

- materialize replay samples itself;
- inspect mmap rows;
- own a loaded model or optimizer;
- communicate with individual DDP ranks;
- calculate game-specific schedules;
- run evaluation work.

### Public loop

The public loop stays at one level of abstraction:

```python
def run(self) -> None:
    while not self.ledger.training_complete:
        self._collect_completed_evaluations()
        self._ensure_self_play_workers_are_running()
        self._ingest_available_games()

        if self.ledger.can_train_quantum(self.replay_manager.live_samples):
            self._train_next_generation()
        else:
            self._wait_for_self_play()

    self._shutdown()
```

Focused helpers contain the necessary sequencing without adding a lifecycle-state framework.

```python
def _ingest_available_games(self) -> None:
    ingestion = self.replay_manager.ingest_available_games(
        self.ledger.model_generation
    )
    if ingestion.samples_added:
        self.ledger.add_samples(
            ingestion.samples_added,
            self.ledger.model_generation,
        )
        self.ledger.save()
```

```python
def _train_next_generation(self) -> None:
    self._pause_all_self_play_workers()

    result = self.trainer_group.train_quantum(
        self.replay_manager.description(),
        self.ledger.progress,
    )

    self.ledger.commit_quantum(result)
    self.ledger.save()

    self._transition_self_play_workers(result.checkpoint)
    self._launch_due_evaluations(result.checkpoint)
```

`TrainerGroup.train_quantum()` is deliberately blocking. The coordinator does not poll self-play, ingest replay, or launch evaluations until the quantum returns. Evaluation processes already running may continue independently.

### Startup, failure, and shutdown

Startup is ordered:

1. load and validate the immutable resolved run configuration;
2. open or create the replay mmap;
3. load the atomic ledger and select its latest complete checkpoint, adopting a newer complete manifest only when
   its optimizer-step generation is the unique next valid quantum;
4. create `TrainerGroup`, whose ranks load the complete training model and optimizer checkpoint;
5. start self-play workers and send a running desired state with the active checkpoint and no statistics request;
6. wait for every worker to acknowledge the exact generation and inference hash;
7. create the local evaluation manager and enter the public loop.

Generation zero is created through the normal checkpoint writer before the run begins and contains the random
training-model state, optimizer state, trimmed inference artifact, and manifest. There is no special checkpoint shape
for generation zero. Run preparation invokes that shared writer once because no trainer rank exists yet; rank zero is
the sole writer for every trained checkpoint thereafter.

Failure handling stays small:

- an unexpectedly exited self-play worker is restarted at the ledger's active checkpoint; its in-memory games and
  unpublished samples may be lost;
- a DDP-rank failure terminates the complete trainer group, leaves the quantum uncommitted, ignores artifacts without
  a final manifest, and ends the run for explicit restart rather than attempting an in-process distributed retry;
- a malformed completed-game file is a fatal boundary error and remains available for inspection;
- an evaluation failure is written as a typed failed result and does not stop training;
- connection EOF is treated as process failure; no heartbeat message or generic liveness protocol is added.

Shutdown sends the stopped desired state to self-play workers and joins them, closes or terminates outstanding
evaluation jobs according to evaluation configuration, closes all trainer ranks, flushes and closes replay, saves
the ledger, and then exits. Process termination is reserved for children that do not honor the ordinary close path.

## ReplayManager

`ReplayManager` is an ordinary synchronous object in the coordinator process.

```python
@dataclass(frozen=True)
class ReplayIngestion:
    games_ingested: int
    samples_added: int
    live_samples: int
    evicted_samples: int


class ReplayDescription(FrozenModel):
    path: Path
    head: int
    size: int
    logical_capacity: int
    maximum_capacity: int
    layout: ReplayLayout


class ReplayManager:
    @property
    def live_samples(self) -> int:
        ...

    def ingest_available_games(self, model_generation: int) -> ReplayIngestion:
        ...

    def description(self) -> ReplayDescription:
        ...

    def close(self) -> None:
        ...
```

`ingest_available_games()`:

1. evaluates and applies logical capacity for the supplied generation;
2. drains every currently available completed-game inbox file;
3. validates and materializes each game;
4. appends its eligible samples to the mmap FIFO;
5. evicts old rows when required;
6. removes each consumed inbox file only after its rows have been written;
7. writes the final header values and flushes the mmap once after the complete drain;
8. returns aggregate ingestion counts.

There is no game-count or time limit. Replay ingestion is required to be materially faster than self-play production. If draining the inbox becomes slow, that is a performance defect to measure and fix rather than a reason to leave the replay stale.

There is no explicit freeze operation. Once ingestion returns and the coordinator enters blocking `train_quantum()`, no code can mutate the replay. Completed games produced during training accumulate in the inbox until the next coordinator iteration.

## Memory-mapped replay store

The replay is one continuously updated, preallocated mmap file:

```text
┌─────────────────────────────┐
│ simple replay header        │
├─────────────────────────────┤
│ fixed-width replay slot 0   │
├─────────────────────────────┤
│ fixed-width replay slot 1   │
├─────────────────────────────┤
│ ...                         │
├─────────────────────────────┤
│ fixed-width final slot      │
└─────────────────────────────┘
```

The file is allocated for the experiment's static maximum capacity. The scheduled replay capacity changes only the logical FIFO capacity.

The simple header contains:

- magic and schema version;
- game and layout identity;
- maximum capacity;
- logical capacity;
- FIFO head;
- live row count;
- eviction count.

The implementation must validate basic header and file-size invariants. It must not introduce double-buffered metadata, a transactional journal, or exact archive-based replay reconstruction.

Clean shutdown flushes and closes the mmap. After an unclean exit, a small number of missing or duplicated samples and a small credit discrepancy are acceptable.

### Fixed-width replay slot

Each experiment defines a static fixed-width row layout containing:

```text
encoded state
policy entry count
padded action IDs
padded visit counts
WDL target
fixed-layout auxiliary targets and eligibility masks
sample weight
source model generation
source timestamp
```

All games use:

```text
policy_entry_count                        : uint8
action_ids[maximum_policy_entries]        : uint16
visit_counts[maximum_policy_entries]      : uint16
wdl_target[3]                             : float32
root_value                                : float32
sample_weight                             : float32
source_model_generation                   : uint32
source_timestamp                          : float64
```

The row dtype uses explicit little-endian numeric fields with no platform-dependent implicit padding. The layout
computes and persists its exact row byte count and a digest of the ordered field definitions. Auxiliary variants add
their own fixed columns; `next_policy` adds another uint8 count, padded uint16 action IDs/counts, and one uint8
eligibility flag.

Configuration validation requires:

- `maximum_policy_entries` between 1 and 255;
- action count representable by `uint16`;
- configured search visits representable by `uint16`;
- maximum derived model generation representable by `uint32`;
- one fixed auxiliary-target layout for the complete run, including zero-width layouts;
- a replay layout compatible with the selected game representation.

Auxiliary targets are fixed by the resolved experiment, not chosen independently per sample. A run may, for
example, define an auxiliary policy head trained against the next player's search visits. The target layout fixes
the tensor shape, dtype, and eligibility-mask shape for every mmap row and training batch. The game-owned training
objective interprets these values; replay storage and DDP transport do not.

Auxiliary target definitions are a discriminated configuration union, not untyped names or arbitrary dictionaries.
Every implemented variant defines its replay storage, batch tensor, augmentation behavior, network head, loss, and
eligibility semantics. The first required variant is `next_policy`: it predicts the search policy at a configured
positive ply offset, initially one ply. Its mmap representation uses the same padded sparse uint16 action/count
layout as the primary policy, while the shared batch builder expands it into an action-sized probability tensor.
The terminal tail has an ineligible mask and contributes no auxiliary cross-entropy. This avoids storing an
action-sized float vector in every replay row, especially for chess.

```python
class NextPolicyTargetConfiguration(FrozenModel):
    kind: Literal['next_policy']
    ply_offset: int
    loss_weight: FloatGenerationSchedule
```

`ply_offset` is static and positive because it affects target meaning but not row width. `loss_weight` is resolved
once per training quantum and may vary by generation without changing the stored target or network head.

Targets may depend on the complete game trajectory. Position encoding is therefore only a mechanical operation
used by materialization, never the materialization boundary itself. The materializer receives the complete validated
game, reconstructs its ordered positions and search observations, and can derive a sample from earlier or later
trajectory elements. A configured next-policy target, for example, reads the search observation at the following
ply. Completed-game records retain every observation required by the run's target layout even when an observation
is not itself eligible to become a primary replay sample.

The primary WDL target is three float32 values ordered `(win, draw, loss)` from the player-to-move perspective of
the encoded position. A natural categorical result is one-hot; an adjudicated target may be the explicitly defined
soft WDL above. Any configured blend with a search root value is resolved by the generation-specific training
objective, using the separately stored root value; materialization does not silently alter the final game result.

### Configurable policy retention

Maximum retained sparse-policy entries are an experiment configuration value, not a hard-coded chess or Go rule.

For every game, materialization:

1. removes zero-count actions;
2. sorts by descending visit count and then ascending action ID;
3. retains the first `maximum_policy_entries` entries;
4. discards the remainder;
5. stores the retained raw counts;
6. normalizes the retained counts only when building a training batch.

This permits, for example:

- chess with 60 retained entries despite its 1,880-action encoded space;
- Go with all board points plus pass retained;
- experiments that intentionally test tighter/looser policy retention.

Ingestion reports the number of truncated policies and aggregate retained and discarded visit mass for both primary
and auxiliary policies. This telemetry is aggregate only and does not add per-row diagnostic columns.

## CreditLedger

`CreditLedger` is an ordinary coordinator-owned object and the only owner of training permission.

It persists:

- completed optimizer steps;
- earned presentation credits;
- consumed presentation credits;
- available presentation credits;
- the active checkpoint-manifest reference.

It derives model generation from completed optimizer steps.

```python
class CreditLedger:
    @property
    def progress(self) -> TrainingProgress:
        ...

    @property
    def model_generation(self) -> int:
        ...

    def can_train_quantum(self, live_samples: int) -> bool:
        ...

    def add_samples(self, sample_count: int, model_generation: int) -> None:
        ...

    def commit_quantum(self, result: TrainingQuantumResult) -> None:
        ...

    def save(self) -> None:
        ...
```

New samples earn credits according to the credit schedule at their ingestion generation. Previously earned credits are not retroactively revalued when the schedule changes. A quantum consumes the amount configured for its source generation. Surplus credits carry forward. Training permission requires both sufficient available credits and at least one global batch of live replay rows. Configuration requires every scheduled logical replay capacity to be at least the global batch size.

The ledger is atomically saved after each nonempty replay ingestion and completed quantum. Small discrepancies between replay contents and ledger credit after an unclean exit are accepted.

## TrainerGroup and DDP ranks

`TrainerGroup` is a coordinator-owned orchestration object. It is not itself a DDP rank and does not own a loaded model.

```text
Coordinator process
└── TrainerGroup object
    ├── DDP rank 0 process
    ├── DDP rank 1 process
    └── additional DDP rank processes
```

All ranks, including rank zero, are symmetric child processes. This keeps CUDA initialization and distributed collectives out of the coordinator and gives every rank the same lifecycle.

Always initialize DDP:

- NCCL for CUDA;
- Gloo for CPU;
- world size one still uses DDP.

### Persistent rank state

Ranks load the selected model and optimizer checkpoint during `TrainerGroup` initialization and retain them across quanta. Reloading model and optimizer during every quantum is not required and would defeat the value of persistent ranks.

After a coordinator restart, a new `TrainerGroup` starts new rank processes and loads the latest complete checkpoint.

### Public interface

```python
class TrainerGroup:
    def __init__(
        self,
        configuration: ExperimentConfiguration,
        starting_checkpoint: CheckpointReference,
    ) -> None:
        ...

    def train_quantum(
        self,
        replay: ReplayDescription,
        progress: TrainingProgress,
    ) -> TrainingQuantumResult:
        ...

    def close(self) -> None:
        ...
```

Trainer-rank transport is another focused duplex `multiprocessing.Connection` per rank, not the self-play protocol
and not a generic communication module. Its command union contains only `TrainQuantumCommand` and
`StopTrainerCommand`; its response union contains `RankTrainingResult`, `RankTrainingFailure`, and
`TrainerStopped`. Exactly one command is outstanding per rank.

```python
class ResolvedTrainingParameters(FrozenModel):
    learning_rate: float
    objective: ResolvedTrainingObjective


class TrainQuantumCommand(FrozenModel):
    replay: ReplayDescription
    source_progress: TrainingProgress
    target_progress: TrainingProgress
    parameters: ResolvedTrainingParameters
```

`ResolvedTrainingParameters` contains the learning rate and one `ResolvedTrainingObjective` resolved for the source
generation. The latter is a discriminated union of frozen Pydantic chess/Go objective values with loss-construction
methods; it is not a dictionary, schedule container, or duplicate transport model. Every rank receives and uses that
same canonical value before entering the hot loop.

`train_quantum()`:

1. derives the source and target generations;
2. resolves learning rate and game-owned training objective once;
3. sends one typed command to every rank;
4. waits for all ranks or the first failure;
5. validates agreement about completed optimizer steps and generation;
6. combines rank statistics;
7. validates the rank-zero checkpoint reference;
8. returns one aggregate result.

Every rank:

1. maps the replay file read-only;
2. derives deterministic rank-local sample indices;
3. builds only its selected batches;
4. sets the supplied learning rate and objective once;
5. executes the complete optimizer quantum through DDP;
6. returns local statistics.

Rank zero additionally writes:

- complete training-model state, including auxiliary heads;
- optimizer state;
- trimmed policy-and-WDL-only JIT inference model;
- checkpoint manifest, written last.

Rank zero is the sole trained-checkpoint writer because the live model and optimizer exist inside rank processes.
`TrainerGroup` owns the aggregate orchestration and validates the single-writer result. Generation-zero run
preparation is the only ownership exception and uses the identical artifact writer and manifest schema.

### Training result

Results contain file references, not loaded models:

```python
class CheckpointReference(FrozenModel):
    generation: int
    manifest_path: Path
    model_path: Path
    optimizer_path: Path
    inference_model_path: Path
    inference_model_sha256: str


@dataclass(frozen=True)
class TrainingQuantumResult:
    completed_optimizer_steps: int
    checkpoint: CheckpointReference
    statistics: TrainingStatistics
```

## Self-play processes and protocol

Self-play workers remain persistent processes and communicate with the coordinator through one duplex `multiprocessing.Connection` per worker.

The generic file-mailbox implementation and all file command identifiers are deleted.

The coordinator sends desired state, not incremental transition commands. Genuine variants are represented as a
discriminated union so a stopped state cannot accidentally carry a checkpoint:

```python
class StatisticsLevel(StrEnum):
    BASIC = 'basic'
    DETAILED = 'detailed'


class RunningSelfPlayState(FrozenModel):
    kind: Literal['running']
    checkpoint: CheckpointReference
    completed_generation_statistics: StatisticsLevel | None


class PausedSelfPlayState(FrozenModel):
    kind: Literal['paused']


class StoppedSelfPlayState(FrozenModel):
    kind: Literal['stopped']


SelfPlayDesiredState: TypeAlias = (
    RunningSelfPlayState | PausedSelfPlayState | StoppedSelfPlayState
)
```

Running desired state always names the exact checkpoint that must be loaded before acknowledgement.
Re-sending the already loaded checkpoint is idempotent and does not reset games or statistics. Only a running state
requests completed-generation statistics, making invalid paused/statistics combinations unrepresentable.

Only one command is outstanding per worker, and the coordinator waits for acknowledgement before sending another. A separate sequence number is unnecessary. A restarted worker receives a fresh connection.

On a model transition, each worker:

1. finishes the current native search batch;
2. stops scheduling new searches;
3. collects old-generation statistics if requested;
4. resets statistics counters;
5. validates and loads the new JIT model;
6. resolves self-play schedules for the new generation;
7. applies new search and game-policy parameters;
8. resets native search trees;
9. acknowledges the exact generation and JIT hash;
10. resumes only for a running desired state.

```python
class RunningSelfPlayStateApplied(FrozenModel):
    kind: Literal['running']
    worker_id: int
    loaded_generation: int
    loaded_inference_model_sha256: str
    completed_generation_statistics: SelfPlayStatistics | None


class PausedSelfPlayStateApplied(FrozenModel):
    kind: Literal['paused']
    worker_id: int


class StoppedSelfPlayStateApplied(FrozenModel):
    kind: Literal['stopped']
    worker_id: int


SelfPlayStateApplied: TypeAlias = (
    RunningSelfPlayStateApplied
    | PausedSelfPlayStateApplied
    | StoppedSelfPlayStateApplied
)
```

Before training, the coordinator sends every worker a paused desired state, then waits
for every acknowledgement. After training, it sends a running state naming the target checkpoint. Every worker
returns its cheap completed-generation counters; only one or two configured workers
receive `DETAILED` and collect expensive distributions and native diagnostics. Statistics are captured before old
counters are reset, but returned only after the new model is loaded, trees are reset, and the exact generation and
inference hash have been acknowledged. Reports identify worker ID, completed generation, and statistics level so
sampled detailed results are not interpreted as global totals.

Completed games continue to cross the self-play/replay boundary as atomic files. Self-play never writes directly into the mmap.

## Checkpoints and approximate restart

There is no publisher abstraction. A checkpoint becomes complete when rank zero writes its manifest after all artifacts.

The coordinator:

1. receives the complete checkpoint reference from `TrainerGroup`;
2. commits optimizer progress, credit consumption, and active checkpoint to the ledger;
3. sends the checkpoint reference to self-play workers;
4. launches due evaluation jobs.

Restart is intentionally simple:

- `ReplayManager` reopens the mmap and validates its basic header;
- `CreditLedger` loads its latest atomic state;
- the coordinator may adopt the newest complete checkpoint manifest if it is ahead of the ledger;
- `TrainerGroup` starts ranks from the selected checkpoint;
- remaining inbox files are ingested normally;
- incomplete checkpoint artifacts without a manifest are ignored.

Exact replay/archive reconstruction, two-phase quantum publication, and exact credit reconciliation are not required.

Adoption is allowed only for the single next generation with the expected optimizer-step count. The coordinator
applies the same ledger quantum commit, including configured source-generation credit consumption, before saving the
adopted checkpoint as active. It never skips multiple generations or guesses progress from loose artifact files.

## Evaluation jobs

Evaluation details may be refined after the core runtime rework, but the initial boundary is fixed:

- evaluations are short-lived processes;
- one process executes one evaluation job;
- multiple jobs run concurrently, initially up to a configured limit such as 16;
- jobs are launched after a completed training quantum;
- the coordinator checks for finished jobs during later outer-loop iterations;
- each job writes one atomic typed result artifact;
- completed results are logged to TensorBoard and the console when collected.

An ordinary coordinator-owned `EvaluationManager` may keep process handles and focused launch/collection logic:

```python
class EvaluationManager:
    def launch_due_jobs(self, checkpoint: CheckpointReference) -> None:
        ...

    def collect_finished_jobs(self) -> tuple[EvaluationResult, ...]:
        ...

    def close(self) -> None:
        ...
```

It is not a process and does not perform evaluation itself. Opponent ladders, result aggregation, retries, retention, and detailed scheduling can be completed in the evaluation-specific implementation phase without changing the coordinator, replay, or trainer boundaries.

## Target source ownership

The target structure is:

```text
experiment/
    configuration.py
    generation_schedule.py
    run.py

runtime/
    coordinator.py
    ledger.py

self_play/
    process.py
    protocol.py
    worker.py
    completed_game.py

replay/
    manager.py
    store.py
    layout.py
    materialization.py
    targets.py

training/
    group.py
    rank.py
    protocol.py
    trainer.py
    model.py
    inference_export.py
    batch.py
    statistics.py

evaluation/
    manager.py
    process.py
    result.py

games/
    implementation.py
    state.py
    chess/
        configuration.py
        implementation.py
        objective.py
        state.py
    go/
        configuration.py
        implementation.py
        objective.py
        state.py
```

Concrete game implementations own:

- native position construction, rules, transitions, terminal results, and adjudication;
- position encoding into the shared packed-plane layout;
- the actual state, primary-policy, and auxiliary-target symmetry transformations;
- training objective schedules;
- game-specific evaluation jobs.

Shared infrastructure owns:

- schedule mechanics;
- mmap FIFO mechanics;
- configurable top-policy retention mechanics;
- completed-game schema, atomic publication, parsing, trajectory validation, and materialization orchestration;
- credit accounting;
- deterministic rank sampling;
- augmentation selection and transformation orchestration;
- packed-state decoding and canonical batch construction;
- DDP process orchestration;
- shared training-model construction and trimmed inference export;
- checkpoint references;
- self-play desired-state transport;
- evaluation process management.

## Concrete game composition

Each process that needs game behavior constructs one concrete chess or Go implementation once from the resolved
experiment union at its process entry point. The frozen configuration, not a live implementation object or native
handle, crosses spawn boundaries. After that single construction match, shared orchestration does not branch on the
game name.

The root composition owns the concrete state, native template instantiation, and training objective consumed by
shared infrastructure. It does not construct a game-specific self-play loop, completed-game implementation,
materializer, or batch builder:

```python
class GameImplementation(
    ABC,
    Generic[
        PositionT,
        NativeSearchT,
    ],
):
    @property
    @abstractmethod
    def configuration(self) -> ExperimentConfiguration:
        ...

    @property
    @abstractmethod
    def network_dimensions(self) -> NetworkDimensions:
        ...

    @property
    @abstractmethod
    def state(self) -> GameStateContract[PositionT]:
        ...

    @abstractmethod
    def create_native_search(
        self,
        device_id: int,
        checkpoint: CheckpointReference,
        parameters: ResolvedSelfPlayParameters,
    ) -> NativeSearchT:
        ...

    @abstractmethod
    def training_objective_at(
        self,
        model_generation: int,
    ) -> ResolvedTrainingObjective:
        ...
```

Evaluation job construction can be added to this composition during the evaluation phase once the common job contract is concrete. It should not be guessed in the core runtime phase.

The composition is not an adapter between duplicate chess/Go types and shared types. It is the canonical place where each concrete game supplies the behavior that genuinely differs.

The network architecture is constructed by the shared trainer from the resolved network dimensions, network
configuration, and run-fixed target/head layout. A game does not manufacture another model wrapper. Evaluation may
add a game-owned job factory during the evaluation phase because opponent and result semantics genuinely differ.

### Game state contract

Python self-play and materialization operate in encoded action-ID space. Native move objects, move encoding,
move decoding, explicit copying, and hashing are not requirements of shared training infrastructure.

```python
class Player(IntEnum):
    FIRST = 1
    SECOND = -1


class WdlTarget(FrozenModel):
    win: float
    draw: float
    loss: float


class GameStateContract(ABC, Generic[PositionT]):
    @property
    @abstractmethod
    def action_size(self) -> int:
        ...

    @property
    @abstractmethod
    def packed_plane_layout(self) -> PackedPlaneLayout:
        ...

    @abstractmethod
    def initial_position(self) -> PositionT:
        ...

    @abstractmethod
    def legal_action_ids(self, position: PositionT) -> tuple[int, ...]:
        ...

    @abstractmethod
    def child_position(self, position: PositionT, action_id: int) -> PositionT:
        ...

    @abstractmethod
    def current_player(self, position: PositionT) -> Player:
        ...

    @abstractmethod
    def terminal_wdl(self, position: PositionT) -> WdlTarget | None:
        ...

    @abstractmethod
    def adjudicated_wdl(
        self,
        position: PositionT,
        reason: TerminationReason,
    ) -> WdlTarget:
        ...

    @abstractmethod
    def encode_network_input(self, position: PositionT) -> PackedPlanePayload:
        ...

    @property
    @abstractmethod
    def augmentation_count(self) -> int:
        ...

    @abstractmethod
    def transform_replay_targets(
        self,
        sample: ReplaySample,
        augmentation_index: int,
    ) -> ReplaySample:
        ...
```

`encode_network_input()` encodes only one network input. It does not construct value or auxiliary targets.
Trajectory-level materialization calls it for the sampled position after inspecting the complete game.

`child_position()` exposes an immutable logical transition. A concrete implementation backed by a mutable board
library may copy internally. The shared caller never needs a `copy()` operation.

`WdlTarget` validates finite nonnegative values summing to one in fixed `(win, draw, loss)` order. Perspective
reversal swaps win and loss. `terminal_wdl()` returns a one-hot WDL result from the current player's perspective for
a naturally terminal position and `None` otherwise. `adjudicated_wdl()` is used only for configured non-natural
endings such as the generation-scheduled maximum ply cap. Go applies its configured scoring rule and returns one-hot
WDL. Chess computes the native material score from the current-player perspective using pawn 1, knight 3, bishop 3,
rook 5, queen 9, king 0, normalized by the 39-point starting material maximum. A scalar score `s` is converted to
soft WDL with remainder `r = 1 - abs(s)`:

```text
win  = max(s, 0)  + r / 3
draw =               r / 3
loss = max(-s, 0) + r / 3
```

Resignation, when later enabled, has a directly known one-hot result and does not require this hook.

The augmentation hook transforms the state, primary policy, and every auxiliary target whose meaning changes under
the selected symmetry. This is necessary for targets such as a later search policy: its action IDs must receive the
same action permutation as the primary policy. Shared batch code chooses the deterministic augmentation index and
orchestrates application; only the game knows the actual transformation.

The native C++ position is authoritative for both chess and Go self-play, completed-game reconstruction, legal
actions, terminal detection, player-to-move, adjudication inputs, and network encoding. Python-chess is not used as a
second rules engine to approve or reject native games. Before cutover, native chess fixtures must lock down the
intended behavior for checkmate, stalemate, castling, en passant, promotion, insufficient material, repetition, and
move-count draws. The training rule set makes the third occurrence an automatic repetition draw and makes 100
half-moves without a pawn move or capture an automatic draw, with checkmate taking precedence; it does not model a
player's choice to claim. Insufficient-material behavior is the native Stockfish-backed implementation fixed by the
fixtures. Any other discovered Stockfish/FIDE difference is either corrected before acceptance or added explicitly
to these documented training rules; Python and C++ must never silently apply different answers to the same run.

### Native search boundary

There is no separate game-owned Python search policy. The native implementation already has one authoritative
`GameSelfPlaySearch<Game>` template with shared request, result, batch, schedule, inference, refresh, and statistics
semantics. Chess, Go 7x7, and Go 9x9 are compile-time instantiations of that implementation.

Python may declare one structural typing protocol that mirrors the bound surface so the shared worker is precisely
typed. It has no implementation, state, adapter, or runtime dispatch; each pybind class satisfies it directly.

The Python bindings must expose the same coarse surface for every instantiation:

```text
NativeSearch
    new_root(position) -> NativeRoot
    search(requests, collect_statistics) -> NativeSearchBatch
    refresh_model(model_generation, model_path)
    update_search_schedule(parameters) -> bool
    inference_statistics() -> statistics
    arena_capacity
    model_generation

NativeRoot
    position
    is_terminal
    play(action_id)
    reset()
    discount(retained_fraction)

NativeSearchRequest
    root
    full_search

NativeSearchResult
    root_value
    visits
    root
```

The current C++ search algorithm does not require duplication or a new abstraction. The binding presentation does
require normalization: chess currently creates roots from FEN/history and reroots by child index, while Go exposes
its native position and reroots by action ID; Go also omits some schedule/root operations exposed for chess. The
rework binds native chess positions at the same coarse level as Go positions, makes action-ID `play()` canonical,
and uses one templated binding helper to generate the common search/root/request/result surface for every template
instantiation. Bound visits use the shared `action_id`/`visit_count` value rather than chess tuples and Go objects
with different Python representations. Game-specific binding code is limited to naming the concrete classes,
constructing the configured initial position, and selecting the native template instantiation.

### Shared self-play worker

Process control, active-game ownership, batched native search, move selection, search-memory collection,
completed-game publication, desired-state handling, model transitions, and basic statistics are shared.

The shared active game contains only typed data, not game-owned orchestration:

```python
class SearchObservation(FrozenModel):
    ply: int
    model_generation: int
    visits: tuple[SparseSearchVisit, ...]
    root_value: float
    selected_action_id: int
    full_search: bool
    sample_weight: float
    search_budget: int
    minimum_root_visits: int


@dataclass
class ActiveSelfPlayGame(Generic[NativeRootT]):
    root: NativeRootT
    action_ids: list[int]
    observations: list[SearchObservation]
    started_at_seconds: float
```

The same frozen `SearchObservation` value is retained in memory and serialized at completion; there is no duplicate
transport model with renamed fields. Every played search ply is recorded. `full_search` directly determines primary
sample eligibility; fast-search observations remain available to configured trajectory targets. The oldest and
newest model generations are derived from observations rather than stored separately.

The worker resolves one immutable parameter value when it loads a model generation. It contains:

```python
@dataclass(frozen=True)
class ResolvedSelfPlayParameters:
    random_opening_plies: int
    full_search_probability: float
    parallel_searches: int
    full_searches: int
    fast_searches: int
    minimum_root_visits: int
    exploration_constant: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    retained_root_visit_fraction: float
    starting_temperature: float
    final_temperature: float
    greedy_after_ply: int
    maximum_game_plies: int | None
    primary_sample_weight: float
```

Every corresponding authored field is either static or a generation schedule in self-play configuration. Resolution
validates counts, positive temperatures and weights, full-search probability in `(0, 1]`, other probability/fraction
ranges, and that full and fast budgets are
compatible with parallel search. When a maximum ply cap exists, random opening plies must be below it. Values remain
fixed until the next model load.

The shared worker:

1. owns the fixed-size active-game slots and native search instance;
2. constructs one shared request per root and performs one batched native search;
3. records every search observation required by the configured primary and auxiliary targets;
4. marks primary-sample eligibility independently from observation retention;
5. applies the common visit-based temperature or greedy move-selection policy;
6. advances the retained native root by the selected action ID, leaving `root.position` as the sole live position;
7. detects completion by applying the state contract to `root.position` and constructs the typed completed
   trajectory;
8. publishes completed records and replaces finished slots;
9. owns cheap game/process counters and asks only selected workers for expensive native statistics;
10. applies the shared desired-state transition order around statistics, model refresh, schedules, and tree reset.

The ordinary turn algorithm is fully specified:

1. a new game starts from `state.initial_position()`;
2. if configured, a generation-scheduled number of uniformly sampled legal opening actions is applied without
   search and retained in the action sequence; zero is valid and is the default, and a terminal random opening is
   discarded and restarted because it contains no training observation;
3. the native search root is created from the resulting position;
4. each ply independently selects a full search with the configured randomized-playout-cap probability and a fast
   search otherwise;
5. a full-search position is primary-sample eligible; a fast-search position is not, but its observation is retained;
6. before a full search, retained root visits are discounted by the configured retained fraction; fast searches may
   reuse their retained root unchanged;
7. the native batch returns positive sparse visit counts and a scalar root value for every nonterminal root;
8. before the configured greedy ply, the action is sampled from raw visit counts raised to inverse temperature; the
   temperature interpolates from the resolved starting value to the resolved final value over that ply range;
9. at and after the greedy ply, the greatest visit count wins with ascending action ID as the deterministic tie
   break;
10. repeated-move penalties and other chess-only sampling modifications are not part of the base algorithm;
11. the selected action advances the retained root through `root.play(action_id)`;
12. a natural terminal position uses `state.terminal_wdl()`; a configured maximum-ply ending uses
    `state.adjudicated_wdl()` and records that termination reason;
13. a nonterminal search result without a positive visit is an invariant failure, not a silently discarded game;
14. loading a new generation keeps action and observation history but resets every retained tree before play resumes.

All random choices use one worker-local generator seeded from the run seed and worker ID. Search observations record
the actual model generation, budget, and minimum-root-visit setting used at that ply, so a game spanning a model
transition remains unambiguous during materialization. Visit-count preprocessing subtracts that observation's
minimum root visits, removes nonpositive results, and applies deterministic top-N retention independently to every
primary or auxiliary sparse policy.

Resignation calibration, material adjudication, disagreement-prefix starts, and other future research policies do not
expand the base state or search boundary. They are added only when scheduled experiments require them and must be
modeled as explicit configured policies around the shared turn algorithm.

### Shared replay and training boundary

Completed-game persistence is shared. It is not a game implementation component or retained legacy replay logic.
The run configuration already selects the game, rules, representation, action mapping, and source revision, so each
file stores only the trajectory and its production metadata:

```python
class TerminationReason(StrEnum):
    NATURAL = 'natural'
    MAXIMUM_PLIES = 'maximum_plies'
    RESIGNATION = 'resignation'
    ADJUDICATION = 'adjudication'


class CompletedSelfPlayGame(FrozenModel):
    schema_version: Literal[1]
    identity: GameIdentity
    created_at_seconds: float
    generation_seconds: float
    action_ids: tuple[int, ...]
    observations: tuple[SearchObservation, ...]
    final_wdl: WdlTarget
    termination_reason: TerminationReason
```

`final_wdl` is from the player-to-move perspective of the final position. Observations are sorted by unique ply,
each selected action agrees with `action_ids[ply]`, and unsearched configured opening plies simply have no
observation. `GameIdentity` contains the worker ID, a process-instance UUID created on worker start, and a
process-local monotonically increasing game number. This avoids persistent publisher state and cannot collide with
files left by an earlier worker process. The shared worker writes JSON to a sibling temporary file, flushes and
closes it, then atomically renames it to the identity-derived final name. Shared ingestion parses the frozen model
directly. There is no publisher object or game-specific completed-game codec.

Shared materialization reconstructs the complete trajectory through the selected `GameStateContract` and validates:

1. every action is legal at its reconstructed position;
2. every observed visit action and selected action is legal;
3. observation plies are unique, ordered, and agree with the played action;
4. the reconstructed final current player and terminal/adjudication result agree with the record;
5. natural and maximum-ply termination reasons agree with the reconstructed state and configured limit.

For each primary-eligible observation, shared materialization then:

1. encodes that reconstructed position through `encode_network_input()`;
2. converts `final_wdl` into the sampled position's player-to-move perspective;
3. preprocesses and retains the primary sparse policy;
4. evaluates each configured auxiliary target definition against the complete reconstructed trajectory;
5. records target eligibility explicitly when the required future observation does not exist;
6. emits the fixed replay row using the observation's weight, model generation, and game timestamp.

For `next_policy(offset=1)`, sample `i` reads the visits from the observation at ply `i + 1`, independently of that
later observation's primary eligibility. The final searched ply therefore has an ineligible auxiliary mask unless a
later searched observation exists. Position encoding happens only after all trajectory-dependent targets have been
resolved. Its action IDs retain the future position's own player-to-move canonical action space; they are not
reinterpreted as moves by the player at sample `i`. The head is explicitly defined to predict that future canonical
policy, and augmentation applies the selected symmetry in that future action space as well.

Every materialized row has one canonical shared shape:

```python
@dataclass(frozen=True)
class ReplaySample:
    encoded_state: PackedPlanePayload
    policy: SparsePolicyTarget
    wdl_target: WdlTarget
    root_value: float
    auxiliary_targets: tuple[AuxiliaryReplayTarget, ...]
    sample_weight: float
    source_model_generation: int
    source_created_at_seconds: float
```

The auxiliary tuple order is exactly the immutable configuration order. Each discriminated target variant has one
canonical typed replay value; `next_policy` contains a sparse policy plus an eligibility bit. `ReplayLayout` lowers
that static tuple to fixed mmap columns exactly once. A run with no auxiliary heads uses an empty tuple and no
auxiliary columns. The shared batch contains auxiliary tensors and masks in the same canonical order, and the
resolved objective is their sole semantic interpreter.

The canonical batch and training-model output contain no chess-named fields:

```python
@dataclass(frozen=True)
class TrainingBatch:
    states: torch.Tensor
    policy_targets: torch.Tensor
    wdl_targets: torch.Tensor
    root_values: torch.Tensor
    auxiliary_targets: tuple[torch.Tensor, ...]
    auxiliary_eligibility: tuple[torch.Tensor, ...]
    sample_weights: torch.Tensor


@dataclass(frozen=True)
class TrainingModelOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    auxiliary_logits: tuple[torch.Tensor, ...]
```

Tensor tuple order is the configured auxiliary-head order and lengths must match exactly. The primary policy and WDL
losses are soft-target cross-entropies. A configured root-value blend is applied by the resolved objective using the
stored observation root value, the same scalar-to-WDL formula defined above, and its generation-resolved weight; the
stored final WDL remains unchanged in replay. Each auxiliary head supplies its configured masked loss and generation-resolved weight. Sample weights are
normalized to mean one for the selected batch and multiply all eligible per-sample loss contributions.

Shared replay infrastructure owns inbox enumeration, configurable top-policy selection, mmap row storage, FIFO
mechanics, capacity, eviction, mapped views, deterministic rank sampling, and replay telemetry. Shared batch code
selects augmentations, asks the state contract to transform each selected row, decodes packed states, expands and
normalizes sparse policies, copies WDL and fixed auxiliary buffers, pins memory, and transfers batches. There are no
separate chess and Go batch builders.

The shared trainer owns the optimizer hot loop, transfer overlap, gradient handling, DDP reductions, and common
statistics. The game-owned `ResolvedTrainingObjective` owns only model-output interpretation and loss construction
for one resolved generation.

### Network and inference outputs

The training model has a run-fixed output layout: policy logits, three WDL logits, and zero or more configured
auxiliary heads. A next-player-policy experiment adds an action-sized auxiliary policy head and its objective adds
the corresponding masked cross-entropy term.

The training checkpoint and inference artifact are intentionally different models:

- the raw training checkpoint contains the backbone, primary policy head, WDL head, every configured auxiliary
  head, and optimizer state required to resume training;
- the inference artifact contains a copied backbone, primary policy head, and WDL head only;
- auxiliary modules are absent from the inference module, not merely ignored by its `forward()` method;
- the inference module's ordinary `forward()` returns exactly normalized `(policy_probabilities,
  wdl_probabilities)`, preserving the existing C++ contract;
- inference export never removes heads from or otherwise mutates the live DDP training model.

Rank zero constructs the dedicated inference model, copies the backbone and primary-head parameters from the
unwrapped training model, switches it to evaluation mode, applies supported inference fusion, scripts and freezes
it, and writes it atomically. Before writing the checkpoint manifest, export validation compares the inference
model's policy and WDL outputs against the training model's primary outputs on a deterministic sample input within
a small fixed numerical tolerance, applying softmax to training logits for that comparison. The manifest is written only
after the raw model, optimizer, and trimmed inference
artifact are complete and hashed.

This removes auxiliary-head storage and computation from self-play and evaluation and requires no auxiliary-output
change in the C++ inference loader. The same export path is used for runs without auxiliary heads so there is one
checkpoint implementation.

### End-to-end authoritative data path

There is exactly one production path:

1. the selected native C++ position and shared native search produce action-ID visits and root values;
2. the shared Python worker makes the configured move decision and accumulates shared `SearchObservation` values;
3. completion writes one atomic shared `CompletedSelfPlayGame` trajectory file;
4. coordinator-owned `ReplayManager` reconstructs and validates the trajectory through the selected native state
   contract, builds primary and configured auxiliary targets, and appends fixed mmap rows;
5. after all workers pause, the final inbox drain and mmap flush establish the immutable quantum description;
6. every DDP rank maps that file read-only, selects deterministic disjoint indices, applies deterministic game-owned
   symmetries through shared orchestration, and constructs the canonical batch;
7. the shared trainer executes the resolved objective on the complete training model, including auxiliary heads;
8. rank zero writes raw training state, optimizer state, and the trimmed policy/WDL inference artifact, then writes
   the manifest last;
9. the coordinator commits progress and credits, workers load the trimmed artifact and reset trees, completed old-
   generation statistics return with their acknowledgements, and evaluation jobs receive the same inference artifact;
10. games produced after workers resume return to step 1 while files accumulated during blocking training are
    ingested on the next loop iteration.

No loaded model, native position, replay object, dense dataset, or training batch crosses a process boundary. Process
messages carry only typed commands, acknowledgements, statistics, failures, progress, replay descriptions, and file
references. Bulk model and replay data cross through checkpoint and mmap files respectively; completed trajectories
cross from self-play to replay as atomic files.

### No game-specific orchestration forks

The following are prohibited:

- a Go coordinator or Go training lifecycle beside the shared coordinator;
- a chess replay manager and Go replay manager with duplicated FIFO or persistence logic;
- separate chess and Go active-game/process loops;
- game-name branching inside shared replay, trainer, or coordinator code after root construction;
- identical configuration models copied into both concrete variants;
- wrapper conversions that copy fields between equivalent shared and game-specific models.

The first implementation of a new shared boundary must migrate both chess and Go before it becomes authoritative. A feature that is intended for later chess experiments cannot be implemented as permanent Go-only infrastructure.

## Implementation phases

The phases are deliberately broad because replay representation, coordinator ownership, DDP access, and game composition are interdependent. Each phase should remove the path it replaces rather than leave compatibility layers.

### Phase 1: canonical configuration, progress, and game contracts

- add generic constant, staged, and linear generation schedules;
- add rounding validation and deterministic interpolation tests;
- convert scheduled experiment fields to generation schedules;
- make optimizer steps the only persisted progress field;
- derive model generation everywhere;
- validate static optimizer steps per quantum and aligned run limits;
- remove optimizer-step and separately persisted model-generation schedule axes;
- make the root `GameImplementation` the only concrete-game composition used by the runtime;
- establish the action-ID `GameStateContract`, completed-trajectory materialization, fixed target-layout, shared
  self-play, shared batch, and objective contracts above;
- remove move encoding/decoding, copying, and hashing from the shared training requirements;
- define run-fixed auxiliary target/head layouts and their symmetry and eligibility semantics;
- trace both chess and Go through those contracts before accepting them;
- remove duplicate or unused game abstractions instead of adapting them.

The phase ends with current runtime behavior intact but configuration, progress, and concrete ownership ready for one integrated runtime replacement.

### Phase 2: integrated core runtime replacement

- define typed replay layout metadata shared by both games;
- add experiment-configured `maximum_policy_entries`;
- implement deterministic top-policy retention and telemetry;
- implement the single-file fixed-slot circular mmap;
- normalize the C++ pybind surface for chess, Go 7x7, and Go 9x9 native search instantiations;
- bind native positions through common action-ID legal-action, child, terminal, player, and packed-input operations;
- make `NativeRoot.play(action_id)` canonical and remove chess child-index rerooting from self-play;
- expose common search schedule, root reset/discount, model refresh, and statistics operations for every instantiation;
- make the native C++ state authoritative for chess and add rule fixtures for every supported terminal/draw rule;
- replace chess/Go completed-game variants with the shared `CompletedSelfPlayGame` record;
- implement shared complete-trajectory validation and materialization through each state contract;
- retain all search observations required by configured later-position auxiliary targets independently of primary
  sample eligibility;
- build training batches directly from mapped rows through one shared builder;
- transform primary and auxiliary action-space targets consistently under game-defined symmetries;
- add run-fixed auxiliary training heads and discriminated target definitions, beginning with `next_policy`;
- export a separate trimmed policy-and-WDL inference model and preserve the existing two-output C++ model contract;
- move replay ingestion into `ReplayManager` in the coordinator process;
- make each call drain the full inbox;
- apply capacity schedules inside ingestion;
- simplify the ledger to atomic progress, credit, and checkpoint state;
- remove trainer-rank replay ownership;
- remove exact archive rebuild and prepared-publication recovery machinery that no longer has an owner;
- introduce the coordinator-owned `TrainerGroup` facade;
- make all ranks persistent symmetric child processes;
- always initialize DDP, including world size one;
- send small replay descriptions rather than replay objects;
- map the replay read-only in every rank;
- resolve learning rate and objective once per quantum;
- retain model and optimizer across quanta;
- make rank zero the sole checkpoint writer;
- aggregate statistics and failures in `TrainerGroup`;
- remove `broadcast_object_list` replay distribution and the single-rank training branch;
- replace file mailboxes with duplex multiprocessing connections;
- implement desired running, paused, and stopped states;
- replace both concrete active-game loops and game-specific self-play policies with the shared action-ID worker;
- implement the specified opening, full/fast search, eligibility, temperature, greedy, root-reuse, terminal, and
  generation-transition decisions in that worker;
- make the shared worker publish completed records and replace finished slots;
- combine generation statistics, model loading, tree reset, and acknowledgement into one transition;
- configure basic versus detailed statistics workers;
- delete the generic communication module and its tests;
- retain atomic completed-game files as the durable data boundary;
- rewrite the public loop at one level of abstraction;
- implement focused ingestion, training, transition, evaluation, and shutdown helpers;
- keep training synchronous and blocking at the coordinator boundary;
- remove the publisher abstraction and separate lifecycle-state machinery;
- enforce run limits between ingestion/training iterations;
- verify restart from mmap, ledger, and latest complete checkpoint.

The phase is complete only when the old replay container, trainer-owned maintainer, replay-object broadcast, file mailbox, old commander lifecycle, and duplicated chess/Go orchestration are deleted. Both games must complete CPU self-play, replay ingestion, mapped batch construction, a DDP quantum, checkpoint activation, and a generation transition through the authoritative path. Projected and actual replay file size, ingestion throughput, mmap batch-read throughput, and DDP throughput are review evidence.

### Phase 3: concurrent evaluation jobs

- introduce focused evaluation job and result types;
- launch multiple short-lived evaluation processes after due checkpoints;
- collect finished jobs during coordinator iterations;
- write atomic result artifacts;
- log completed results to TensorBoard and the console;
- complete concurrency, device assignment, timeout, opponent, and retention behavior.

### Phase 4: integrated validation and cleanup

- remove superseded modules, names, settings, tests, and documentation;
- validate clean startup, quantum transitions, worker restart, coordinator restart, and shutdown;
- run chess, Go 7x7, and Go 9x9 smoke experiments;
- measure replay ingestion, mmap random reads, DDP throughput, memory use, policy truncation, and evaluation concurrency;
- record target-hardware evidence before further optimization.

## Required validation

The implementation is complete only when tests cover:

- constant, staged, and linear generic schedules for `int` and `float`;
- all four rounding modes and clamped interpolation;
- invalid schedule combinations and stage ordering;
- optimizer-step progress and derived generations;
- configuration round trips for chess and both Go board sizes;
- fixed replay layout and file-size validation;
- native chess rule fixtures and identical chess/Go bound state/search surfaces;
- FIFO insertion, capacity changes, wraparound, and eviction;
- deterministic policy top-N selection and discarded-mass telemetry;
- shared completed-game round trips, trajectory reconstruction, legality validation, and perspective-correct WDL;
- natural one-hot WDL, Go scored adjudication, chess material-to-soft-WDL adjudication, and perspective reversal;
- next-policy target construction, terminal-tail masking, sparse mmap storage, and dense batch expansion;
- consistent augmentation of state, primary policy, and next-policy targets;
- concurrent read-only mmap access from multiple spawned ranks;
- deterministic disjoint DDP sampling;
- DDP world size one and multi-rank training;
- persistent model and optimizer state across quanta;
- trimmed inference export without auxiliary modules and numerical agreement with training primary outputs;
- rank failure propagation and rank-zero checkpoint writing;
- deterministic full/fast eligibility, temperature/greedy selection, root reuse, and worker-local random seeding;
- self-play desired-state pause, model transition, tree reset, statistics, and resume;
- complete inbox draining after acknowledged pause and queued-game ingestion after training;
- approximate restart from mmap, ledger, and complete checkpoint;
- concurrent evaluation launch and finished-result collection;
- complete native and Python regression suites.

Repository validation follows the project commands, including:

```powershell
uv run ruff format
uv run ruff check --fix
python -m pytest --import-mode=importlib .\test -q
```

Native extension and native tests must also run for phases that change bound replay, self-play, inference, or training behavior.

## Explicitly accepted tradeoffs

- A small number of samples or credits may be lost or duplicated after an unclean exit.
- Replay policy entries below the configured top-N cutoff are discarded and the retained visits are renormalized during batch construction.
- The coordinator does not supervise unrelated work while blocked in a training quantum.
- World-size-one DDP overhead is accepted to retain one training path.
- Native C++ chess rules are authoritative for training once their supported draw and terminal semantics are fixed by
  tests; Python-chess is not a runtime cross-check.
- Active games may span model generations; every observation retains its actual producing generation and trees are
  reset on transition.
- Completed-game files are consumed after mmap ingestion rather than retained as an exact replay-rebuild archive.
- Evaluation details may mature after the core runtime boundaries are implemented.
- Replay file size is less important than fixed-width simplicity and shared mmap access, but it remains measured experiment evidence.

These tradeoffs are intentional and must not be reversed by adding defensive compatibility layers, transactional replay recovery, duplicate configuration models, or alternate runtime modes without a new recorded design decision.
