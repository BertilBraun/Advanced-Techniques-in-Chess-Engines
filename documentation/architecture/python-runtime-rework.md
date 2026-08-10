# Python runtime architecture and rework record

## Status and purpose

This document defines the authoritative Python architecture implemented by Phases 1 through 3 and records the
remaining Phase 4 validation work. It is not a compatibility specification for removed Python structures.

Implementation status: Phases 1 through 3 and the Phase 4 production-reachability and documentation cleanup slice
are `accepted`. Phase 4's integrated smoke, throughput, concurrency, and target-hardware evidence is in progress
under the authorized R11 rented-node validation pass.
R10 is a separate standalone layer above this runtime and is `awaiting_user_review`; it does not change the Python
training topology defined here.

The accepted rework replaced file mailboxes, commander-owned component details, trainer-owned replay maintenance,
copied Python replay snapshots, and exact recovery machinery with a small synchronous coordinator and explicit
process boundaries.

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
└── game-specific objective schedules
```

Each component evaluates its schedules once at its natural generation boundary:

- `ReplayManager.ingest_available_games(generation)` applies replay capacity before ingestion;
- `CreditLedger.add_samples(count, generation)` applies the credit policy for newly ingested samples;
- `TrainerGroup.train_quantum(replay, progress)` resolves learning rate and objective once before dispatching the quantum;
- self-play workers resolve search and game-policy parameters once while loading the target generation;
- `EvaluationManager` owns its separate elapsed-time cadence and evaluates it on every coordinator-loop iteration.

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
        self.evaluation_manager.collect_completed_jobs()
        self.evaluation_manager.schedule_due_jobs(
            self.ledger.active_checkpoint,
            self.run_clock.elapsed_seconds,
        )
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
3. sends the checkpoint reference to self-play workers.

The evaluation manager observes that checkpoint on the next coordinator-loop iteration. Checkpoint completion does not itself define the evaluation cadence.

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

Evaluation measures the best completely published model available at fixed elapsed-time boundaries. It is not
scheduled by generation or optimizer step because experiments with different models, objectives, or augmentation
costs may advance those axes at materially different rates. Hardware and the evaluation ladder are fixed for a
comparison set, so elapsed training time is the comparison axis.

### Replaced stack audit

Before Phase 3, the repository had no coordinator-integrated evaluation lifecycle. `experiment/run.py` validated
chess evaluation paths and hashes, but `training/coordinator.py` never launches evaluation. The only production-like
entry into `games/chess/evaluation/process.py` is the standalone `benchmark_evaluation_suite.py` tool. Its
`EvaluationProcess` then creates another layer of multiprocessing tasks, selects devices through a round-robin
integer cycle, and directly logs metrics. The runtime, protocol, scheduling helper, benchmark tool, and their tests
therefore mostly exercise one another rather than the current master coordinator.

Within that chess-only graph:

- `experiment/evaluation_protocol.py` is the authoritative paired-opening/report model for the old stack, but embeds
  chess FEN and duplicates concepts that Phase 3 needs for both games;
- `games/chess/evaluation/model.py`, `paired_match.py`, `types.py`, and `process.py` split one semantic match across
  callable adapters, duplicate move/terminal types, the chess board, model loading, native C++ search, random play,
  historical checkpoints, Stockfish via `python-chess`, aggregation, JSON, and TensorBoard;
- fixed-dataset evaluation loads the legacy HDF5 `SelfPlayDataset`, whose missing-file builder downloads October 2024
  Lichess human games through `database.py`; it measures policy top-1/top-5/top-10 and value loss against training
  samples rather than the engine-search policy metrics selected for Phase 3;
- `prepare_opening_suite.py` samples FENs from an external archive into static TSVs; it does not generate or version
  likely engine lines;
- `evaluation_schedule.py` contains only device cycling and historical-generation selection and has no coordinator
  owner;
- `plateau.py` and `evaluate_plateau.py` are referenced only by their CLI/tests and have no live training caller;
- Go exposes evaluation-shaped configuration fields but has no Go evaluator, external engine, dataset path, or match
  implementation;
- `run_cutechess_gauntlet.py` is a separate manual Cute Chess command builder, not part of Python model inference or
  the coordinator lifecycle.

The reusable authoritative pieces are narrower: complete checkpoint/inference artifacts, the normalized native C++
state/search bindings used by self-play, the game state contracts, and the shared TensorBoard writer. Phase 3 builds
on those and deletes the rest of the graph.

The initial cadence is 20 minutes. The coordinator asks the evaluation manager to observe elapsed time and the
current complete checkpoint on every outer-loop iteration, before it may enter a blocking training quantum. For each
due boundary, the manager selects the newest checkpoint whose publication time is at or before that boundary and
schedules its complete evaluation suite. If a boundary is crossed while the coordinator is blocked in training,
the next outer-loop iteration still selects the checkpoint that was available at the boundary, not the checkpoint
published after it.

Elapsed time is the persisted monotonic active-run clock used by the coordinator: it starts after run preparation,
accumulates across coordinator restarts, and excludes time while the experiment is stopped. Dataset and opening
generation therefore do not consume training time. Self-play waits, training, checkpoint publication, and the
contention caused by the fixed evaluation topology do consume it. When the coordinator is waiting for replay rather
than training, its wait timeout is bounded by the next evaluation boundary so scheduling is not delayed by an
unbounded self-play wait.

If one blocking quantum crosses several boundaries, the manager schedules each due boundary with the checkpoint
that was available at that boundary. Ordinarily the quantum is shorter than the cadence, so this does not create a
large backlog.

The manager is an ordinary coordinator-owned object, not a process. It owns scheduling, pending jobs, process
handles, the next device-cycle index, completion, failure publication, and reporting:

```python
class EvaluationManager:
    def schedule_due_jobs(
        self,
        checkpoint: CheckpointReference,
    ) -> tuple[EvaluationJob, ...]:
        ...

    def collect_completed_jobs(self) -> tuple[EvaluationResult, ...]:
        ...

    def close(self) -> None:
        ...
```

`schedule_due_jobs()` returns quickly. Every configured dataset or opponent comparison becomes its own short-lived
evaluation job process. Due jobs enter one FIFO owned by the manager, which starts up to the configured maximum
concurrency and starts the next queued job whenever a process finishes. Jobs from one evaluation suite are therefore
concurrent rather than one suite process running categories sequentially. A job may
itself batch many active games through the native C++ search. The manager never loads a model, plays a game, reads a
dataset row, or controls an external engine.

The coordinator loop remains at one level of abstraction:

```python
while not self.ledger.training_complete:
    self.evaluation_manager.collect_completed_jobs()
    self.evaluation_manager.schedule_due_jobs(
        self.ledger.state.active_checkpoint,
    )
    self._ensure_self_play_workers_are_running()
    self._ingest_available_games()

    if self.ledger.can_train_quantum(self.replay_manager.live_samples):
        self._train_next_generation()
    else:
        self._wait_for_self_play()
```

An evaluation process receives the frozen experiment configuration, one resolved `EvaluationJob`, and one assigned
device ID. It constructs the concrete game once, validates all referenced artifacts, runs the job, writes one atomic
typed result, and exits. No loaded model, native position, dataset object, or dense tensor crosses the spawn
boundary. The deterministic job identity and result path are derived from the run, requested elapsed boundary,
definition ID, and candidate checkpoint.

### Evaluation configuration and jobs

Each chess or Go experiment owns one evaluation configuration composed from the same canonical shared types. It
contains:

- the elapsed cadence, initially 1,200 seconds;
- a nonempty tuple of evaluation device IDs owned by the shared training topology;
- one 1,200-second per-job deadline and shutdown grace period;
- one maximum active-job count, initially 10;
- a tuple of uniquely named evaluation definitions;
- game-specific Stockfish or KataGo artifact configuration where used.

The definition union contains only genuine job variants:

```text
EvaluationDefinition
├── FixedDatasetEvaluationDefinition
├── RandomOpponentEvaluationDefinition
├── PolicyRandomOpponentEvaluationDefinition
├── PreviousCheckpointEvaluationDefinition
├── ReferenceCheckpointEvaluationDefinition
├── StockfishEvaluationDefinition
└── KataGoEvaluationDefinition
```

Checkpoint opponents select either a positive previous-evaluation-boundary offset or the same elapsed boundary from
an explicit reference-run manifest.
Job construction resolves that selector to the existing canonical `CheckpointReference`; process messages do not
carry a second model-reference representation. The initial ladder uses offsets one, two, and three, showing progress
against the checkpoints selected 20, 40, and 60 minutes earlier. Older offsets alternate explicitly: 80, 120, 160,
and 200 minutes run on even-numbered boundaries, while 100, 140, 180, and 220 minutes run on odd-numbered boundaries.
The reference manifest records the checkpoint selected at every elapsed boundary, allowing a candidate to play the
baseline available at the same training time. Fixed-generation opponents are removed. There is no inspection/full
tier, teacher-evaluation special case, or plateau rule.

Chess additionally runs Stockfish skill levels 0, 1, 2, and 3 at every boundary and retains its search/policy random
diagnostics. Go uses fixed-dataset evaluation, historical/reference checkpoints, and KataGo at 64 visits;
its saturated random-opponent diagnostics are removed from the screening ladder.

Every resolved job records:

- its deterministic identity and definition ID;
- requested boundary seconds, which are also the reporting step;
- the candidate `CheckpointReference`;
- its deterministic seed and deadline;
- the assigned evaluation device ID;
- fixed model search and inference parameters;
- the exact dataset or opening-suite reference and any resolved opponent checkpoint required by that variant. The
  frozen experiment supplied to the child remains the canonical owner of external executable, model, and protocol
  configuration.

### Parallelism and device assignment

All definitions due at one boundary become independent jobs. The manager assigns
device IDs by cycling through the configured nonempty evaluation-device tuple in stable job order. If there are more
jobs than active slots, excess jobs wait in FIFO order; their deadlines start only when their processes launch. If
there are more active jobs than devices, several jobs intentionally share a device. The child activates its assigned device before Torch
initializes CUDA models or the native extension constructs a search. This is the complete Phase 3 assignment policy.
Every job has the same 20-minute timeout as the cadence. The manager terminates a job that exceeds it, writes a typed
deadline failure artifact, and logs the failure explicitly. The coordinator collects these failures before scheduling
due jobs on each outer-loop iteration.

Dataset, both random-opponent variants, and external-engine matches load one project inference model. Checkpoint
matches load two and must be sized accordingly during target-hardware validation. Stockfish consumes additional CPU.
A GPU-backed KataGo process also consumes the configured accelerator. Its device therefore belongs in the same explicit evaluation
device tuple and may contend with project inference jobs assigned there by the simple cycle. The KataGo subprocess
sees only its assigned device through its private environment. Evaluation uses the training topology's CPU/CUDA
device type; Phase 3 does not add a second placement policy.

### Shared match execution

One shared asynchronous match runner owns active games, initial-state reconstruction, player assignment, candidate
action selection, random play, maximum-ply handling, result perspective, and aggregation. For each opening it plays
a color/player-swapped pair. Search candidates and checkpoint models use fixed search visits, no root noise, zero
action temperature, and ascending action ID as the deterministic tie break. The policy-only candidate batches the
same active turns through the trimmed inference artifact, masks to legal actions, and greedily chooses the highest
policy probability with ascending action ID as the tie break. It does not construct native search.

Within a search job, all candidate turns are submitted together to one native C++ search; within a policy-only job,
they are submitted together to one direct Torch inference batch. All checkpoint-opponent turns are submitted
together to the opponent's separate native search. Model-versus-model therefore owns two
inference models but retains batching within each model. Random play uses a job-local generator. Each external-engine
job owns exactly one engine subprocess. That process is reused for the full job rather than restarted for every move
or game. No external-engine process is shared between evaluation jobs.

Every opening produces two games with candidate player order swapped. Seeds are derived from the run seed,
definition ID, requested boundary, pair index, and player stream. Results store the job seed and pair seed; they do
not persist a separate seed value for every ply.

Chess uses a draw at the configured maximum ply. Go uses its configured area-score adjudication. Both policies are
explicit configuration validated by the concrete game. Resignation is disabled initially for project models and
external engines so engine-specific resignation behavior does not distort comparisons.

### Engine-generated opening suites

Match openings are immutable, inspectable artifacts generated before the coordinator starts. If the configured
artifact does not exist, run preparation generates it; if it exists, run preparation validates its manifest and
hashes and reuses it. A mismatched artifact is an error and is never silently overwritten. Creating a new suite
requires a new version or output path.

The runtime representation is an action-ID sequence from the game's configured initial position. The manifest also
stores the engine/build configuration, source revision, rules, representation identity, path probability, final
packed-position digest, and a human-readable rendering. The manifest embeds PGN for chess and SGF for Go so the
selected openings can be inspected without decoding action IDs.

The initial builder uses deterministic engine-guided beam expansion:

1. start from the configured initial position;
2. expand four plies, corresponding to two full chess moves or four alternating player actions;
3. ask the configured external engine for a sparse move distribution at every retained frontier position;
4. add log move probability to the path score;
5. expand only the configured top moves and retain a bounded configured beam, rather than attempting the enormous
   exhaustive chess tree;
6. merge transpositions by packed-position digest, retaining the highest-probability action sequence with ascending
   action IDs as the tie break;
7. discard terminal positions;
8. write the 50 highest-probability distinct final positions and their complete action sequences.

The initial settings are four plies, eight expanded actions per frontier position, a 512-position beam,
and 50 output openings. These values are ordinary immutable build configuration and may be adjusted before the
first benchmark. Stockfish MultiPV scores are converted to a documented normalized distribution for this builder;
KataGo supplies its searched move distribution. The suite is generated once, inspected, and then frozen for every
experiment in a comparison set.

### Fixed evaluation datasets

Chess, Go 7x7, and Go 9x9 each use one small engine-labelled fixed dataset containing between 480 and 520 unique
positions. Run preparation generates a missing configured dataset before any training or evaluation process starts
and otherwise validates and reuses it. Evaluation jobs never race to construct shared data.

The builder asks the configured external engine to play complete games against itself under fixed rules and search
limits. It samples moves from the labelled search distribution with one manifest seed and fixed sampling temperature;
generation is therefore diverse but exactly reproducible. The engine plays every source game to completion, but the
builder retains only every third position, using one fixed manifest-recorded ply offset. It continues complete games
until at least 480 unique retained positions exist and stops retaining positions at 520. This keeps opening,
middlegame, and endgame coverage while requiring several source games and keeping evaluation small. Duplicate
positions do not consume the limit.

Each dataset row contains only:

- the canonical packed network input;
- a sparse engine policy target over action IDs;
- the engine-selected action ID;
- a source-game ID and ply for inspection and grouped metrics.

The fixed-dataset job loads the trimmed inference artifact and evaluates raw policy output without candidate MCTS,
augmentation, value targets, game outcomes, training weights, root-value blends, or auxiliary heads. It reports:

- accuracy of the model's top action against the engine-selected action;
- cross-entropy from the complete normalized sparse engine policy to the model policy.

Stockfish uses fixed-node MultiPV search. The builder converts each principal variation's side-to-move WDL expected
score through a configured softmax temperature and renormalizes over the reported legal moves; the exact conversion,
MultiPV width, and temperature are part of the dataset version. KataGo uses the unique `order: 0` move as its selected
action and separately normalizes searched move weights as the soft policy label. These soft
targets make cross-entropy distinct from top-action accuracy. Human moves, played outcomes, engine values,
calibration metrics, and symmetry-consistency datasets are optional later research additions, not part of the
initial Phase 3 dataset.

The immutable dataset manifest contains schema and dataset version, game and exact rules, packed representation,
position count, complete inspectable source-game records, engine executable/model/configuration hashes, search limits,
deterministic seed, builder source revision, row-layout digest, and data-file hash. Dataset data uses one small
fixed-width memory-mappable file. Each row stores a target count plus bounded action-ID and probability arrays, so
the semantic target remains sparse without a variable-size row format. It does not reuse the training replay layout
or the obsolete HDF5 `SelfPlayDataset`.

### External engines

Stockfish uses a clean UCI boundary through `python-chess`, following the
[official integration documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Developers.html). One job reuses its configured UCI subprocess, verifies
its reported identity and options, disables pondering and resignation, and applies fixed nodes, threads, hash,
MultiPV, and optional strength settings. The configured executable is hashed. Protocol failure fails the job; there
is no per-move engine restart. Deployment treats Stockfish as an external GPLv3 executable: node provisioning must
place the explicitly configured binary and preserve the
[corresponding license and source-offer obligations](https://github.com/official-stockfish/Stockfish/blob/master/Copying.txt)
when the binary is redistributed.

KataGo uses its
[asynchronous JSON analysis protocol](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md)
rather than stateful GTP. The job keeps one KataGo process
alive, submits several active positions with stable request IDs, and accepts out-of-order responses. Every request
supplies the complete move history, board size, exact rules, komi, fixed maximum visits, and the requested policy
fields. The unique `order: 0` move is authoritative for match play; `weight` is used only for policy labels and
opening probabilities. This boundary supports batched match play and the offline dataset/opening builders without implementing a
second protocol. The executable, network, backend, and analysis configuration are pinned and hashed. Go coordinate
translation and exact rule composition are the only game-specific protocol concerns. Node provisioning installs the KataGo
executable, a compatible network, and an analysis configuration outside the evaluation child; run preparation only
validates and hashes those configured files. KataGo's repository code uses a
[permissive license with separately licensed bundled dependencies](https://github.com/lightvector/KataGo/blob/master/LICENSE);
the selected network artifact's terms must be reviewed separately before redistribution.

### Results, reporting, and statistics

The result union has three variants:

- `FixedDatasetEvaluationResult`;
- `MatchEvaluationResult`;
- `FailedEvaluationResult`.

Match results contain the exact job, per-game pair/opening/player identities, pair seed, initial and played action
sequences, termination reason, candidate outcome, plies, and duration. The aggregate contains wins, draws, losses,
mean score, first-player score, second-player score, pair count, and a paired-bootstrap score interval. Elo is optional descriptive output, not a training
gate. Dataset results contain the exact job, position count, source-game count, top-action accuracy, cross-entropy,
and duration. Failed results contain the exact job, a closed failure phase (`validation`, `setup`, `execution`,
`deadline`, `cancelled`, or `missing_artifact`), a concise message, child exit code when available, and the path to a
captured traceback. Success and failure share only job identity and timing; unrelated nullable result fields are not
combined into one catch-all model.

Every child writes one result JSON atomically. The manager validates it after process exit, logs a concise console
summary, and writes TensorBoard scalars below `evaluation/<definition_id>/...` at the requested cadence boundary,
regardless of when the job actually finishes. The candidate generation identifies the checkpoint available at that
boundary. It also logs candidate generation and optimizer steps at that elapsed boundary and rewrites one grouped
Markdown summary containing status, score, W-D-L, player-order scores, and duration for direct TensorBoard inspection.
Per-game actions remain in the result artifact rather than being emitted as thousands of TensorBoard text
records.

There is no plateau automation. `experiment/plateau.py`, `tools/evaluate_plateau.py`, and their tests are deleted.
Evaluation results inform user decisions; they do not automatically stop or promote an experiment.

### Replacement and deletion scope

Phase 3 is a replacement, not a second evaluation stack. The following current files have no surviving runtime
authority and are deleted together with imports, configuration fields, and fixtures that exist only for them:

- `py/src/experiment/evaluation_protocol.py`, `evaluation_schedule.py`, and `plateau.py`;
- all of `py/src/games/chess/evaluation/`;
- `py/src/games/chess/evaluation_dataset.py`, `dataset.py`, `data_loader.py`, `database.py`,
  `dataset_statistics.py`, and `self_play_statistics.py` after their remaining Phase 1/2 callers are confirmed gone;
- `py/tools/benchmark_evaluation_suite.py`, `evaluate_plateau.py`, `evaluate_repetition_random.py`,
  `prepare_chess_evaluation_dataset.py`, and `prepare_opening_suite.py`;
- `py/reference/main-monitoring-openings-50.tsv` and `pilot-openings.tsv`, replaced by versioned generated opening
  manifests;
- `py/test/test_evaluation_protocol.py`, `test_evaluation_schedule.py`, `test_evaluation_runtime.py`,
  `test_plateau.py`, `test_evaluate_plateau.py`, `test_prepare_chess_evaluation_dataset.py`, and
  `test_prepare_opening_suite.py`.

`py/tools/run_cutechess_gauntlet.py` and its test remain an optional manual export/compatibility tool; they are not
called by the coordinator and do not define the canonical evaluation contracts. Historical benchmark artifacts and
documents remain archival evidence, but normative evaluation documentation is updated to point here or deleted if
it only describes a removed interface.

The existing coordinator, experiment configuration, run entry point, game implementations, checkpoint retention,
and TensorBoard writer are reworked in place to compose the new shared evaluation package. They do not gain parallel
adapter models. The native C++ search binding remains authoritative; Phase 3 changes C++ only if the shared match
runner exposes a concrete missing batch operation or result field, and deletes any superseded evaluation-only native
entry point rather than retaining compatibility wrappers.

Concretely, Phase 3 reworks `py/src/training/coordinator.py` (or its accepted Phase 2 destination),
`py/src/experiment/configuration.py`, `py/src/experiment/base_configuration.py`, `py/src/experiment/run.py`,
the `py/src/training/checkpoint` package, `py/src/util/tensorboard.py`, `py/src/games/implementation.py`, both concrete game
configuration and implementation modules, and the checked-in experiment configurations. It replaces
the former evaluation optimization note as a normative design document. Training-quantum statistics remain owned by
the `py/src/training/trainer` package and are not generalized into an evaluation result model.

### Failure, restart, shutdown, and retention

Evaluation failure never stops training. An exception caught inside a job writes a typed failed result. A child that
exits without an artifact is converted by the manager into a failed result with its exit status. There are no
automatic retries, retry counters, backoff schedules, heartbeats, or recovery journals.

The manager stores one atomic state snapshot containing the next cadence boundary and the resolved jobs in the
currently scheduled suites. At startup it reads that snapshot and existing valid result artifacts. A job without a
result is relaunched only when all referenced artifacts still exist; otherwise it is recorded as failed. There is no
event journal, attempt history, or reconstruction beyond that snapshot and result scan.

On shutdown the manager stops launching work, waits for the configured short grace period, terminates
remaining evaluation children, and writes cancelled failures for running and queued jobs. Successful and failed
evaluation artifacts are kept for the run. Scheduled elapsed-boundary inference checkpoints are retained so the
run remains usable as a later reference manifest; pending jobs additionally retain every checkpoint they reference.

## Source ownership

The authoritative source structure is:

```text
experiment/
    base_configuration.py
    configuration.py
    generation_schedule.py
    progress_telemetry.py
    resource_telemetry.py
    run.py

self_play/
    completed_game.py
    configuration.py
    native_search.py
    parameters.py
    protocol.py
    worker.py

replay/
    batch_loader.py
    configuration.py
    contracts.py
    layout.py
    manager.py
    materialization.py
    store.py

training/
    batch.py
    configuration.py
    coordinator.py
    credit_ledger.py
    network.py
    objective.py
    progress.py
    run_limits.py
    self_play_group.py
    targets.py
    checkpoint/
        contracts.py
        paths.py
        persistence.py
        retention.py
    trainer/
        contracts.py
        group.py
        rank.py

evaluation/
    configuration.py
    contracts.py
    dataset.py
    engine.py
    manager.py
    match.py
    openings.py
    preparation.py
    process.py
    scheduling.py
    statistics.py

games/
    composition.py
    contracts.py
    implementation.py
    representation.py
    chess/
        contract.py
        configuration.py
        stockfish.py
        training.py
        interactive/
        uci/
    go/
        contract.py
        configuration.py
        katago.py
        training.py
```

Concrete game implementations own:

- native position construction, rules, transitions, terminal results, and adjudication;
- position encoding into the shared packed-plane layout;
- the actual state, primary-policy, and auxiliary-target symmetry transformations;
- training objective schedules;
- construction of the bound native search instantiation for an evaluation job;
- external-engine action/coordinate translation and exact rule composition;
- human-readable opening and game rendering.

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
- elapsed evaluation scheduling and process management;
- active match orchestration, player balancing, seeding, aggregation, and reporting;
- fixed-dataset storage, loading, inference, and metrics;
- engine-guided opening and dataset build orchestration.

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

Evaluation adds only focused construction hooks for the concrete bound native search and external opponent. It does
not add a chess or Go match loop, result model, dataset loader, scheduler, or evaluation process. Those are shared.

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
    maximum_random_opening_plies: int
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
compatible with parallel search. When a maximum ply cap exists, maximum random opening plies must be below it. Values
remain fixed until the next model load.

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
2. if configured, an opening length is sampled uniformly from zero through the generation-scheduled maximum, then
   that many uniformly sampled legal actions are applied without search and retained in the action sequence; zero is
   valid and is the default, and a terminal random opening is
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
9. the coordinator commits progress and credits, workers load the trimmed artifact and reset trees, and completed
   old-generation statistics return with their acknowledgements; a later elapsed evaluation boundary references the
   same trimmed inference artifact without creating an evaluation-specific export;
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

Accepted Phase 2 implementation evidence:

- the old replay container, trainer-owned replay maintainer, replay broadcast, generic mailbox, commander lifecycle,
  publisher, separate game self-play loops, and superseded trainer runtime are deleted;
- relative to the accepted Phase 1 commit, the implementation changes 109 files, adds 4,233 lines, and deletes
  13,311 lines, for a net reduction of 9,078 lines;
- the configured 2.5-million-row replay files are exactly projected and allocated from one layout: chess uses
  456-byte rows and 1,140,000,118 bytes (1.062 GiB), Go 7x7 uses 362-byte rows and 905,000,118 bytes
  (0.843 GiB), and Go 9x9 uses 618-byte rows and 1,545,000,118 bytes (1.439 GiB);
- the opt-in real-runtime CPU smoke completes self-play, publication, ingestion, mapped loading, one-rank DDP,
  checkpoint activation, and generation transition for both games. On the local tiny validation models it measured
  chess at 565 ingested samples/s, 24 mapped rows/s, and 4.8 DDP samples/s, and Go 7x7 at 5,748 ingested
  samples/s, 2,238 mapped rows/s, and 7.8 DDP samples/s. These tiny-model figures validate instrumentation and
  end-to-end operation; they are not production performance claims;
- the ordinary Python suite passes with 186 tests and 10 intentional skips, the opt-in Phase 2 integration suite
  passes both games, and the compiled native suite passes all 11 suites.

### Phase 3: concurrent evaluation jobs

- replace generation-based evaluation settings with the fixed elapsed-time cadence and record the requested boundary;
- introduce the canonical definition, resolved job, opponent, dataset, opening-suite, per-game, aggregate, and
  result unions described above;
- add the coordinator-owned `EvaluationManager` and call its nonblocking collection and scheduling operations on
  every outer-loop iteration before training;
- retain one simple configured evaluation-device tuple, assign due definitions by cycling over it, and start all due
  definitions concurrently;
- run one short-lived process per dataset or opponent job and batch all active games within that process through the
  shared native C++ search or direct policy inference path;
- run distinct `search-random` and `policy-random` jobs: reuse one candidate native search for the former and use
  direct batched greedy policy inference for the latter; reuse one candidate native search for external matches and
  two separate searches for checkpoint matches, without adding another C++ search implementation;
- implement one shared paired-match loop, deterministic seeds, player-order swaps, cap policy, raw game records,
  paired-bootstrap aggregation, and TensorBoard/console reporting;
- implement immutable engine-guided opening-suite generation, including the four-ply bounded beam, transposition
  deduplication, 50 selected sequences, typed manifests, and inspectable chess/Go renderings;
- implement immutable engine-self-play fixed-dataset generation that retains every third position until reaching
  480 to 520 unique positions, with atomic reuse, packed rows, sparse policy labels, top-action accuracy, and policy
  cross-entropy;
- implement Stockfish through one clean UCI client and KataGo through one asynchronous JSON analysis client, with
  pinned and hashed executables, models, configurations, rules, and search limits;
- write one atomic typed artifact for every success, failure, timeout, or shutdown cancellation and implement the
  small manager-state-snapshot and result-scan restart behavior;
- rely on the current retain-all checkpoint policy for pending and running jobs, retain evaluation results, and remove retries,
  backoff, suite tiers, historical rotation, and plateau automation;
- delete the old chess `ModelEvaluation`, `EvaluationProcess`, paired-match adapters, chess-named shared protocol,
  HDF5 evaluation dataset/database/loader/statistics graph, preparation and benchmark tools, obsolete opening TSVs,
  plateau modules, obsolete settings, and all tests that target those superseded interfaces;
- migrate chess, Go 7x7, and Go 9x9 configurations and add focused shared, protocol-fixture, and opt-in real-engine
  integration tests.

Phase 3 is complete only when chess and both Go sizes use the same manager, process, match, dataset, statistics, and
reporting path; all configured jobs for one due suite launch concurrently across the configured device cycle; generated
artifacts are reproducible and inspectable; and the complete superseded evaluation graph is deleted rather than
retained beside the replacement.

Phase 3 was implemented as feature-sized ownership commits:

1. **Canonical contracts and immutable inputs:** shared configuration/contracts, Stockfish UCI and KataGo JSON
   clients, opening and dataset manifests/builders, preparation wiring, and protocol/builder tests.
2. **Shared evaluation execution:** raw fixed-dataset inference, the asynchronous paired-match runner, search and
   policy-only random opponents, checkpoint, Stockfish, and KataGo opponents, typed results, aggregation, and
   focused chess/Go composition hooks.
3. **Elapsed asynchronous lifecycle:** the manager state snapshot, simple device cycling, short-lived job processes,
   elapsed scheduling, collection/reporting, failure/shutdown behavior, retain-all checkpoint compatibility, and coordinator/run
   integration.
4. **Migration and deletion:** all checked-in chess/Go configurations migrated, the complete superseded graph listed
   above deleted, obsolete documentation and tests replaced, and the full native and Python validation matrix run.

Each commit leaves its introduced package internally tested and formatted. Temporary one-for-one adapters are not
part of this sequence; the final migration changes controlled callers directly.

The initial implementation therefore has five deliberate policy defaults: 20-minute elapsed boundaries evaluated
with the checkpoint available at each boundary and a matching 20-minute job deadline; one concurrent process per enabled definition assigned by cycling
evaluation devices; three preceding evaluated checkpoints plus retained generations 10 through 100 as historical
opponents; four-ply,
50-position generated opening suites; and one engine-self-play dataset retaining every third position until it has
480 to 520 positions per game/ruleset. Search limits, Stockfish WDL
temperature, sampling temperature, game counts, and the concrete evaluation-device tuple are benchmarked configuration
values, not new architecture decisions.

Accepted Phase 3 implementation evidence:

- chess, Go 7x7, and Go 9x9 load the same evaluation definitions and use the same manager, job process, match,
  dataset, statistics, and reporting path;
- every game schedules separate `search-random` and `policy-random` series; the latter directly batches the trimmed
  model policy, masks illegal actions, and uses greedy legal action selection without native search;
- all due definitions launch concurrently, device assignment is a stable cycle, historical jobs require an actually
  older boundary checkpoint, and TensorBoard uses the requested elapsed boundary as its step;
- chess evaluates Stockfish levels 0 through 3, all games evaluate the preceding 20-, 40-, and 60-minute checkpoints
  and each available retained generation from 10 through 100, and every job has a 20-minute deadline;
- generated openings contain exactly four plies and 50 distinct positions; generated datasets retain every third
  position until 480 to 520 positions exist and retain inspectable complete source-game records;
- the old chess evaluation package, HDF5 dataset/database/loader/statistics graph, plateau automation, standalone
  preparation/benchmark tools, and TSV opening suites are deleted. From authorization through the final code commit,
  Phase 3 changes 78 files, adds 4,011 lines, and deletes 4,683 lines for a net reduction of 672 lines;
- Ruff passes. The Windows suite passes 168 tests with 12 intentional native/real-engine skips. The freshly built
  extension-backed Linux suite passes 195 tests with 4 real-infrastructure skips. The native CTest target passes;
- opt-in Stockfish and KataGo smoke tests are present but were not run because local executable/model/configuration
  artifacts were not configured. Those real-engine and target-hardware contention measurements remain review-time
  deployment validation, not a second implementation path.

### Phase 4: integrated validation and cleanup

Status: the production-reachability and documentation cleanup slice is `accepted`. The remaining smoke,
performance, concurrency, and target-hardware evidence below is pending future authorization. R10 does not start or
complete this runtime-validation work.

- audit that no superseded modules, names, settings, tests, or normative documentation survived their owning phase;
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
- elapsed-boundary selection before training and delayed-boundary selection after a blocking quantum;
- correct checkpoint selection for every boundary crossed during one blocking quantum;
- concurrent evaluation launch, deterministic device cycling, and finished-result collection;
- paired chess/Go opening reconstruction, player swaps, deterministic seeds, cap handling, and aggregation;
- direct greedy-policy-versus-random play with legal-action masking and no native search construction;
- batched random and two-model match execution through the normalized native search bindings;
- deterministic engine-guided opening generation, beam bounds, transposition removal, manifests, and renderings;
- deterministic missing-or-reused fixed-dataset preparation, every-third-position sampling, the 480-to-520 range,
  packed rows, and policy metrics;
- Stockfish UCI and KataGo asynchronous JSON transcript fixtures, plus opt-in real-engine smoke tests;
- evaluation process failure, missing artifact, deadline, restart scan, retain-all checkpoint compatibility, and shutdown behavior;
- repository searches proving removal of the legacy chess evaluation, HDF5 dataset, and plateau graphs;
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
- Evaluation may launch after an exact 20-minute boundary while the coordinator is blocked in a training quantum;
  it still evaluates the checkpoint that was available at that boundary and reports at that boundary's elapsed-time
  step.
- Evaluation failures are durable results and are not retried automatically.
- Replay file size is less important than fixed-width simplicity and shared mmap access, but it remains measured experiment evidence.

These tradeoffs are intentional and must not be reversed by adding defensive compatibility layers, transactional replay recovery, duplicate configuration models, or alternate runtime modes without a new recorded design decision.
