# Python architecture rework

## Status and purpose

This document defines the target Python architecture for the experiment runtime. It is an implementation plan, not a compatibility specification for the current Python structure.

The rework should replace the current mixture of file mailboxes, commander-owned component details, trainer-owned replay maintenance, copied Python replay snapshots, and exact recovery machinery with a small synchronous coordinator and explicit process boundaries.

The design priorities, in order, are:

1. simple ownership and readable control flow;
2. one canonical typed configuration for every concept;
3. high-throughput self-play, replay ingestion, and DDP training;
4. straightforward restart from mmap, ledger, and checkpoint files;
5. deterministic behavior where it affects experiments;
6. approximate rather than transactionally exact recovery of replay samples and credits.

Small replay or credit discrepancies after an unclean exit are acceptable. The runtime must not add complicated transactional recovery to prevent the loss or duplication of a negligible number of samples.

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
@dataclass(frozen=True)
class TrainingProgress:
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

        if self.ledger.can_train_quantum:
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
    self._pause_required_self_play_workers()

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

## ReplayManager

`ReplayManager` is an ordinary synchronous object in the coordinator process.

```python
@dataclass(frozen=True)
class ReplayIngestion:
    games_ingested: int
    samples_added: int
    live_samples: int
    evicted_samples: int


@dataclass(frozen=True)
class ReplayDescription:
    path: Path
    head: int
    size: int
    logical_capacity: int
    maximum_capacity: int
    layout: ReplayLayout


class ReplayManager:
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
6. removes the consumed inbox file;
7. returns aggregate ingestion counts.

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
value and auxiliary targets
eligibility fields
sample weight
source model generation
source timestamp
```

All games use:

```text
policy_entry_count                        : uint8
action_ids[maximum_policy_entries]        : uint16
visit_counts[maximum_policy_entries]      : uint16
```

Configuration validation requires:

- `maximum_policy_entries` between 1 and 255;
- action count representable by `uint16`;
- configured search visits representable by `uint16`;
- a replay layout compatible with the selected game representation.

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
- experiments that intentionally test tighter policy retention.

Policy truncation telemetry records:

- samples whose policy was truncated;
- retained and discarded visit counts;
- discarded visit-mass fraction;
- largest discarded individual visit count.

Truncation is intentional and never silently removes entries without telemetry.

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

    @property
    def can_train_quantum(self) -> bool:
        ...

    def add_samples(self, sample_count: int, model_generation: int) -> None:
        ...

    def commit_quantum(self, result: TrainingQuantumResult) -> None:
        ...

    def save(self) -> None:
        ...
```

New samples earn credits according to the credit schedule at their ingestion generation. Previously earned credits are not retroactively revalued when the schedule changes. A quantum consumes the amount configured for its source generation. Surplus credits carry forward.

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
        configuration: TrainerConfiguration,
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

- model state;
- optimizer state;
- JIT inference model;
- checkpoint manifest, written last.

Rank zero is the sole checkpoint writer because the live model and optimizer exist inside rank processes. `TrainerGroup` owns the aggregate orchestration and validates the single-writer result.

### Training result

Results contain file references, not loaded models:

```python
@dataclass(frozen=True)
class CheckpointReference:
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

The coordinator sends desired state, not transition commands:

```python
class SelfPlayMode(StrEnum):
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPED = 'stopped'


@dataclass(frozen=True)
class SelfPlayDesiredState:
    mode: SelfPlayMode
    checkpoint: CheckpointReference
    collect_completed_generation_statistics: bool
```

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
10. resumes only if desired mode is `running`.

```python
@dataclass(frozen=True)
class SelfPlayStateApplied:
    worker_id: int
    mode: SelfPlayMode
    loaded_generation: int
    loaded_inference_model_sha256: str
    completed_generation_statistics: SelfPlayStatistics | None
```

All workers may return cheap counters. Only one or two configured workers collect expensive distributions and native diagnostics. Reports identify worker ID, generation, and statistics level so sampled detailed results are not interpreted as global totals.

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

training/
    group.py
    rank.py
    protocol.py
    trainer.py
    batch.py
    statistics.py

evaluation/
    manager.py
    process.py
    result.py

games/
    training_contract.py
    chess/
    go/
```

Concrete game implementations own:

- completed-game schema and validation;
- transformation from a completed game into fixed replay rows;
- state and target decoding;
- data augmentation;
- batch construction;
- training objective schedules;
- self-play schedule semantics;
- game-specific evaluation jobs.

Shared infrastructure owns:

- schedule mechanics;
- mmap FIFO mechanics;
- configurable top-policy retention mechanics;
- credit accounting;
- deterministic rank sampling;
- DDP process orchestration;
- checkpoint references;
- self-play desired-state transport;
- evaluation process management.

## Implementation phases

Each phase should remove the path it replaces rather than leave compatibility layers.

### Phase 1: canonical schedules and progress

- add generic constant, staged, and linear generation schedules;
- add rounding validation and deterministic interpolation tests;
- convert scheduled experiment fields to generation schedules;
- make optimizer steps the only persisted progress field;
- derive model generation everywhere;
- validate static optimizer steps per quantum and aligned run limits;
- remove optimizer-step and separately persisted model-generation schedule axes.

### Phase 2: fixed-width mmap replay

- define typed replay layout metadata;
- add experiment-configured `maximum_policy_entries`;
- implement deterministic top-policy retention and telemetry;
- implement the single-file fixed-slot circular mmap;
- convert chess, Go 7x7, and Go 9x9 materialization;
- build training batches directly from mapped rows;
- measure projected and actual replay file size and batch-read throughput.

### Phase 3: coordinator-owned replay and ledger

- move replay ingestion into `ReplayManager` in the coordinator process;
- make each call drain the full inbox;
- apply capacity schedules inside ingestion;
- simplify the ledger to atomic progress, credit, and checkpoint state;
- remove trainer-rank replay ownership;
- remove exact archive rebuild and prepared-publication recovery machinery that no longer has an owner.

### Phase 4: TrainerGroup and mmap DDP

- introduce the coordinator-owned `TrainerGroup` facade;
- make all ranks persistent symmetric child processes;
- always initialize DDP, including world size one;
- send small replay descriptions rather than replay objects;
- map the replay read-only in every rank;
- resolve learning rate and objective once per quantum;
- retain model and optimizer across quanta;
- make rank zero the sole checkpoint writer;
- aggregate statistics and failures in `TrainerGroup`;
- remove `broadcast_object_list` replay distribution and the single-rank training branch.

### Phase 5: self-play desired-state connections

- replace file mailboxes with duplex multiprocessing connections;
- implement desired running, paused, and stopped states;
- combine generation statistics, model loading, tree reset, and acknowledgement into one transition;
- configure basic versus detailed statistics workers;
- delete the generic communication module and its tests;
- retain atomic completed-game files as the durable data boundary.

### Phase 6: simple coordinator lifecycle

- rewrite the public loop at one level of abstraction;
- implement focused ingestion, training, transition, evaluation, and shutdown helpers;
- keep training synchronous and blocking at the coordinator boundary;
- remove the publisher abstraction and separate lifecycle-state machinery;
- enforce run limits between ingestion/training iterations;
- verify restart from mmap, ledger, and latest complete checkpoint.

### Phase 7: concurrent evaluation jobs

- introduce focused evaluation job and result types;
- launch multiple short-lived evaluation processes after due checkpoints;
- collect finished jobs during coordinator iterations;
- write atomic result artifacts;
- log completed results to TensorBoard and the console;
- complete concurrency, device assignment, timeout, opponent, and retention behavior.

### Phase 8: integration and cleanup

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
- FIFO insertion, capacity changes, wraparound, and eviction;
- deterministic policy top-N selection and discarded-mass telemetry;
- concurrent read-only mmap access from multiple spawned ranks;
- deterministic disjoint DDP sampling;
- DDP world size one and multi-rank training;
- persistent model and optimizer state across quanta;
- rank failure propagation and rank-zero checkpoint writing;
- self-play desired-state pause, model transition, statistics, and resume;
- complete inbox draining and queued-game ingestion after training;
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
- Evaluation details may mature after the core runtime boundaries are implemented.
- Replay file size is less important than fixed-width simplicity and shared mmap access, but it remains measured experiment evidence.

These tradeoffs are intentional and must not be reversed by adding defensive compatibility layers, transactional replay recovery, duplicate configuration models, or alternate runtime modes without a new recorded design decision.
