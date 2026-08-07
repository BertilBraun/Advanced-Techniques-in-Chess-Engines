# Multi-game experimentation rework

## Purpose

Turn the chess training system into a configuration-driven research platform
for rapid AlphaZero experiments. Small-board Go provides the short feedback
loop; promising game-independent changes then progress to larger Go boards and
chess.

The expected experiment progression is:

1. screen an idea on 7x7 Go;
2. confirm a clear improvement on 9x9 Go;
3. test game-independent improvements on chess;
4. combine confirmed improvements for longer chess training.

A 7x7 Go run should target a useful learning curve in about six hours on one
or two GPUs. The initial benchmark run will determine the final model, search
budget, hardware allocation, and run duration. The user decides which results
justify another run, progression to 9x9, integration into chess, or adoption as
a new standard.

`THINGS_TO_TRY.md` is the experiment backlog. This document defines the
platform work needed to run and compare those experiments.

## Implementation workflow

This file is the authoritative plan and progress ledger. Work proceeds one
task at a time under these states:

- `pending`: ready for a future implementation pass;
- `in_progress`: the currently authorized task;
- `awaiting_user_review`: implemented and validated for manual review;
- `accepted`: reviewed and accepted by the user;
- `blocked`: needs a recorded decision before work can continue.

The tasks are intentionally phase-sized because their component changes are
interdependent. Only the user marks a task `accepted` and authorizes work or
continuation. An agent works only on the named task and makes feature-sized
commits as coherent parts are implemented and validated. The user may review
those commits and either request revisions or authorize continued work on the
same task.

Every implementation handoff reports:

- what was completed;
- what remains within the current task;
- the feature-sized commits made;
- validation commands and results;
- changes that deserve special review;
- unclear points, deviations, and decisions still required.

When all deliverables are complete, the agent sets the task to
`awaiting_user_review`. When a decision prevents safe progress, it sets the
task to `blocked`. It does not mark work `accepted` or begin another task.

Each task finishes with:

- implementation and focused cleanup of the paths it replaces;
- relevant formatting, static checks, compilation, and unit tests;
- a concise record of decisions, validation results, and unresolved issues;
- feature-sized commits that are individually coherent and validated.

Component-level work may temporarily leave the complete training pipeline
unavailable. Full integration and target-hardware validation occur after the
components have been assembled.

## Target system

### Experiment definition

The existing frozen Pydantic run-configuration system is the foundation of the
experiment configuration. It is adapted to load an experiment authored as one
YAML file and extended with a discriminated union of complete
`ChessExperimentConfiguration` and `GoExperimentConfiguration` variants.
Game-specific configuration stays inside its game variant; Go board size is a
field of `GoExperimentConfiguration`.

The configuration covers:

- game rules and representation;
- model and training objective;
- self-play and search parameters;
- replay capacity, sampling, and training cadence;
- evaluation opponents, budgets, and cadence;
- worker topology and resource allocation;
- run duration, output paths, seeds, and publication cadence.

Loading performs cross-field validation, including supported board sizes,
model shapes, action counts, worker-to-device assignments, evaluation
compatibility, and resource limits. A queue-validation command resolves every
selected YAML file before any experiment starts.

At run creation, the resolved Pydantic model, including defaults, is written as
canonical JSON in the run directory. The YAML remains the authored experiment;
the JSON records the exact effective configuration used by the run.

Search parameters are runtime values. Layout-dependent native types use the
supported compile-time game and board-size instantiations selected during
startup.

### Training lifecycle

Presentation-credit training is the single training lifecycle. The commander
starts persistent self-play and trainer processes, maintains replay while
credits accumulate, starts one optimizer quantum when enough credits exist,
publishes the resulting model generation, schedules evaluation, records
telemetry, and continues until a configured run limit is reached.

Credit parameters are required training configuration rather than an optional
mode. Credit accounting, recovery, replay maintenance, model publication,
evaluation scheduling, and shutdown each have focused owners. The commander
coordinates those components at one level of abstraction.

### Native search path

The production system has one MCTS implementation: the native C++ search used
by self-play, evaluation, and interactive analysis. Python owns supervision,
experiment policy, target construction, training, and result aggregation.
Python modules that supervise native work use canonical names such as
`SelfPlay`, `ModelEvaluation`, and `AlphaZeroBot`.

### Game specialization

Chess and Go are concrete first-class implementations. Shared orchestration,
search control, replay indexing, training lifecycle, evaluation scheduling,
and reporting operate on their typed game contracts.

The concrete implementations determine:

- position and action types;
- initial position, legal actions, and immutable child transitions;
- terminal detection, result, scoring, repetition, and ko semantics;
- action encoding and decoding;
- hashing and symmetries;
- network input representation and tensor encoding;
- policy, value, and auxiliary targets;
- model definition and loss;
- compact completed-game and replay representations.

The completed-game and replay boundaries are first established with chess.
The next implementation pass traces the resulting chess path and extracts its
game-specific decisions behind explicit C++ and Python contracts. It also
separates shared experiment settings from chess settings. The native Go game
then implements those contracts, and its integration provides the concrete
second use case for any further boundary adjustments.

Python remains responsible for experiment and process orchestration, game-level
self-play policy, replay maintenance, target construction, training, evaluation
scheduling, and reporting. Native code remains responsible for game-state
operations and complete batched tree searches with direct model inference.
Python supervisors consume completed native search results at the existing
coarse boundary.

### Go rules and position representation

The first Go implementation supports configurable 7x7 and 9x9 boards, komi,
pass moves, two-pass termination, a configured scoring rule, a maximum-move
safety bound, and simple ko represented by the ko point.

The native representation uses the shared fixed-size
`BitBoard<BoardSize>` value type in `cpp/src/util/BitBoard.hpp`. It stores its
words in `std::array<uint64_t, word_count>` and maintains zeroed padding bits
for board sizes that do not fill the final word.

```cpp
template <size_t BoardSize>
class BitBoard {
    static constexpr size_t bit_count = BoardSize * BoardSize;
    static constexpr size_t word_count = (bit_count + 63) / 64;
    using Storage = std::array<uint64_t, word_count>;
};
```

`BitBoard` is a small mutable value type for local algorithms. Its public
mutation operations preserve its representation invariants, and serialized
word access is read-only. Game positions and replay records provide immutable
ownership boundaries: move application constructs a new position, and a
published replay record is not subsequently modified.

Chess and Go share the bitboard storage and mechanical encoding utilities
without sharing plane semantics. A compact chess state may store binary planes
as `std::array<BitBoard<8>, BinaryPlaneCount>` beside its scalar-plane array,
while Go histories store arrays of absolute black and white bitboards.
Stockfish continues to use its own bitboard type inside chess rules. Conversion
occurs only at the chess encoding boundary.

Packed neural-network states use one explicit cross-language layout: binary
planes are plane-major arrays of 64-bit words serialized in little-endian byte
order, each plane's padding bits are zero, and the game-specific scalar payload
follows in its declared scalar representation. C++ game-specific
encoded-position types compose fixed-size bitboard arrays with their scalar
storage. They may share
mechanical serialization, hashing, equality, and tensor expansion without
sharing plane meanings.

Python uses a small immutable packed-plane value type backed by one contiguous
payload rather than raw, untyped bytes. Board size, binary-plane count, scalar
layout, and expected payload length belong to the game contract or type rather
than each instance. Active replay does not allocate one Python object per
plane or bitboard. Its measured per-sample overhead and projected memory at
the configured 2.5-million-sample capacity are review evidence before the
representation is accepted.

`GoPosition<BoardSize, HistoryLength>` contains:

- an array of historical boards, each with absolute black and white bitboards;
- player to move;
- ko point;
- consecutive pass count;
- move number;
- komi and rules metadata;
- an optional cached hash when profiling justifies it.

History index zero is the current board and increasing indices are older
boards. Creating a child copies the retained history one position back and
writes the new black and white boards at index zero. Positions are handled as
immutable values:

```cpp
GoPosition makeMove(const GoPosition& parent, Action action);
```

At the initial position, every unavailable historical black and white plane is
zero-initialized. Empty history therefore represents boards before any stones
were placed.

Absolute black and white history is converted to the current-player
perspective while encoding a neural-network batch.

Expanded tree nodes retain complete positions. A child edge initially stores
its action, prior, visit and value statistics, and child-node index. Its
position is created once when the child becomes a node. The native Go rules,
action mapping, history updates, terminal detection, scoring, and symmetries
receive deterministic unit tests against independently checked fixtures.

The `codex/go-experimental-rework` branch is a reference for tested rules and
integration ideas. Its position, history, and action interfaces are adapted to
the design above.

### Completed-game records

Each self-play worker publishes every completed game immediately as one file:

1. serialize to a temporary file in the run inbox;
2. flush and close it;
3. atomically rename it to its final filename.

The filename identity is derived from the run, worker, and worker-local
monotonic game number. One trainer-side replay maintainer owns ingestion. For
each published game it:

1. reads and validates the record;
2. appends one framed record to the archive for its model generation;
3. durably commits that append;
4. materializes eligible replay samples;
5. updates replay credits and telemetry;
6. removes the consumed inbox file.

A model-generation archive is a sequence of independently readable game records with
enough framing to detect an incomplete final append during recovery. This
provides prompt sample availability, a small steady-state inbox, easy per-
generation inspection, and replay reconstruction from durable archives.

A completed-game record contains:

- schema, game, rules, representation, run, worker, and model-generation
  metadata;
- initial state or equivalent reconstruction data;
- the action sequence and final result;
- per-move legal actions and sparse search visits;
- root value and the search statistics required by configured targets;
- move-selection mode, search budget, resignation/adjudication information,
  and sample eligibility or weighting data.

The schema retains the source information needed to derive planned targets
such as outcome, search policy, root value, remaining game length, and a later
move's search policy. Target builders transform completed games into the
compact replay representation selected by the experiment.

### Active replay and batch construction

The initial active replay is a simple bounded FIFO containing approximately
2.5 million compact game-specific samples in RAM. Python objects or similarly
direct containers are suitable for the first implementation. Each sample uses
packed bitboards, sparse `(action_id, visit_count)` policy data, configured
value and auxiliary targets, sample weight, and source model generation.

Model generation supports replay-freshness telemetry, including the average
generation lag sampled by each training batch.

Replay maintenance and optimizer work use separate phases:

1. the maintainer ingests completed games and updates the FIFO;
2. the trainer freezes a replay snapshot for an optimizer quantum;
3. every DDP rank trains from its local copy of that snapshot;
4. ingestion resumes and applies games accumulated during training.

Credits are awarded for eligible positions successfully materialized into the
active replay. The existing deterministic distributed sampler is adapted so
that ranks receive reproducible, non-overlapping samples from their snapshots.

The batch builder samples indices, applies game-specific augmentation, expands
packed states directly into preallocated final-shaped input and target
buffers, and transfers contiguous pinned-memory batches asynchronously. CPU
preparation of the next batch overlaps GPU training of the current batch.

The first end-to-end run records replay memory use, materialization throughput,
batch preparation time, transfer time, replay age, and credit balance. These
measurements determine any later representation or loader optimization.

### Evaluation

Evaluation runs concurrently with self-play and training on an elapsed
20-minute cadence. The evaluation worker waits for the first cadence boundary,
selects the newest completely published model checkpoint, evaluates it, and
then waits for the remainder of the time until the next cadence boundary. A
six-hour screening run therefore produces approximately 18 strength
measurements and can be inspected or stopped early.

The configured ladder includes:

- MCTS against random play;
- policy-only play against random play;
- previous and selected milestone models;
- fixed external engine levels appropriate to the selected game.

Chess uses its Stockfish opponents. Go adds an external engine adapter and a
fixed low-search opponent ladder suitable for both 7x7 and 9x9. Engine command,
rules, komi, board size, time/search limits, concurrency, game count, and
strength settings belong to the Go evaluation configuration.

Each checkpoint records opponent and model identity, colors, seeds or paired
openings, search budget, raw outcomes, wins/draws/losses, score, uncertainty,
duration, and failure status. Its model metadata records elapsed training time,
completed games, optimizer steps, and model generation. Evaluation results and
training telemetry share the experiment identity and elapsed-time axis for
direct comparison across runs.

The initial benchmark freezes one evaluation ladder and cadence for all
experiments in a comparison set.

### Experiment queue

The Linux experiment runner accepts an ordered set of validated YAML files and
a typed pool of resource slots. Each slot defines:

- explicit CUDA device indices;
- CPU affinity;
- RAM limit;
- run and log directories.

The runner validates the complete queue, assigns ready experiments to free
compatible slots, launches each experiment in its own process group, captures
logs and exit status, releases resources at exit, and immediately starts the
next compatible experiment. Success and failure both advance the queue.

A durable queue summary records pending, running, completed, and failed
experiments; start and finish times; assigned resources; process identifiers;
exit codes; and artifact locations. Termination handling signals the full
process group and records the resulting status.

## Execution ledger

| ID | Task | Status |
| --- | --- | --- |
| R1 | Remove Python MCTS and obsolete games | accepted |
| R2 | Credit-only training lifecycle and commander cleanup | accepted |
| R3 | Chess completed-game persistence and replay materialization | accepted |
| R4 | Chess RAM replay, batch construction, and DDP integration | accepted |
| R5 | Chess game-contract and configuration extraction | accepted |
| R6 | Shared bitboard and packed-plane representation | accepted |
| R7 | Native Go game implementation | accepted |
| R8 | Go pipeline integration | awaiting_user_review |
| R9 | Go evaluation and elapsed checkpoint scheduling | pending |
| R10 | Resource-aware experiment queue | pending |
| R11 | Integrated validation and benchmark preparation | pending |
| R12 | Target-hardware baseline and screening experiments | pending |

Current authorization: R8 is awaiting user review after removal of the obsolete desktop visualization layer.
R1 through R7 are accepted. No later phase is authorized.

### R1 — Remove Python MCTS and obsolete games

Deliverables:

- remove the Python MCTS and MCTS-node implementation, its graph and speed
  utilities, and production or test paths that execute Python search;
- remove `SelfPlayPy`, `AlphaZeroBotPy`, and the Python-search evaluation
  implementation;
- remove the runtime implementation switch and make native search the single
  production path;
- rename the native-backed Python supervisors to canonical modules and classes:
  `SelfPlay`, `ModelEvaluation`, and `AlphaZeroBot`;
- move evaluation results, paired-match decisions, visit normalization, and
  other retained non-search concepts into focused canonical modules before
  removing their former owners;
- remove Checkers, Connect4, Hex, and TicTacToe implementations, settings,
  visualizations, tests, and selection paths;
- retain the chess Python state and orchestration code required at the coarse
  native boundary;
- update imports, configuration, entry points, documentation, and tests to the
  canonical native-backed path.

Review evidence:

- repository search finds no Python MCTS implementation or implementation-mode
  switch;
- self-play, evaluation, and interactive entry points instantiate only the
  canonical native-backed supervisors;
- retained shared evaluation and policy utilities have no dependency on a
  deleted search implementation;
- focused chess self-play and evaluation tests pass through native search;
- the complete available Python suite contains no collection dependency on the
  removed games or Python MCTS.

### R2 — Credit-only training lifecycle and commander cleanup

Deliverables:

- make presentation-credit training the required training configuration and
  direct `CommanderProcess.run()` lifecycle;
- remove the iteration-based training loop, trainer, replay buffer, gating
  process, iteration messages, iteration telemetry, settings, entry points, and
  tests;
- move helpers still required by the persistent DDP trainer or dataset
  evaluation into focused modules, then make the persistent trainer the
  canonical `TrainerProcess`;
- remove credit-versus-iteration branches from the commander, self-play
  process, run configuration, and training configuration;
- retain model generation as checkpoint identity while credits and optimizer
  steps control training progress;
- decompose the commander lifecycle into focused initialization and recovery,
  replay maintenance, credit observation, optimizer-quantum, publication,
  evaluation, telemetry, and shutdown operations;
- keep the public commander loop readable at one level of abstraction;
- update tests around the single lifecycle, training failure cleanup, worker
  pause/resume, recovery, publication, evaluation, and shutdown.

Review evidence:

- repository search finds no legacy iteration trainer, gating process,
  iteration replay buffer, or training-mode branch;
- a valid run configuration always contains the credit schedule required to
  start training;
- deterministic commander tests cover waiting for credits, running one
  quantum, publication, recovery, evaluation scheduling, run limits, and clean
  shutdown;
- persistent DDP ranks remain alive across optimizer quanta;
- a short credit-training smoke path produces the next model generation.

### R3 — Chess completed-game persistence and replay materialization

Deliverables:

- define the versioned chess completed-game schema;
- publish one atomic completed-game file per worker game;
- implement the single-owner inbox consumer and framed per-model-generation archives;
- implement crash recovery for inbox files and incomplete final archive
  records;
- build the chess target materializer;
- connect materialized eligible positions to replay credits;
- add archive inspection and replay-rebuild commands;
- replace the chess sample-writing path.

Review evidence:

- concurrent publisher tests produce complete, uniquely named files;
- interruption tests recover every durably published game exactly once;
- archive rebuild produces the same ordered compact chess samples and credits
  as live ingestion;
- stored chess fixtures can derive every initially configured target.

### R4 — Chess RAM replay, batch construction, and DDP integration

Deliverables:

- implement the bounded compact FIFO and FIFO eviction for chess samples;
- define the replay freeze/ingest phase transition;
- distribute a frozen snapshot to persistent DDP ranks;
- adapt deterministic rank sampling;
- implement chess augmentation and direct preallocated batch encoding;
- overlap pinned-memory batch preparation and device transfer;
- emit capacity, eviction, credit, freshness, memory, and timing telemetry;
- replace the chess disk-backed training path.

Review evidence:

- FIFO capacity, order, eviction, and phase transitions pass deterministic
  tests;
- DDP tests show reproducible, non-overlapping rank samples;
- encoded chess batches match reference fixtures;
- a short chess optimizer smoke test completes.

### R5 — Chess game-contract and configuration extraction

Deliverables:

- trace chess-specific assumptions through C++ game state and actions, search,
  inference encoding, bindings, replay representation, and serialization;
- trace chess-specific assumptions through Python self-play control, sample and
  target construction, augmentation, batching, model and loss construction,
  evaluation, and reporting;
- define the C++ game contract used by chess, including position lifecycle,
  legal actions, transitions, terminal results, action mapping, input encoding,
  and representation dimensions;
- define the corresponding Python contract for game-level orchestration,
  compact samples, targets, batch construction, model and loss creation,
  augmentation, and game-specific evaluation configuration;
- convert the chess path to use those contracts while retaining its current
  behavior and external interfaces;
- inventory current settings and assign each setting to shared experiment
  infrastructure or the chess experiment variant;
- adapt the frozen Pydantic configuration into a complete
  `ChessExperimentConfiguration` with cohesive shared component
  configurations;
- add YAML loading, canonical resolved JSON output, and queue validation;
- route configuration explicitly to Python and native entry points;
- remove each converted static setting when its canonical typed owner is in
  use;
- add parsing, default-resolution, cross-field, and round-trip tests;
- provide a minimal valid chess experiment file.

Review evidence:

- the chess example validates and resolves deterministically;
- invalid chess and shared-setting combinations fail with precise errors;
- a contract inventory identifies every discovered chess-specific assumption,
  its canonical owner, and the converted call site;
- chess component tests pass through the extracted C++ and Python contracts;
- representative chess positions produce unchanged legal actions, encoded
  inputs, policy indices, targets, and model shapes.

The task ends for user inspection after the chess abstraction is complete.

### R6 — Shared bitboard and packed-plane representation

Deliverables:

- complete `BitBoard<BoardSize>` as the canonical fixed-size square-board bit
  container and add focused constexpr and runtime tests;
- keep mutation through invariant-preserving value operations and expose word
  storage read-only;
- canonicalize padding bits when constructing or deserializing boards whose
  final word is only partially used;
- define the canonical plane-major packed layout, including point mapping,
  word order, byte order, scalar placement, and validation of payload length;
- add a small immutable Python packed-plane value type backed by one
  contiguous payload, with representation dimensions owned by the game
  contract and no per-plane Python objects;
- use `std::array<BitBoard<8>, BinaryPlaneCount>` for compact chess binary
  planes beside the existing scalar-plane array;
- convert Stockfish bitboards only at the chess encoding boundary;
- share mechanical word serialization, hashing, equality, and direct batch
  expansion where C++ chess and future Go representations have identical
  needs;
- make C++ and Python produce and consume the same packed layout, verified by
  shared fixtures, without routing Python replay operations through native
  bindings or deduplicating game-specific plane construction;
- convert Python chess replay and batch construction from raw state bytes to
  the packed-plane value type while retaining one contiguous payload per
  sample;
- update the chess completed-game, replay, and batch fixtures to the bitboard
  representation without changing their game-specific schemas or plane
  meanings.

Review evidence:

- tests cover empty/full boards, point mapping, set algebra, iteration,
  multiword boards, and canonical padding for 7x7, 8x8, and 9x9;
- serialized bitboards round-trip with stable word ordering and zeroed padding;
- C++ and Python encode the shared fixtures to identical packed bytes and
  decode them to identical tensors;
- chess compact states and decoded batches match their pre-conversion fixtures;
- replay metrics record the packed value type's per-sample overhead and
  projected memory at 2.5 million samples, with no per-plane Python-object
  allocation;
- Stockfish rule and move-generation code continues to use its native bitboard
  representation.

### R7 — Native Go game implementation

Deliverables:

- use the shared bitboard and packed-plane utilities established in R6;
- implement the immutable Go position and history representation for the
  supported compile-time board sizes;
- implement initial position, legal action generation, immutable transitions,
  capture, suicide handling, simple ko, pass, terminal detection, scoring, and
  the maximum-move safety result;
- implement Go action encoding and decoding, network input encoding, hashing,
  and board symmetries required by the C++ contract;
- represent Go neural-network inputs with game-specific binary-plane and
  scalar semantics over the canonical packed layout established in R6;
- expose focused bindings for independently exercising Go positions, actions,
  transitions, terminal results, scoring, and encoding;
- adapt the extracted C++ contract where the concrete Go implementation
  demonstrates a missing semantic requirement;
- record each contract adjustment and its corresponding chess adaptation.

Review evidence:

- deterministic Go fixtures pass for captures, suicide, ko, pass termination,
  scoring, zero-initialized history, history shifts, actions, and symmetries;
- fixtures cover both 7x7 and 9x9 compile-time instantiations;
- bound Go operations agree with native results and independently checked
  fixtures;
- chess contract tests remain unchanged or receive documented adaptations for
  a contract change required by Go.

The task ends for user inspection with a tested native Go game, before it is
connected to self-play or training.

### R8 — Go pipeline integration

#### Revised shared-search architecture

User review rejected the parallel simplified Go search as the canonical R8
design. The optimized direct chess self-play algorithm is the performance
baseline and must become game-agnostic; the current `GameSearch` implementation
must not replace it without retaining its mature batching and overlap behavior.

The final native design has one game-parameterized search engine and tree with
two typed facades:

- `GameSelfPlaySearch<Game>` owns full/fast training schedules and submits
  typed root requests to `BatchedGameSearch<Game>`, maximizing throughput
  across independent games. Per-root concurrency remains configurable, so
  training can prefer batches from different games while analysis can reserve
  several leaves from one tree.
- `GameAnalysis<Game>` owns one retained root and exposes policy-only, counted,
  and deadline-limited analysis. It uses the same `BatchedGameSearch<Game>`
  executor without training noise and returns action IDs, candidate statistics,
  outcome/value data, depth, and principal variation records.

Both workloads use the same game contract, arena tree, selection, expansion,
backup, inference batch dispatcher, model refresh, and statistics. They are
separate entry points and request types rather than a mode plus nullable fields.
A model-vs-model evaluator owns one engine and inference batcher per model; it
batches positions assigned to the same model, never different models in one
inference call.

The authoritative implementation is built by generalizing the current
`DirectSelfPlaySearch` and chess `SearchTree`, preserving:

- multiple inference workers and multiple outstanding batches;
- overlap of tree selection and encoding with device inference;
- preallocated input and output slots, pending-batch completion, and avoidance
  of unnecessary worker waits;
- fixed/reusable tree storage, generation-safe handles, subtree reclamation and
  retention, reservations, virtual loss, and batched backup;
- full/fast search schedules, root noise, model refresh, and detailed timing and
  throughput statistics.

The optimized implementation is separated into focused template components
without virtual dispatch: `SearchTypes`, `SearchTree`, `SearchInference`, the
hot-loop `BatchedSearchExecutor`, the lifecycle-owning `BatchedGameSearch`
facade, `GameSelfPlaySearch`, and `GameAnalysis`. `InferencePipeline` owns
preallocated tensor slots, device execution, completion, and refresh. The
simplified `GameSearch` and all legacy MCTS trees are removed.

Raw inference owns only explicit dimensions, policy tensors, outcome tensors,
device execution, batching, and refresh. Dimensions are required at every
construction boundary; shared inference has no chess defaults or chess
includes. Generic result processing validates shapes and probabilities and
normalizes policy over actions supplied by the game contract. Concrete game
contracts own positions, legal actions, action IDs, child construction,
terminal values, packed encoding, and outcome perspective. Shared result types
must not contain chess `Board`, `Move`, or `MoveScore` values.

Analysis search is game-generic in R8 and is bound for chess, Go 7x7, and Go
9x9. `ChessAnalysisSession` is only the chess presentation layer: it owns
FEN/UCI replay, legal move application, retained-root navigation, and
translation of generic action IDs and principal variations to UCI. Go exposes
the same generic analysis facade with typed Go roots; a future Go protocol UI
can add its own coordinate presentation without changing search or inference.

Implementation sequence:

1. Record correctness and throughput baselines for optimized direct chess
   search, then remove `EvalMCTS`, `EvalMCTSNode`, their bindings and tests, and
   production-bound speed helpers that have no runtime caller.
2. Split bindings into common inference/search, concrete chess, concrete Go,
   and chess-analysis registration units; benchmarks remain standalone targets.
3. Define the complete native game contract and make inference require
   explicit dimensions. Split generic inference configuration/statistics from
   game-specific decoded results.
4. Generalize the mature chess arena `SearchTree` over the game contract and
   validate identical chess behavior plus 7x7 and 9x9 Go behavior.
5. Extract the inference-worker, pending-batch, and root-work machinery into
   `BatchedSearchExecutor` without changing its issue/complete scheduling.
6. Route chess and Go self-play and multi-game model evaluation through
   `GameSelfPlaySearch<Game>`. Remove the simplified `GameSearch` after parity
   and model-refresh tests pass.
7. Route chess and Go analysis through `GameAnalysis<Game>` while retaining the
   thin chess session and FEN/UCI presentation boundary.
8. Remove superseded `MCTS`, `DirectSelfPlaySearch`, `InteractiveSearch`,
   `EvalSearchTree`, legacy C++ `InferenceClient`, standalone
   `InferenceResultProcessing`, duplicate trees, and the Python cluster
   `InferenceClient` after their callers have migrated.
9. Validate deterministic chess/Go configuration, encoding, targets and loss;
   archive/replay recovery; CPU self-play and optimizer publication; model
   refresh and tree reset; interactive deadlines, cancellation and subtree
   retention; complete native/Python regressions; and optimized chess
   self-play throughput against the pre-refactor baseline.

Deliverables:

- extend the root experiment union with a complete
  `GoExperimentConfiguration`, including board size, rules, representation,
  model, training objective, self-play, replay, and evaluation settings;
- add minimal valid 7x7 and 9x9 Go experiment files and include them in queue
  validation;
- select supported native Go instantiations from the resolved experiment;
- connect Go positions, action dimensions, and input dimensions to batched
  search and direct inference;
- add the Python Go implementation for compact samples, target construction,
  augmentation, batch encoding, model, loss, and coarse self-play
  orchestration;
- use the R6 Python packed-plane value type for Go replay samples, adding only
  Go-specific plane construction and augmentation;
- extend the completed-game schema, target materializer, active replay, and
  batch construction path with their Go-specific representations;
- connect Go completed-game publication, replay credits, DDP sampling, and
  optimizer input to the infrastructure established in R3 and R4;
- adapt the C++ and Python contracts where end-to-end Go integration reveals a
  concrete missing requirement;
- remove converted chess assumptions from shared orchestration and search
  entry points.

Review evidence:

- 7x7 and 9x9 configurations validate and resolve deterministically;
- invalid Go-specific combinations fail with precise errors;
- encoded Go inputs, policy targets, value targets, symmetries, model outputs,
  and losses match deterministic fixtures;
- Go archive rebuild and replay tests match live ingestion;
- short CPU self-play and optimizer smoke tests complete for Go;
- a Go model publication resets active trees after loading the new model;
- chess component and smoke tests continue to pass through the shared path.

The task ends for user inspection with Go connected through configuration,
self-play, inference, and training.

### R9 — Go evaluation and elapsed checkpoint scheduling

Deliverables:

- add the elapsed 20-minute evaluation loop to the experiment lifecycle, using
  the newest completely published model at each cadence boundary;
- integrate Go match execution with the shared search and inference path;
- implement and verify the external Go engine protocol adapter;
- define fixed 7x7 and 9x9 opponent ladders;
- produce common result records, summaries, uncertainty, and TensorBoard data;
- attach elapsed time, completed games, optimizer steps, and model generation
  to each evaluated checkpoint;
- retain chess-specific Stockfish configuration in the chess variant.

Review evidence:

- deterministic match fixtures verify colors, results, seeds, and summaries;
- protocol tests cover Go board setup, rules, komi, moves, pass, and result
  handling;
- a short scheduled smoke run publishes complete checkpoint artifacts.

### R10 — Resource-aware experiment queue

Deliverables:

- define queue and resource-slot schemas;
- validate all experiment files and assignments before launch;
- implement CUDA visibility, CPU affinity, RAM limits, process groups, logging,
  and exit capture;
- schedule the next compatible job as soon as a slot is released;
- persist and display the queue summary;
- add scheduler tests using short test-local processes.

Review evidence:

- tests verify allocation exclusivity and maximum permitted parallelism;
- success, failure, and termination release their slots and update status;
- queue order is deterministic for the same configurations and resources.

### R11 — Integrated validation and benchmark preparation

Deliverables:

- assemble the complete chess and Go pipelines;
- run the complete relevant native and Python test suites;
- run short local end-to-end smoke experiments for chess, 7x7 Go, and 9x9 Go;
- verify artifacts, replay recovery, DDP synchronization, model publication,
  evaluation scheduling, and clean shutdown;
- prepare frozen target-hardware benchmark configurations and commands;
- document remaining hardware gates in this ledger.

Review evidence:

- all component and integration checks are recorded with exact commands;
- each smoke experiment produces a coherent run directory and resolved
  configuration;
- benchmark inputs are fixed before renting the compute node.

### R12 — Target-hardware baseline and screening experiments

Deliverables:

- measure the 7x7 baseline on the intended one- and two-GPU slots;
- record self-play throughput, inference occupancy, training throughput,
  replay behavior, evaluation overhead, memory, and strength over time;
- select the standard six-hour screening configuration;
- verify that the selected configuration produces a useful learning curve;
- run an initial queued comparison from `THINGS_TO_TRY.md`;
- promote results through 9x9 Go and chess according to the progression in
  this plan.

Review evidence:

- run artifacts are complete and directly comparable;
- the baseline configuration and evaluation ladder are frozen for subsequent
  experiments;
- the ledger records measured limitations and the next approved experiment.

## Decision and issue log

Record only decisions that change this plan or issues that block the current
task.

| Date | Task | Type | Record | Resolution |
| --- | --- | --- | --- | --- |
| 2026-08-07 | R8 | Desktop visualization cleanup | The retained Pygame grid, chess and Go visual strategies, mutable native-Go board adapter, font asset, and standalone Go inspector had no production training, evaluation, UCI, or deployment consumer after removal of the legacy human-play graph. | Remove the complete desktop presentation graph and its visualization-only tests, delete `CurrentGameVisuals`, and regenerate the locked training environment without Pygame. Preserve typed game/search/analysis boundaries for future web presentation. The exact Windows suite passes 313 tests with 5 native-dependent skips, the fresh-extension suite passes all 351 tests, and Ruff passes. Return R8 to `awaiting_user_review`. |
| 2026-08-07 | R8 | Python configuration and legacy cleanup | Chess and Go experiment variants duplicated ownership and validation of the shared run and training configuration, while manually removed legacy bot, tournament, dataset-training, inspection, and optimization entrypoints left stale imports, dependencies, and documentation. | Introduce `BaseExperimentConfiguration` as the canonical owner of `ExperimentRunConfiguration` and `TrainingArgs`, leaving concrete variants to validate only game-specific fields. Retain the production chess evaluation/database path and its `SelfPlayDataset` consumer, colocate the sole UCI legal-move selection helper with the server, remove the orphan legacy graph and unused Optuna dependency, and update entrypoint documentation. The exact Windows suite passes 314 tests with 6 native-dependent skips, the fresh-extension suite passes all 356 tests, the focused UCI suite passes 17 tests, and Ruff passes. Return R8 to `awaiting_user_review`. |
| 2026-08-07 | R8 | Inference ownership review | `InferenceModel.hpp` contained implementation-only TorchScript loading and refresh validation despite having no independent owner, while `InferencePipeline.hpp` opened with the full game-result processing implementation before declaring the pipeline. | Delete the model header and keep its private implementation beside `InferenceRunner` in `InferencePipeline.cpp`; retain only the prepared-model ownership type in the public pipeline interface. Forward-declare `processInferencePosition` near its result type and place its required template implementation after the pipeline declaration. Native extension, test, and benchmark targets compile; native tests and Ruff pass. Return R8 to `awaiting_user_review`. |
| 2026-08-07 | R8 | Review revision | Chess and Go duplicated the mechanical expansion and packing of binary and scalar planes; `SearchGame` verified only the concrete composition types instead of constraining their generic consumers; native Go symmetry code was reachable only through tests and Python bindings while Python already owns training augmentation. | Introduce one `EncodedPlanes` representation for shared packed and tensor mechanics, constrain every game-generic search and inference consumer with `SearchGame`, and delete the native Go symmetry implementation, binding surface, stubs, and tests. Retain Python's Go symmetry implementation as the sole training-augmentation owner. Return R8 to `awaiting_user_review`; native targets and tests, focused Go tests, and Ruff pass. The complete Python suite passes 350 tests with the same unrelated scalar-to-WDL expectation failure retained from checkpoint `62f8517`. |
| 2026-08-07 | R8 | Native game-structure handoff | Chess and Go now expose one compile-time-verified `SearchGame` composition surface, but retain their natural internal state implementations. Their implementation, encoding, binding, and chess-presentation files have uniform ownership; superseded `GameContract`, `GoTypes`, policy-mapping wrappers, and identity-only Go bindings are removed. Maximum-move Go endings now produce scored, replay-eligible adjudications under the configured area scoring rule. | Return R8 to `awaiting_user_review`. The extension compile check and unified native suite pass, focused native-backed Go tests pass, Ruff passes, and the complete Python suite passes 351 tests with one unrelated failure retained from pre-restructure checkpoint `62f8517`: its scalar-to-WDL formula changed without updating the existing expected-distribution test. |
| 2026-08-07 | R8 | Review authorization | The native chess and Go implementations expose the same search semantics through unverified static contracts, but their rules, state, action, policy mapping, encoding, and binding ownership remain inconsistently structured. Maximum-move Go games are expected to be common during early training, so discarding their positions would also discard a material part of the initial learning signal. | Continue R8 with a C++-only structural cleanup: define compile-time game concepts; give chess and Go the same implementation, encoding, and binding topology; expose each game to search through one canonical composition type; remove superseded contracts and adapters rather than retaining compatibility layers; leave Python chess on python-chess and retain only required native Go and coarse search/root bindings. Score a maximum-move Go position with the configured area scoring and record the ending as a safety adjudication with a winner or draw. |
| 2026-08-07 | R8 | Augmentation review | Replay must retain one packed sample while training applies one randomly selected game symmetry after DDP rank sampling; the native search contract does not consume augmentation. | Keep chess identity/file-mirror selection in `build_chess_training_batch` and all eight Go dihedral transforms in `build_go_training_batch`, remapping state and sparse policy together before decoding. Add batch-boundary fixtures for both games; Python remains the sole Go augmentation owner after the native symmetry surface was removed. |
| 2026-08-07 | R8 | Master integration | The Python consolidation and augmentation work was initially committed on a detached worktree while `master` independently advanced the native contracts, Go cap adjudication, shared encoding, and symmetry removal. | Integrate the five Python commits onto `master` in dependency order, retain completed-game schema v2 and scored replay-eligible Go cap adjudications during conflict resolution, and keep native Go symmetry deleted. The resulting master passes all 353 extension-backed Python tests, the 311-test Windows suite with 6 native-dependent skips, the unified native test target, and Ruff. |
| 2026-08-07 | R8 | Native-backed Python validation | The freshly built extension initializes the shared LibTorch inter-op pool before a spawned trainer rank configures PyTorch, and the Go objective derived float32 target labels from bfloat16 model logits. | Avoid resetting inter-op threads when the requested value is already active, and keep Go outcome/MCTS target interpolation in the replay target dtype. The complete extension-backed Python suite now passes 353 tests, including Go checkpoint publication. |
| 2026-08-07 | R8 | Python pipeline consolidation | Python still maintained parallel completed-game publishers, archive/recovery formats, replay containers, batch loaders, Go optimizer/DDP lifecycle, synchronous Go orchestration, state contracts, and active-game loops. | Use one canonical identity/publisher and indexed archive frame, one packed replay/snapshot/metrics/maintainer/batch-loader implementation, one persistent TrainerProcess for world-size-one and DDP training, concrete chess/Go training implementations behind one typed contract, one Commander lifecycle with Go evaluation intentionally disabled until R9, one active-game pool, and a shared state-contract base. Remove GoTrainingLifecycle, GoDistributedTraining, and the obsolete test-only DistributedTraining module; extract live self-play statistics and training-batch types while retaining SelfPlayDataset only for its evaluation/database consumers. |
| 2026-08-07 | R8 | Native dead-code cleanup | Shared search still exposed unused convenience searches, test-only tree diagnostics, duplicate model-generation state, broad game-contract helpers, a stateless chess action codec, and a separate Go symmetry-type header; executor selection/completion was deeply nested and inference consumption required an accessor callback. | Remove the dead APIs and duplicate state, keep only operations consumed by generic search in each game contract, put chess encoding/decoding/UCI on `ChessAction`, construct `GoAction` from points, colocate `GoSymmetry` with its operations, add `InferenceDimensions::encodedSize`, pass typed position ranges through inference, name the live leaf reservation `inference_pending`, extract executor stages, and make `InferenceStatistics` the executor's cumulative telemetry representation. Retain the bounded C++20 atomic wait/notify slot state machine because it parks rather than spins and safely reuses preallocated inference buffers. |
| 2026-08-07 | R8 | Readability cleanup | Indexed native loops inconsistently used verbose loop syntax, raw inference tensors escaped into the search executor for legal-policy processing, component ownership was undocumented, and repeated clock arithmetic obscured the optimized search flow. | Use the zero-overhead `range` utility for numeric iteration, make `InferencePipeline` return validated game-legal inference results and own wait/processing telemetry, delete `SearchInference`, document each search/inference component at its header, and centralize monotonic timing in `Timing.hpp`. |
| 2026-08-07 | R8 | Follow-up cleanup | Native dimension arithmetic still used signed integers, Go split its small value types across separate headers, Go position members used suffix underscores, positional aggregate initialization obscured field meaning, and unused concurrency utilities remained. | Use `std::size_t` for inference dimensions with explicit conversion only at the LibTorch tensor boundary; consolidate `GoBoard` and `GoAction` in `GoTypes`, add typed non-pass `GoAction::point()`, use `m_` position members and C++20 designated initializers for semantic aggregates, and delete the unreferenced native `ThreadPool` and self-tested-only `BlockingQueue`. |
| 2026-08-07 | R8 | Review handoff | The complete Go training pipeline and the follow-up native ownership cleanup are implemented through the shared optimized search path. | Return R8 to `awaiting_user_review` after the full native graph compiled, all 11 suites in the unified native test target passed, all 352 Python tests passed, Ruff passed, model refresh reset retained chess and Go trees, and no R9 evaluation functionality was added. |
| 2026-08-07 | R8 | Review rejection | The shared optimized search is functionally complete, but `BatchedSearch.hpp` combines the arena, root ownership, request/result types, scheduling, inference dispatch, execution, refresh, and statistics in one thousand-line template header; inference infrastructure also remains scattered at the native source root, and analysis is exposed as a chess-owned engine rather than a shared workload with chess presentation. | Return R8 to `in_progress`. Split cohesive template components into focused headers, move search/inference infrastructure under `search`, introduce a shared analysis facade usable by each game contract, and reduce chess units to chess rules, self-play policy, and FEN/UCI presentation. |
| 2026-08-07 | R8 | Shared-search completion | Chess self-play, Go 7x7/9x9 self-play, and retained-tree chess analysis previously reached the optimized arena through overlapping but separately owned scheduling layers. | Make `BatchedGameSearch<Game>` and `GameSearchTree<Game>` authoritative for every native workload, preserve multi-worker/outstanding-batch overlap, let chess analysis select one retained root with parallel leaf searches, and remove `MCTS`, `EvalMCTS`, `EvalSearchTree`, `DirectSelfPlaySearch`, and the duplicate analysis scheduler. |
| 2026-08-07 | R8 | Boundary cleanup | Generic inference and module registration still carried names and files inherited from the chess-only implementation. | Replace the `DirectInference` layer with dimension-explicit `InferenceModel` and `InferencePipeline` components under `search`; isolate common, chess, Go, self-play, and analysis bindings; remove unused clients/result-processing layers; and keep only FEN/UCI/candidate/PV translation in `ChessAnalysisSession`. |
| 2026-08-07 | R8 | Native ownership cleanup | The first shared implementation left search lifecycle, optimized scheduling, inference, bindings, and chess aliases grouped into broad or misleading units. | Split the public `BatchedGameSearch` lifecycle from `BatchedSearchExecutor`, make `GameSelfPlaySearch<Game>` and `GameAnalysis<Game>` the only workload facades, rename the canonical runtime input to `InferenceConfiguration`, remove `direct` metric names, move Stockfish initialization into chess registration, and keep game-specific binding files as presentation only. |
| 2026-08-07 | R8 | Header ownership cleanup | The project precompiled header injected Stockfish namespaces, chess action/board constants, integer aliases, move formatting, and generic helpers into every native translation unit, concealing dependencies and leaking chess into Go/search compilation. | Keep `common.hpp` as a neutral standard-library/LibTorch preamble only; make `ChessAction` own its action count and UCI value presentation, make `ChessRepresentationDimensions` and `CompressedEncodedBoard` own encoding dimensions and packed sizes, use standard fixed-width integer types, and prohibit Stockfish namespace imports from public headers. |
| 2026-08-07 | R8 | Review rejection | The first R8 implementation added a simplified game-generic Go search beside the optimized chess search; production chess continued through chess-specific `MCTS`, `DirectSelfPlaySearch`, and `SearchTree`, so the result was not one shared pipeline. | Return R8 to `in_progress`. Generalize the optimized chess direct-search implementation and mature arena over game contracts, preserve separate typed batched-game and single-tree analysis workloads, and remove the simplified and legacy parallel implementations after migration. |
| 2026-08-05 | R8 | Design decision | A Go game terminated by the maximum-move safety bound has no result target and produces no eligible replay samples or credits; its completed-game record remains archived for telemetry and recovery. | Use R7's explicit terminal-without-value contract rather than inventing a training label. |
| 2026-08-06 | R8 | Contract adjustment | Chess's compile-time inference dimensions prevented resolved 7x7 and 9x9 Go experiments from sharing the direct-inference artifact boundary. | Make inference artifacts carry explicit channel, row, column, action, and outcome dimensions; validate resolved Go configuration against the selected native template before starting self-play. |
| 2026-08-06 | R8 | Design change | Go exposed shared search-tree and replay-sampling mechanics alongside genuinely different state, action, target, augmentation, model, and loss semantics. | Instantiate one typed native game-search template for chess, Go 7x7, and Go 9x9; share packed storage, deterministic rank sampling, credit, publication, and recovery infrastructure while retaining concrete per-game Python contracts. |
| 2026-08-06 | R8 | Scope boundary | Go evaluation requires the external engine adapter, opponent ladders, and elapsed scheduling assigned to R9. | Keep the typed Go evaluation configuration in R8, but do not execute Go evaluation until R9 is authorized. Existing chess evaluation remains unchanged. |
| 2026-08-06 | R3/R4 | Design change | The disk replay and its reanalysis sidecars were replaced together by completed-game archives and RAM snapshots; the reanalysis settings were removed rather than left inert in the new ownership model. | Review with the combined R3/R4 implementation; any future reanalysis design must use completed-game or snapshot ownership explicitly. |
| 2026-08-06 | R3/R4 | Design change | Archive frame headers retain identity, credit totals, and eligible-sample counts so restart scans metadata but materializes only the newest capacity-sized tail. Memory mapping is deferred because the current object replay would require a second packed columnar representation and 10 GB across four ranks is acceptable. | Reconsider shared read-only memory only if target-hardware measurements show RAM or DDP snapshot publication is limiting. |
| 2026-08-06 | R6 | Design change | R6 establishes the packed-plane representation needed by Go before implementing Go: tested fixed-size C++ bitboards, one explicit cross-language layout, a contiguous Python packed value type, and a behavior-preserving chess integration. | Share storage mechanics and fixtures, not chess/Go plane semantics; prohibit per-plane Python objects and review projected replay memory before acceptance. |
| 2026-08-06 | R7 | Contract adjustment | Go can terminate at the maximum-move safety bound without a game value, while the chess contract previously exposed only a required terminal result. | Add an optional `terminalValue` contract operation; chess returns its existing result for terminal positions and no value for ongoing positions, while maximum-move Go returns no value. |
| 2026-08-06 | Post-R7 | Design cleanup | Chess board state, action encoding, input encoding, and replay history remained at the native source root after the game-specific directories were introduced. | Consolidate their authoritative implementations as `ChessBoard`, `ChessAction`, `ChessEncoding`, and `ChessHistory` units under `cpp/src/games/chess`; keep R8 pending. |
