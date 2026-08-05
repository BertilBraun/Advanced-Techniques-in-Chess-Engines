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

The shared interfaces should be extracted at the point where both concrete
games need them. This keeps the contracts driven by working chess and Go use
cases.

Python remains responsible for experiment and process orchestration, game-level
self-play policy, replay maintenance, target construction, training, evaluation
scheduling, and reporting. Native code remains responsible for game-state
operations and complete batched tree searches with direct model inference.
Python advances completed searches at the existing coarse boundary.

### Go rules and position representation

The first Go implementation supports configurable 7x7 and 9x9 boards, komi,
pass moves, two-pass termination, a configured scoring rule, a maximum-move
safety bound, and simple ko represented by the ko point.

The native representation uses a fixed-size bitboard:

```cpp
template <size_t NumSquares>
struct BitBoard {
    static constexpr size_t NUM_WORDS = (NumSquares + 63) / 64;
    std::array<uint64_t, NUM_WORDS> words;
};
```

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
position is created once when the child becomes a node. Every active self-play
tree is reset when a newly published model is loaded. The native Go rules,
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
2. appends one framed record to the archive for its model iteration;
3. durably commits that append;
4. materializes eligible replay samples;
5. updates replay credits and telemetry;
6. removes the consumed inbox file.

An iteration archive is a sequence of independently readable game records with
enough framing to detect an incomplete final append during recovery. This
provides prompt sample availability, a small steady-state inbox, easy per-
iteration inspection, and replay reconstruction from durable archives.

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
| R1 | Typed experiment configuration | pending |
| R2 | Concrete chess/Go game boundary and Go integration | pending |
| R3 | Completed-game persistence and replay materialization | pending |
| R4 | RAM replay, batch construction, and DDP integration | pending |
| R5 | Go evaluation and elapsed checkpoint scheduling | pending |
| R6 | Resource-aware experiment queue | pending |
| R7 | Integrated validation and benchmark preparation | pending |
| R8 | Target-hardware baseline and screening experiments | pending |

Current authorization: plan review only.

### R1 — Typed experiment configuration

Deliverables:

- inventory the existing run-configuration models and every consumer affected
  by game selection;
- adapt the current frozen Pydantic configuration into complete chess and Go
  variants while preserving the existing typed configuration structure;
- add YAML loading, canonical resolved JSON output, and queue validation;
- route configuration explicitly to Python and native entry points;
- progressively remove static settings as each value becomes owned by the
  adapted run configuration;
- add parsing, default-resolution, union-discrimination, cross-field, and
  round-trip tests;
- provide minimal valid chess, 7x7 Go, and 9x9 Go experiment files.

Review evidence:

- the three example files validate and resolve deterministically;
- invalid game-specific combinations fail with precise errors;
- affected entry points consume typed configuration values.

### R2 — Concrete chess/Go game boundary and Go integration

Deliverables:

- identify the minimal shared contracts from the concrete chess and Go needs;
- implement the fixed-size Go bitboard and immutable position/history model;
- implement and test Go rules, scoring, action mapping, encoding, and
  symmetries for 7x7 and 9x9;
- select supported native game instantiations from the experiment
  configuration;
- connect Go positions to batched search and inference;
- add the Go model, loss, target definitions, and coarse Python self-play
  orchestration;
- reset every active self-play tree when a new model publication is loaded;
- remove game assumptions replaced by the new typed boundary.

Review evidence:

- deterministic Go fixtures pass for captures, suicide, ko, pass termination,
  scoring, zero-initialized history, history shifts, actions, and symmetries;
- chess component tests continue to pass through the new boundary;
- model-refresh tests verify that no search statistics survive publication of a
  new model;
- short CPU smoke searches complete for both games with valid policy and value
  shapes.

### R3 — Completed-game persistence and replay materialization

Deliverables:

- define versioned chess and Go completed-game schemas;
- publish one atomic completed-game file per worker game;
- implement the single-owner inbox consumer and framed per-iteration archives;
- implement crash recovery for inbox files and incomplete final archive
  records;
- build game-specific target materializers;
- connect materialized eligible positions to replay credits;
- add archive inspection and replay-rebuild commands;
- replace the converted sample-writing path.

Review evidence:

- concurrent publisher tests produce complete, uniquely named files;
- interruption tests recover every durably published game exactly once;
- archive rebuild produces the same ordered compact samples and credits as live
  ingestion;
- stored fixtures can derive every initially configured target.

### R4 — RAM replay, batch construction, and DDP integration

Deliverables:

- implement the bounded compact FIFO and FIFO eviction;
- define the replay freeze/ingest phase transition;
- distribute a frozen snapshot to persistent DDP ranks;
- adapt deterministic rank sampling;
- implement game-specific augmentation and direct preallocated batch encoding;
- overlap pinned-memory batch preparation and device transfer;
- emit capacity, eviction, credit, freshness, memory, and timing telemetry;
- replace the converted disk-backed training path.

Review evidence:

- FIFO capacity, order, eviction, and phase transitions pass deterministic
  tests;
- DDP tests show reproducible, non-overlapping rank samples;
- encoded chess and Go batches match reference fixtures;
- a short optimizer smoke test completes for both games.

### R5 — Go evaluation and elapsed checkpoint scheduling

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

### R6 — Resource-aware experiment queue

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

### R7 — Integrated validation and benchmark preparation

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

### R8 — Target-hardware baseline and screening experiments

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
| 2026-08-05 | R2 | Open decision | Define the result target, sample eligibility, and weight for a Go game terminated by the maximum-move safety bound. | Decide before implementing safety-cap target materialization. |
