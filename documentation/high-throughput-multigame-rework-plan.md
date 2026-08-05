# PreRework-based multi-game experimentation rework plan

Status: active implementation plan.

Reference baseline: `PreRework` / `f8cb82a`.

This plan replaces the clean-slate Go-first rework direction. The rework starts
from the proven `PreRework` chess platform, preserves its high-throughput native
search, inference, training, and evaluation behavior, and generalizes that
system incrementally for small-board Go and configuration-driven experiments.

## 1. Objective

Build a research platform that can test AlphaZero improvements quickly on
small-board Go and transfer promising game-independent improvements to chess.

The intended experimental progression is:

1. screen ideas on 7x7 Go;
2. confirm promising results on 9x9 Go;
3. run bounded chess transfer pilots;
4. use confirmed improvements in the final chess training run.

The initial target is a useful 7x7 Go learning curve in approximately six hours
on one or two GPUs. The actual duration, topology, model, and search budget are
frozen only after the complete reworked system can be run on the target compute
node.

Small-board Go is a screening environment, not proof of chess transfer. One
clear positive 7x7 result is sufficient to promote an idea to the next test.
Small, ambiguous, or publication-level claims require additional seeds or
confirmation. Game-specific ideas are not expected to transfer.

## 2. Migration strategy

The work is an incremental restructuring of `PreRework`, not another
clean-slate implementation.

- Preserve current `master` under an explicit rework-prototype branch before
  changing its history.
- Preserve `PreRework` unchanged as the authoritative working reference.
- Restore `master` to the selected `PreRework` snapshot.
- Work on `master` after the safety references exist.
- Preserve the proven native hot paths until their generalized replacements
  exist.
- When a component is replaced, remove its obsolete implementation and
  compatibility code in the same phase. Do not accumulate a permanent legacy
  path.
- Intermediate commits need not provide a complete end-to-end training system.
  Changed components must still be focused, formatted, compiled where
  applicable, and covered by relevant deterministic or unit tests.
- Full system integration, performance validation, and learning experiments
  occur after the planned components are assembled and compute is available.

Historical web play, UCI, interactive play, and Lichess deployment are not
initial rework requirements. Their reference implementations remain available
on `PreRework` and may be restored after the training and experimentation path
is complete.

## 3. Responsibility boundaries

### 3.1 Python

Python retains the high-level responsibilities that are easy to develop and
change:

- typed experiment configuration and validation;
- process and hardware orchestration;
- selection of full, fast, progressive, mixed, or other configured root-search
  budgets;
- game-level decisions such as resignation, early adjudication, sample
  eligibility, and target weighting;
- advancement of completed root searches and high-level game cohorts;
- completed-game ingestion and replay maintenance;
- game-specific target construction and batch encoding;
- model definitions, losses, auxiliary heads, training, and autograd;
- checkpointing, model publication, evaluation scheduling, aggregation, and
  reporting;
- multi-experiment queue scheduling.

Python may keep an independent trusted rules implementation for differential
testing, such as `python-chess`. It is not the production search implementation.

### 3.2 C++

C++ retains the proven performance-critical path:

- native game positions, legal actions, transitions, terminal detection, and
  scoring;
- action and policy mapping;
- network-input encoding;
- MCTS tree storage, selection, expansion, backup, subtree reuse, and virtual
  loss;
- multi-root search and leaf batching;
- LibTorch model loading, inference, reusable buffers, CUDA streams, and model
  refresh;
- native search and inference telemetry.

Python configures a cohort of root searches and calls the coarse native search
interface. C++ completes those searches without returning individual leaves to
Python. Python may then advance games and choose the next root budgets.

Policies that operate once per root may be selected in Python. Policies that
inspect or change individual simulations, such as adaptive stopping or
sequential halving, execute in C++ from typed runtime configuration.

### 3.3 Model refresh

Publishing a new self-play model drains active native work and resets all
search trees. The initial implementation does not retain visits or values
created by an older model generation. Cross-generation tree retention is a
separate future experiment.

Inference caching is disabled in the initial restored baseline. It is restored
only if a measured workload justifies it.

## 4. Game specialization

One experiment selects one game. Native hot paths use compile-time game and
layout specialization; search parameters and research features remain runtime
configuration.

The Go configuration contains a board-size value. The native factory maps that
value to an explicitly compiled supported specialization, initially 7x7 and
9x9. Separate `Go7ExperimentConfiguration` and
`Go9ExperimentConfiguration` serialization types are not required.

Do not design a speculative universal game framework in advance. Begin with
the requirements demonstrated by the existing chess implementation and the Go
implementation. Extract a shared contract only when both concrete games need
the behavior. Keep genuinely different model, replay, target, and loss
structures game-specific.

The initial game-specific boundary is expected to cover the following.

### 4.1 Native game/search behavior

- position and action types;
- board dimensions and action-space size;
- initial-position construction;
- legal-action generation;
- immutable-style child construction;
- terminal detection, result, scoring, and value perspective;
- action encoding, decoding, and policy indexing;
- state hashing, equality, and repetition or ko semantics;
- compact network-input encoding;
- supported exact symmetries;
- completed-game serialization primitives.

### 4.2 Python learning behavior

- typed rules and representation configuration;
- model construction and native inference artifact export;
- completed-game decoding;
- compact replay-sample construction;
- input, policy, value, weight, and auxiliary-target construction;
- augmentation and symmetry selection;
- batch encoding;
- loss calculation and output-head handling;
- game-specific evaluation opponents and result interpretation.

### 4.3 Model artifact contract

Every exported inference artifact identifies at least:

- the game and concrete native specialization;
- rules and representation version;
- board and input shape;
- input dtype;
- action count;
- output heads and their shapes;
- model generation and artifact hash.

The native worker validates this metadata before model warm-up or refresh.

## 5. Configuration

### 5.1 Authored configuration

Experiments are authored as YAML files. A frozen Pydantic model validates every
field at load time. The root model is a discriminated union of complete game
branches, initially:

- `GoExperimentConfiguration`;
- `ChessExperimentConfiguration`.

Shared structures may be nested inside both branches, but independently
selectable unions must not permit invalid combinations such as Go rules with a
chess model, chess replay schema, or Stockfish opponent.

Search features use explicit typed variants rather than loosely related
booleans. Game and fixed data layout are compile-time specialization choices;
budgets, FPU, noise, temperature, stopping, full/fast selection, and similar
search behavior are runtime values.

### 5.2 Resolved run snapshot

The YAML file is the single authored experiment definition. At launch, its
validated Pydantic model is serialized with all resolved defaults into
canonical JSON inside the run directory. This JSON is an immutable run record,
not a second authoring source.

The snapshot also records:

- source revision and worktree state;
- dependency and native build identity;
- model initialization artifact and hash;
- complete seed configuration;
- resolved resource assignment;
- game, representation, replay, and model schema versions.

The active process uses the frozen resolved model. Changing YAML defaults or
source code after launch does not mutate an existing run. Training continuation
under changed configuration is deferred.

### 5.3 Validation

Provide a lightweight command that:

- parses one or more experiment YAML files;
- validates every complete game-specific branch;
- resolves defaults;
- reports configuration errors before resource allocation;
- optionally writes or displays canonical JSON for inspection.

The experiment runner validates its entire queue before starting the first
experiment.

## 6. Go implementation

### 6.1 Initial rules

The initial Go environment uses:

- configurable board size, initially supporting 7 and 9;
- simple ko;
- one fixed scoring rule and suicide rule per experiment;
- configured komi;
- pass as a native action;
- termination after two consecutive passes;
- a safety ply cap;
- explicit censored handling for safety-capped games.

The exact scoring, suicide, komi, history initialization, and capped-game target
semantics are frozen in configuration and tests before training.

Simple ko avoids complete rules-history storage. Rare cycles remain bounded by
the safety ply cap.

### 6.2 Bitboards and positions

Use a zero-overhead fixed-size bitboard wrapper:

```cpp
template <std::size_t NumSquares>
struct BitBoard {
    static constexpr std::size_t NUM_WORDS = (NumSquares + 63) / 64;
    std::array<uint64, NUM_WORDS> words;
};
```

The concrete implementation follows repository naming and integer-alias rules.

A Go position contains:

- black stones;
- white stones;
- player to move;
- ko point;
- consecutive passes;
- move number;
- configured komi/rules metadata as required;
- configured board history;
- an optional cached state hash if measurement justifies it.

Positions are treated as immutable. Child construction has the logical form:

```cpp
void makeChildPosition(
    const GoPosition &parent,
    Action action,
    GoPosition &destination
);
```

No undo API is required.

### 6.3 History

Absolute black and white boards are stored directly in every expanded
position. `history[0]` is the current board and increasing indices are older
boards. When a child is created, the retained parent histories are copied one
slot toward the back and the child board is written at index zero.

The initial history depth is configurable, with four or eight positions as the
first profiles. Missing initial history, channel order, ko representation, and
symmetry transformation are defined by deterministic fixtures.

Perspective canonicalization occurs only while encoding the native inference
batch.

### 6.4 Tree storage

- Complete positions exist only for materialized search nodes.
- An unmaterialized child edge stores its action, prior, visit/value
  statistics, virtual-loss state where applicable, and an invalid node index.
- Selecting an unmaterialized edge constructs its child position once and
  retains it in the allocated node.
- Preserve the proven bounded node-slot arena, generations, subtree reuse, and
  recycled per-node child-vector capacity from `PreRework`.
- Keep all legal actions in the standard PUCT baseline.
- Do not introduce top-k or top-p edge pruning into the baseline. Candidate
  restriction is a separately configured research feature.
- Do not replace per-node child vectors with a central edge arena without new
  benchmark evidence.

## 7. Self-play, search, and inference

Restore the final proven `PreRework` native path before experimenting with a
different topology:

- multiple self-play worker processes per GPU;
- many concurrent games per worker;
- native multi-root MCTS;
- one or more persistent direct-inference workers/model replicas per process as
  resolved by configuration;
- reusable pinned input and output slots;
- complete native inference batches and outstanding CUDA work;
- one tree owner for mutation where used by the proven scheduler;
- subtree reuse within one model generation;
- explicit pause, drain, refresh, resume, and shutdown behavior;
- batch, inference, selection, wait, tree, and throughput telemetry.

The restored reference topology is taken from the final measured PreRework
configuration and benchmark artifacts, not reconstructed from an older summary.
Alternative worker, thread, inference-worker, game-count, batch, timeout, or
GPU-sharing configurations are experiments after restoration.

Python selects per-root search type and budget. The initial configurable search
families include the standard baseline and the already required hooks for:

- fixed budgets;
- progressive budgets;
- mixed fast/full playout-cap randomization;
- root noise and temperature;
- FPU policy;
- policy/value target eligibility and weights;
- resignation and game-level termination policy.

Additional ideas remain in `THINGS_TO_TRY.md` and enter production one coherent
feature at a time.

## 8. Completed-game persistence

### 8.1 One atomic inbox file per completed game

Each self-play worker writes each completed game immediately as one file:

1. serialize the complete game to a temporary file in the destination
   filesystem;
2. flush and close it;
3. atomically rename it into the replay-maintainer inbox.

No producer-side list of completed games, size threshold, timeout, background
multi-game shard, shared append file, or cross-process file lock is required.

The filename contains a simple stable identity derived from run identity,
logical worker identity, and the worker's monotonically increasing game index.
The same move sequence produced twice remains two valid games. RAM replay
samples do not retain this identifier unless a concrete debugging requirement
needs it.

### 8.2 Single replay maintainer and per-generation archives

Exactly one trainer-side replay maintainer owns inbox consumption. For each
published game it:

1. reads and validates the complete record;
2. ignores it if that stable game identity is already present in the durable
   archive;
3. appends one complete game record to the append-only archive for its source
   model generation/iteration;
4. makes the archive append durable;
5. materializes eligible training samples into the active RAM replay;
6. awards credits for the newly materialized eligible positions;
7. deletes the consumed one-game inbox file.

Only the replay maintainer writes the archives, so no file locking is needed.
Archives preserve one logical record per game so game counts and individual
records remain easy to inspect. The exact text or binary codec is chosen with
the completed-game schema; the design does not require disk-ready training
tensors.

On restart, the maintainer uses archived game identities to tolerate a crash
after an archive append but before inbox deletion without double-appending or
double-crediting the game. The RAM replay is reconstructed from the newest
archives until its configured position capacity is reached.

### 8.3 Completed-game record

The durable record stores the primitive information required by the initial
training objectives and likely generic auxiliary targets:

- run, worker, game, model-generation, rules, and representation identity;
- initial position or sufficient compact per-ply state;
- actions played and player to move;
- root legal actions as required;
- root priors;
- child visit counts;
- child value sums or Q values when configured;
- root value;
- configured and actual search budget;
- full/fast/search-strategy identity;
- selected action;
- target eligibility and weighting inputs;
- terminal result and reason;
- resignation information;
- seeds needed to reproduce stochastic decisions;
- focused aggregate search diagnostics required by configured reports or
  target builders.

Do not store expanded neural-network tensors. Do not collect speculative
diagnostics without a concrete target, report, or debugging use.

Completed adjacent searches allow the trainer to derive terminal value,
remaining game length, next-player policy, and similar auxiliary targets
without changing native self-play.

## 9. Active RAM replay and training batches

### 9.1 Initial simple implementation

The active replay is a bounded FIFO containing approximately 2.5 million
compact game-specific samples. The initial implementation favors clarity over
an optimized columnar layout:

- ordinary Python containers and game-specific sample objects are acceptable;
- packed boards remain bitboards or fixed word arrays;
- policy targets remain sparse action-ID/visit-count pairs;
- expanded input and dense policy tensors are not retained;
- each sample records its source model generation for replay-freshness
  telemetry;
- new samples evict the oldest samples automatically.

Do not introduce coordinated sample/action rings, shared-memory replay, mmap
training shards, compaction, leases, or other storage optimizations before an
end-to-end measurement demonstrates a need.

### 9.2 Maintenance and training phases

Replay mutation and optimizer work are phase-separated.

During replay maintenance:

- the single maintainer consumes completed games;
- it appends durable archives;
- it materializes samples and updates the FIFO;
- it calculates newly earned presentation credits.

During an optimizer quantum:

- the replay is frozen;
- the maintainer does not insert or evict samples;
- newly completed games accumulate safely in the durable inbox;
- DDP ranks train from the same frozen replay population.

After training, replay maintenance resumes and catches up with the inbox.

### 9.3 Credits

Credits are based on newly materialized eligible positions, not merely received
files or completed-game count:

```text
new eligible replay positions * configured replay factor
    = newly earned presentation credits
```

Training consumes exact presentation credits according to the configured
global batch size and optimizer steps. The initial replay-factor search range is
approximately 1-30, with 4-6 as the expected baseline region.

### 9.4 DDP and sampling

The initial implementation may duplicate the complete frozen replay in every
DDP rank. Reuse the proven deterministic distributed sampling principles:

- all ranks validate the same replay snapshot/population;
- one global sampling schedule is partitioned into disjoint rank-local
  samples;
- incomplete global batches are dropped rather than padded with duplicates;
- sampling and augmentation are reproducible from explicit seeds;
- each rank reports compatible progress before publication.

Shared-memory replay and a single centralized batch producer are deferred until
measurement shows that replay copies, construction time, or host memory are a
problem.

### 9.5 Batch construction

Game-specific batch construction receives sampled compact objects and writes
directly into preallocated final-shape tensors:

```text
inputs:            [batch, channels, height, width]
policy_targets:    [batch, action_count]
value_targets:     [batch, value_dimensions]
auxiliary_targets: game/head dependent
```

Requirements:

- no per-sample Torch tensor creation;
- no stacking of already-created sample tensors;
- sparse policies are expanded directly into the final dense policy buffer;
- augmentation is applied while decoding;
- pinned CPU buffers are reused;
- full contiguous batches transfer asynchronously to the assigned GPU;
- CPU preparation of the next batch overlaps GPU training where the simple
  implementation permits it.

Generic PyTorch `DataLoader` behavior is used only where it preserves these
properties. The existing deterministic distributed batch sampler may be
adapted rather than replaced speculatively.

### 9.6 Replay telemetry

At minimum report:

- active positions and completed games ingested;
- positions generated and evicted;
- earned and consumed credits;
- actual replay factor;
- mean and distribution of source-model age in sampled batches;
- batch-construction and transfer time;
- optimizer throughput;
- inbox backlog during training;
- replay and rank-process host memory during final integration.

## 10. Evaluation

Evaluation remains integrated with the training run as in `PreRework`.
Configured checkpoints are claimed every 20 minutes of elapsed training time;
a six-hour run therefore has 18 scheduled progress points.

Evaluation executes concurrently with self-play and training using the
experiment's fixed evaluation resource assignment. Its cost and interference
are identical configuration-controlled parts of every compared arm. Due
checkpoints may queue if the evaluator is busy, but they are never silently
skipped.

Every result records requested checkpoint time, actual model publication time,
actual evaluation start/completion time, model identity, search settings,
opponent identity, raw games, and resource cost. Partial or manually stopped
runs retain all completed checkpoint results.

### 10.1 Chess

Preserve the successful PreRework ladder and efficient native/direct search:

- MCTS versus random;
- policy-only versus random;
- previous and selected milestone checkpoints;
- pinned Stockfish skill levels;
- pinned fixed-node Stockfish;
- fixed holdout/dataset metrics where configured;
- paired colors/openings, raw outcomes, W/D/L, score, and uncertainty.

The number of historical-model matches may be reduced later through
configuration after the restored behavior is available.

### 10.2 Go

Provide the analogous ladder:

- MCTS versus seeded random;
- policy-only versus random;
- previous and selected milestone checkpoints;
- fixed baseline Go models;
- a pinned external Go engine under compatible board size, rules, komi, and
  fixed compute.

External-engine evaluation is required before Go ablations are considered
ready. The engine binary, network, protocol configuration, rules, komi, and
search/time limit are recorded in provenance. The exact engine and strength
conditions are selected while implementing the adapter.

Evaluation uses common search settings across experiment arms so network
quality and training efficiency can be compared. Native-search evaluation is
reported separately when the tested search feature itself is the subject of
the experiment.

## 11. Multi-experiment runner

The initial runner is intentionally small. It accepts an explicit queue of
experiment YAML paths and fixed Linux resource slots.

Before starting work it:

- validates every experiment configuration;
- verifies required source/build/model/external-engine artifacts;
- resolves GPU, CPU, RAM, and output-directory assignments;
- fails before launch on overlapping or impossible exclusive assignments.

For each resource slot it:

- sets the slot's `CUDA_VISIBLE_DEVICES` mapping;
- applies CPU affinity and, where practical, NUMA placement;
- applies a configured RAM limit/reservation through Linux facilities where
  practical;
- launches the complete experiment in its own process group;
- captures stdout, stderr, exit code, resolved configuration, and status;
- starts the next queued experiment immediately when the slot becomes free,
  whether the previous experiment completed or failed.

The runner maximizes configured parallelism and does not require manual
attention between six-hour runs. Initial scope excludes automatic experiment
retry, training resume, source updates, configuration mutation, sophisticated
fairness, or adaptive resource placement.

Evaluation is internal to each experiment and uses resources reserved by that
experiment. A failed experiment retains diagnostics and does not block later
queue entries.

## 12. Experimental method

The main comparison criterion is playing strength achieved under the same
configured hardware and elapsed training time.

Record at least:

- external-opponent and fixed-baseline strength over elapsed time;
- learning-curve AUC;
- final strength at the common cutoff;
- uncertainty of match results;
- games and eligible positions per hour;
- simulations and inference positions per second;
- model batch occupancy and latency;
- optimizer steps and samples per second;
- replay factor and source-model age;
- GPU, CPU, host-memory, and idle-time telemetry.

Use matched initial weights, root seeds, evaluation settings, and resource
assignments where applicable. One run is sufficient for rapid screening of a
large clear effect. Repeat ambiguous results and confirm promoted generic ideas
on 9x9 before chess transfer. Negative and neutral experiments remain recorded.

The initial feature order is:

1. stable standard AlphaZero baseline;
2. progressive simulation budgets;
3. mixed fast/full search;
4. adaptive search termination;
5. FPU variants;
6. replay factor and model-publication cadence;
7. progressive model scaling;
8. global-context architecture;
9. generic auxiliary heads;
10. reanalysis or restart-state generation;
11. Gumbel or sequential-halving search;
12. combined confirmed system;
13. chess transfer.

`THINGS_TO_TRY.md` is the research backlog, not authorization to implement all
features during the infrastructure rework.

## 13. Implementation phases

The phases define dependency order and acceptance evidence. Work inside a phase
may be intertwined. End-to-end training need not work after every intermediate
commit, but the affected components must have focused structural validation.

### Phase 0: preserve and restore PreRework

- Preserve current `master` as the rework-prototype safety branch.
- Preserve `PreRework` unchanged.
- Restore `master` to `f8cb82a`.
- Build the native extension and run relevant native/Python tests.
- Verify a short non-training self-play, trainer, publication, and evaluation
  smoke where the environment permits it.
- Freeze the final measured PreRework topology, configuration, benchmark
  commands, and expected artifacts.

Acceptance: the selected reference snapshot and exact commands are known, and
the existing chess components are validated as far as the available local
environment permits. Long training and performance execution remain deferred
until compute is available.

### Phase 1: configuration and initial game boundaries

- Replace mutable/static settings with the YAML/Pydantic configuration model.
- Add complete Go and chess branches and canonical run snapshots.
- Define the initial native game/search and Python learning boundaries from the
  two concrete implementations.
- Add the model-artifact contract and native factory dispatch.
- Adapt chess orchestration, search, training, and evaluation to the new
  configuration.
- Remove superseded settings and compatibility code during the replacement.

Acceptance: representative Go and chess YAML files validate; invalid mixed-game
configurations fail; model metadata is authenticated; affected configuration
and chess components compile and pass focused tests. Full training integration
may remain incomplete.

### Phase 2: native Go integration and evaluation

- Integrate the tested Go bitboard, rules, state, encoding, action mapping, and
  symmetry implementation.
- Add 7x7 and 9x9 native specializations selected by Go configuration.
- Generalize the proven PreRework native search and direct LibTorch inference
  interfaces without changing their chess behavior unnecessarily.
- Add Go model construction, export, target construction, augmentation, and
  loss handling.
- Retain Python root/game orchestration.
- Add random, policy-only, checkpoint, milestone, and external-engine Go
  evaluation with 20-minute checkpoint scheduling.
- Add deterministic rules, encoding, transition, terminal, policy-map, search,
  model-artifact, and evaluation fixtures.
- Remove superseded prototype adapters as their replacements become complete.

Acceptance: focused Go and chess native tests pass; Go inference artifacts run
through native search; deterministic Go match fixtures complete; the external
engine adapter records reproducible provenance. No learning-curve run is
required yet.

### Phase 3: completed-game records and RAM replay

- Define typed chess and Go completed-game schemas.
- Add atomic one-game producer files.
- Add the single inbox consumer and per-generation append-only archives.
- Add crash-safe duplicate recognition around archive append and inbox
  deletion.
- Add game-specific target materializers and the bounded in-memory FIFO.
- Record compact sparse policies and model generation.
- Add phase-separated maintenance, credit accounting, and optimizer quanta.
- Adapt deterministic DDP sampling with one frozen replay copy per rank.
- Add preallocated pinned batch construction for both games.
- Remove the replaced rolling disk-replay, HDF5 trainer dataset, compaction,
  mmap/shard-loading, and compatibility code.

Acceptance: game fixtures round-trip through disk, archive, RAM replay, and
training batch; recovery neither loses nor double-credits games; FIFO eviction
and sparse policy construction are correct; DDP partitions are disjoint and
deterministic; relevant unit and short CPU integration tests pass.

### Phase 4: lifecycle and experiment runner

- Integrate the frozen YAML configuration into run creation and provenance.
- Preserve model publication and concurrent 20-minute evaluation scheduling.
- Add the explicit experiment queue and fixed resource slots.
- Add GPU visibility, CPU affinity, practical RAM isolation, process-group
  cleanup, logging, and failure continuation.
- Add experiment status and aggregate result discovery.
- Exclude resume and retry behavior from the initial implementation.
- Remove replaced legacy launch/configuration paths.

Acceptance: a local synthetic queue validates all entries, uses configured
slots, records success and failure, cleans up processes, and starts later jobs
without manual intervention.

### Phase 5: complete integration and compute readiness

- Assemble full Go and chess paths through configuration, self-play, native
  inference/search, completed-game persistence, RAM replay, DDP training,
  publication, evaluation, and reporting.
- Run complete relevant native and Python unit/integration suites.
- Prepare frozen 7x7, 9x9, and chess configurations.
- Prepare the exact compute-node build, smoke, monitoring, runner, and artifact
  collection commands.
- Inspect host-memory usage, DDP replay copies, batch construction, inbox
  backlog, and evaluation concurrency during the first complete run.
- Optimize only measured bottlenecks.

Acceptance: the system is structurally ready for the rented compute node. No
playing-strength, throughput, or six-hour-duration claim is made before target
execution.

### Phase 6: baseline and ablation execution

- Establish the standard 7x7 baseline and determine the useful screening
  duration from its measured learning curve.
- Confirm the selected topology and evaluation cadence.
- Establish the 9x9 confirmation profile.
- Queue controlled feature experiments in the planned order.
- Promote clear improvements through 7x7, 9x9, and bounded chess gates.
- Freeze the confirmed combined system before the final chess run.

Acceptance: results contain immutable configurations, source/build/model
identity, complete telemetry, scheduled evaluations, raw match records, and
comparable reports.

## 14. Validation and implementation discipline

- Commit coherent feature-sized changes iteratively.
- Preserve unrelated workspace changes.
- Fully type Python boundaries and use explicit game-specific structures.
- Use the repository C++20, PCH, integer aliases, formatting, warnings, and
  clang-tidy configuration.
- Keep search hot paths allocation-conscious, but do not replace measured
  adequate structures without evidence.
- Use deterministic fixtures for rules, encodings, policy maps, search
  accounting, serialization, sampling, credits, and evaluation.
- During intermediate phases, run all checks relevant to the changed
  components even when the complete application is temporarily not runnable.
- Before compute execution, run the complete relevant native and Python suites.
- Do not run performance benchmarks, prolonged self-play, training, or full
  evaluation until the user confirms the target compute environment is
  available.
- Report every command run, failure, skipped hardware gate, and deferred
  measurement honestly.

## 15. Immediate next actions

1. Create the safety reference for the current Go rework prototype.
2. Restore `master` to `PreRework` without modifying the reference branch.
3. Validate the restored baseline and record exact commands/environment gaps.
4. Replace the static settings system with the typed YAML configuration root.
5. Derive the first shared game boundary from the existing chess and Go
   implementations.

Do not begin replay replacement, scheduler work, or research-feature ablations
before the configuration and two concrete game boundaries are understood.
