# Go-first research platform: target architecture and migration inventory

Status: implementation contract for the clean-slate rework. The `PreRework`
branch at commit `f8cb82a` is the authoritative legacy snapshot; compatibility
with its Python APIs, checkpoints, replay files, and run configurations is not
required.

## 1. Goals

The rework must produce a small, auditable research platform for equal-hardware,
equal-wall-clock AlphaZero-style experiments on 7x7 Go, with 9x9 Go as a
configuration change. It must:

- make the game, search algorithm, search budget, stopping policy, FPU policy,
  model schedule, objectives, optimizer, start-state policy, and evaluation
  protocol explicit in a fully resolved immutable run configuration;
- keep experiment orchestration reusable without pretending that model, dataset,
  or loss semantics are game-independent;
- have exactly one production MCTS implementation, in native C++;
- support reproducible independent seeds and record enough provenance to explain
  every training example and every search-compute decision;
- preserve the proven operational ideas from the current system: wall-clock
  limits, hardware validation, credit-based replay accounting, distributed
  training, atomic checkpoint publication, artifact retention, evaluation
  scheduling, and resource telemetry;
- make fixed, progressive, mixed fast/full, and adaptive search isolated
  strategies suitable for pre-registered ablations; and
- favor focused modules, explicit typed data, and replaceable boundaries over
  compatibility adapters and speculative generality.

The first confirmatory experiment is deferred until a separate GPU node is rented.
This rework includes the runner, telemetry, evaluation protocol, and local/system
validation needed to be ready for that experiment, but not the rental run or a
claim about baseline playing strength.

## 2. Non-goals

- Preserving old checkpoint, optimizer, replay, JSON configuration, Python API,
  UCI, web-play, or deployment compatibility.
- Porting chess, checkers, Connect Four, Hex, or Tic-Tac-Toe during the Go
  implementation. Chess is a later boundary test, not an initial requirement.
- Designing a universal tensor, dataset row, neural network, loss, adjudication
  rule, or game metadata schema.
- Providing a Python MCTS fallback or maintaining two search implementations.
- Implementing Gumbel/sequential-halving search, auxiliary heads, population-based
  optimization, or restart-state sampling before the initial four-way search
  comparison is operational.
- Establishing final simulation counts, komi, adaptive thresholds, or baseline
  strength before separate calibration runs.
- Recreating the interactive chess engine, GUI, web service, Lichess deployment,
  or Stockfish evaluation surface in the initial platform.

## 3. Ownership boundaries

### 3.1 Game-owned Python code

Each game package owns the meaning and shape of its learning data:

- immutable game and rules configuration;
- encoded-state, training-sample, and training-batch types;
- replay payload serialization and decoding;
- model construction and typed model outputs;
- loss construction, target interpretation, and game-specific metrics;
- data augmentation and symmetry transforms;
- evaluation opponents, suites, color/komi stratification, and game diagnostics;
- translation between the native game/search result and the game-specific replay
  payload.

For Go, this includes board size, komi, scoring rule, ko rule, history planes,
action layout, pass handling, symmetry, score/value targets, model heads, and any
future Go-specific auxiliary targets. The shared trainer asks the selected game
module to decode batches, run its model, and calculate a typed loss result; it
does not inspect Go tensor planes or output-head names.

The extension point is a small typed game module/factory interface. It is not a
common base class for samples or models and must not use raw dictionaries,
dynamic attribute access, or `Any`.

### 3.2 Shared Python code

Shared code owns operations whose semantics do not depend on the rules of Go:

- parsing, validating, resolving, hashing, and storing experiment configuration;
- run identity, manifests, source revision, seed allocation, approvals, and
  hardware/topology validation;
- process lifecycle, GPU assignment, wall-clock enforcement, and graceful
  shutdown;
- self-play work scheduling and native worker lifecycle;
- replay shard lifecycle, capacity, sampling, credits, reuse accounting, and
  schema dispatch without interpreting game payloads;
- optimizer-step scheduling, DDP lifecycle, checkpoint publication, and
  artifact retention;
- elapsed-time checkpoint and evaluation scheduling;
- generic resource, throughput, and search-provenance aggregation;
- experiment matrices, multi-seed aggregation, confidence intervals, and report
  generation.

Shared code may depend on explicit game interfaces. A game package must not
import process orchestration, replay storage implementations, or experiment
runner internals.

### 3.3 Native C++ code

Native code is the only owner of:

- production game-state mutation used during search;
- legal action generation, terminal detection, scoring, hashing, copying, and
  symmetry primitives;
- MCTS tree selection, expansion, backup, action selection, tree reuse, and
  search termination;
- PUCT/FPU/noise semantics and search-budget accounting;
- inference batching and result association;
- per-move search telemetry emitted to Python.

The shared native search core is parameterized by a concrete game implementation
and compiled into explicit bindings. It must not contain Go constants, and the Go
implementation must not contain PUCT policy. Compile-time composition is
preferred over virtual dispatch inside the simulation loop.

Python owns orchestration and training, but never traverses a search tree. A slow
Python Go rules oracle is permitted under tests only; it is not a self-play or
evaluation implementation.

## 4. Target module layout

Names may receive small adjustments during implementation, but ownership must
remain as shown.

```text
py/
  src/az/
    config/
      models.py                 # frozen Pydantic configuration graph
      resolution.py             # authoring config -> complete resolved config
      manifest.py               # hashes, revision, seed and environment record
    experiment/
      runner.py
      matrix.py
      reporting.py
      evaluation_schedule.py
      artifact_retention.py
    runtime/
      topology.py
      processes.py
      resource_telemetry.py
      shutdown.py
    self_play/
      coordinator.py
      native_worker.py
    replay/
      envelope.py               # universal provenance only
      codec.py                  # typed codec protocol
      storage.py
      sampling.py
      credits.py
    training/
      trainer.py
      distributed.py
      checkpoints.py
      optimizer.py
    evaluation/
      coordinator.py
      protocol.py
      statistics.py
    games/
      api.py                    # narrow factories/protocols
      go/
        configuration.py
        samples.py
        replay_codec.py
        model.py
        losses.py
        augmentation.py
        evaluation.py
  test/
    unit/
    integration/
    system/

cpp/
  src/v2/
    games/
      game_concepts.hpp
      go/
        GoState.hpp
        GoState.cpp
        GoEncoding.hpp
        GoEncoding.cpp
        GoSymmetry.hpp
        GoSymmetry.cpp
    search/
      SearchConfiguration.hpp
      SearchTree.hpp
      Puct.hpp
      Fpu.hpp
      Budget.hpp
      Stopping.hpp
      SearchTelemetry.hpp
    inference/
      InferenceModel.hpp
      InferenceBatcher.hpp
    self_play/
      SelfPlayRunner.hpp
    bindings/
      module.cpp
      go_bindings.cpp
  test/v2/
```

New code lives alongside legacy code until its end-to-end replacement is
validated. Legacy roots are then removed in a dedicated cleanup stage; they
must not be wrapped indefinitely.

## 5. Typed configuration contract

The launched run artifact contains one fully resolved, frozen Pydantic model with
`extra="forbid"`. Defaults may be used while authoring matrices, but all defaults
are materialized before validation, hashing, approval, and launch. Important
values must not be inherited from module globals or mutable dataclasses.

The root contains these categories:

- `experiment`: name, hypothesis/arm identifier, seed, duration, checkpoint
  times, output location, source revision, and manifest policy;
- `hardware`: provider/offer identity, expected GPU model/count, CPU/RAM/disk
  minima, and hourly cost;
- `topology`: trainer ranks/devices, self-play workers/devices, native threads,
  inference workers/batches, data-loader workers, and evaluation concurrency;
- `game`: a discriminated union, initially `GoGameConfiguration`, containing
  board size, komi as an integer number of half-points, scoring, ko, ply cap,
  history, and resignation policy;
- `model`: game-selected model family plus a discriminated schedule;
- `search`: algorithm, budget, stopping, FPU, root exploration, temperature,
  tree reuse, inference batching, and explicit backup discount;
- `self_play`: start-state strategy, game concurrency, policy-target eligibility,
  and sample weighting;
- `replay`: capacity, credit/reuse policy, shard/storage parameters, and
  game-payload schema version;
- `training`: batch size, optimizer, learning-rate schedule, precision,
  objective configuration owned by the game, and checkpoint cadence;
- `evaluation`: common search settings, checkpoint schedule, games/pairs,
  confidence procedure, and game-owned opponent/suite configuration;
- `telemetry` and `retention`: output cadence, required metrics, and artifact
  retention rules.

Closed behavioral choices use discriminated unions rather than feature flags:

```text
game
  GoGameConfiguration
  ChessGameConfiguration                         # later

search.algorithm
  PuctSearchConfiguration
  GumbelSearchConfiguration                      # later

search.budget
  FixedSearchBudget
  ProgressiveSearchBudget                       # explicit elapsed-time stages
  MixedSearchBudget                             # cheap/full probability and caps

search.stopping
  FullBudgetStopping
  AdaptiveSearchStopping                        # minimum work and calibrated rule

search.fpu
  ParentValueFpu
  ReducedParentValueFpu
  VisitedChildMeanFpu

model.schedule
  FixedModelSchedule
  ProgressiveModelSchedule

self_play.start_states
  InitialStateOnly
  RestartStatePool                              # later
```

Mixed search explicitly defines which searches contribute policy targets and
their weights; every state may still contribute a value target. Strategy
composition is validated so, for example, adaptive stopping has a finite maximum
budget and progressive stages are ordered by elapsed time.

## 6. Replay ownership and contract

Replay storage never assumes FEN, moves in UCI notation, piece counts, an 8x8
board, a fixed action count, or a particular tensor layout. Each stored record
has two parts.

Universal envelope/provenance:

- run identifier, game identifier, and payload schema version;
- stable sample/game identifiers, seed lineage, creation time, and ply;
- source model/checkpoint version;
- configured search strategy and budget class;
- configured cap and actual simulations;
- stop reason and policy-target eligibility/weight;
- value-target eligibility/weight;
- root visit count, entropy, top-two margin, and optional prefix/full
  disagreement measurements;
- termination reason for the completed game;
- replay credit identity needed for exactly accounted reuse.

Game-specific versioned payload:

- encoded position/history;
- sparse or dense policy/visit target in the game's action space;
- value/score and auxiliary targets defined by the game;
- game-specific terminal and diagnostic metadata.

The selected `ReplayCodec[Sample, Batch]` serializes payloads and creates typed
batches. Shared storage validates envelope and schema identity but treats payload
bytes as opaque. Changing Go planes or heads increments the Go payload schema;
no backward-compatibility reader is required unless a later experiment explicitly
needs it.

## 7. Native Go/search boundary

The initial native Go contract supports board size 7 or 9 using the same code:

- action space `board_size * board_size + 1`, with pass as the final action;
- area scoring with explicit `komi_half_points: int`; scores are compared in
  doubled integer points, never by unconstrained binary floating-point equality;
- positional superko rejects a stone-placement action that recreates any earlier
  board position in the game;
- suicide is illegal: a stone-placement action is legal only when, after removing
  captured opposing stones, the placed stone's connected group has a liberty;
- pass is always exempt from positional-superko repetition rejection, so two
  consecutive passes remain legal and can terminate the game;
- two consecutive passes as normal termination;
- a configured safety ply cap reported as a distinct censored termination. No
  heuristic winner or synthetic terminal value is substituted: every position
  from a capped game has `value_target_weight = 0`, while a position's policy
  target remains eligible exactly when its search strategy normally makes it
  eligible;
- deterministic state hashing and exact copy semantics;
- eight dihedral symmetries with action and tensor round trips;
- history sufficient for the configured encoder.

The Go implementation exposes the operations required by the native game concept:
legal actions, apply action, current player, terminal result, canonical encoding,
state hash, equality/copy, and symmetry mapping. Search receives only this
concept, a typed inference interface, a typed search configuration, and an
explicit random stream.

Search returns the selected action, root targets, value information, terminal
information where applicable, and a `SearchTelemetry` record. Python bindings
must expose typed result objects rather than positional tuples or dictionaries.
No Stockfish type, FEN/UCI representation, chess action constant, or board-size
constant may appear in `cpp/src/v2/search` or `cpp/src/v2/inference`.

## 8. Determinism and experiment identity

- A run has a required root seed. A documented, stable derivation creates
  independent seeds for process, worker, game, search, Dirichlet noise, action
  sampling, replay sampling, model initialization, augmentation, and data-loader
  order.
- Native code receives seeds from Python. Production randomness must not use
  `std::random_device`, wall-clock time, process ID, unordered container iteration,
  or an implicit global generator.
- Random draws belong to stable logical entities such as `(run, worker, game,
  ply, purpose)`, so process timing does not silently change seed assignment.
- The resolved configuration hash, source commit, dirty-tree patch hash, build
  information, dependency versions, hardware identity, and complete seed
  derivation version are stored in the manifest.
- Same configuration, seed, build, and deterministic execution mode must produce
  identical native rule trajectories and search results in single-thread tests.
  GPU kernels and concurrent production scheduling may be nondeterministic; this
  must be declared in the manifest and evaluated through independent run seeds,
  never concealed as bitwise reproducibility.
- Checkpoint resume restores every relevant Python RNG, native stream counter,
  replay sampler state, optimizer/scaler state, and elapsed-time schedule state.

## 9. Testing and validation strategy

Validation is layered; a later layer does not substitute for an earlier one.

1. **Configuration unit tests:** discriminators, forbidden fields, cross-field
   invariants, complete resolution, stable hashes, and serialization round trips.
2. **Go rules oracle:** a small test-only Python implementation is cross-checked
   against C++ across fixtures and randomized legal trajectories: liberties,
   captures, suicide, ko/superko, pass, termination, scoring, hashing, copying,
   7x7/9x9 behavior, and symmetry round trips.
3. **Native search unit tests:** hand-built deterministic trees verify selection,
   expansion, alternating backup, explicit discount, each FPU policy, root noise,
   temperature, budget strategies, adaptive stop reasons, policy eligibility,
   telemetry, and exact simulation accounting.
4. **Replay/training unit tests:** envelope validation, Go payload round trips,
   augmentation consistency, loss math, policy masks/weights, credit accounting,
   replay capacity/reuse, and checkpoint resume.
5. **Integration tests:** native self-play produces consumable Go shards; a
   trainer consumes them and publishes a loadable model; workers refresh models;
   elapsed-time schedules select the expected strategy.
6. **System smoke test:** one local short run performs self-play, replay,
   optimization, checkpointing, restart, evaluation, telemetry, graceful
   shutdown, and report generation. The later full training run is the final
   system-level test, not a replacement for these checks.
7. **Static/format validation:** Python type annotations and Pydantic types follow
   repository rules; run `ruff format`, `ruff check --fix`, and
   `python -m pytest --import-mode=importlib .\test -q` from `py`. Native changes
   build Release and Debug test targets, run CTest, and use the repository
   clang-format/clang-tidy configuration.

Tests that require GPUs or external infrastructure are marked integration and
skip with an explicit reason when prerequisites are absent. Native-extension
collection must also skip clearly rather than fail because a local build is
missing.

## 10. Sequential implementation stages

Each stage is implemented by a worker, reviewed for logic and boundaries, fully
validated, and committed before the next begins.

### Stage 1: architecture contract and reference point

Deliver this document and retain `PreRework` at the authoritative legacy commit
`f8cb82a` while continuing on `master`.

Acceptance: boundaries, inventory, replacements, stages, and deferred work are
explicit; no production behavior changes.

### Stage 2: configuration and manifest v2

Add the frozen typed configuration graph, resolved manifest, game registry, seed
derivation, and CLI validation/printing. Do not adapt legacy `TrainingArgs`.

Acceptance: representative fixed/progressive/mixed/adaptive Go configurations
resolve without hidden values; invalid combinations fail clearly; hashes and
round trips are tested; no import-time game selection exists in v2.

### Stage 3: native Go correctness core

Add test-only Python rules, native Go state/encoding/symmetry, typed bindings, and
cross-implementation/property tests.

Acceptance: all listed rule cases pass for 7x7 and 9x9; randomized native/oracle
trajectories agree; state copies, hashes, encodings, and symmetries are
deterministic; no search or training behavior is introduced.

### Stage 4: generic native PUCT and inference

Extract/build the game-independent search, seeded RNG, inference batching, typed
results, fixed budget, and complete telemetry. Implement against Go only.

Acceptance: deterministic tree fixtures pass; simulation totals and backup signs
are exact; search/inference contain no Go/chess constants; a fixed-budget native
Go search runs through Python.

### Stage 5: Go model, replay payload, and losses

Add the Go-owned model, typed outputs, samples/batches, codec, augmentation, and
baseline policy/value loss; add shared envelope and storage.

Acceptance: native results round-trip through replay into a training batch;
symmetry transforms preserve actions/targets; masked/weighted losses match
fixtures; shared replay code does not inspect Go payloads.

### Stage 6: credit trainer and checkpoint lifecycle

Rebuild reusable credit accounting, sampling, DDP trainer, optimizer schedule,
atomic checkpoints, resume, publication, and artifact retention against v2
interfaces.

Acceptance: exact replay reuse is demonstrated by tests; restart reproduces
ledger/optimizer/scheduler/RNG state; a CPU integration test trains a Go model
from generated shards and publishes a loadable checkpoint.

### Stage 7: self-play/runtime orchestration

Add topology validation, native worker processes, model refresh, wall-clock
shutdown, resource telemetry, and failure propagation.

Acceptance: a local multi-process smoke run generates data while training,
refreshes its model, respects the time limit, shuts down cleanly, and records no
orphan workers or uncredited samples.

### Stage 8: search-compute strategies

Add progressive and mixed budgets, adaptive trace collection/stopping, and the
three FPU policies as independent native strategies.

Acceptance: every strategy has deterministic unit fixtures; mixed searches set
policy weights correctly; progressive stages use elapsed time; adaptive runs
record actual simulations and stop reasons; fixed-search behavior is unchanged.

### Stage 9: evaluation and research reporting

Add common-search paired Go evaluation, elapsed-time checkpoint scheduling,
bootstrap intervals, AUC/strength-per-hour aggregation, multi-seed matrices, and
the required diagnostic report.

Acceptance: synthetic fixtures validate statistics; evaluations are color/komi
balanced; reports include games/positions, actual simulations, budget class,
policy eligibility, utilization, optimizer/replay totals, stop frequency, and
prefix/full disagreement; evaluation cost is accounted separately.

### Stage 10: legacy removal and repository cleanup

Delete the superseded implementation and rename/move v2 into the final paths.
Keep historical documents and benchmark artifacts as evidence, clearly labeled
legacy.

Acceptance: imports/build files contain no legacy game selector, Python MCTS,
`TrainingArgs`, chess/Stockfish dependency, UCI/web deployment, or old replay
schema; tests collect cleanly; documentation and entry points describe only the
new Go platform.

### Stage 11: end-to-end readiness

Run the complete local/system validation, configuration freeze tooling, and a
small non-claiming throughput/correctness pilot.

Acceptance: a fresh checkout can build, validate, run, stop, resume, evaluate,
and report a short 7x7 experiment from one resolved configuration. Final
strength calibration and rented equal-hardware six-hour runs remain deferred.

## 11. Deletion and migration inventory

The disposition below applies after replacements pass their acceptance stages.
“Retain” means retain the capability, not necessarily the current file or API.

### 11.1 Python production code

| Current path | Disposition | Replacement or reason |
| --- | --- | --- |
| `py/src/mcts/` | Delete | Native `cpp/src/v2/search`; no Python MCTS. |
| `py/src/self_play/SelfPlayPy.py` | Delete | `az/self_play/native_worker.py`. |
| `py/src/eval/AlphaZeroBotPy.py`, `ModelEvaluationPy.py` | Delete | Native Go evaluation through `az/evaluation`. |
| `py/src/settings.py`, `settings_common.py` | Delete | Fully resolved `az/config`; no import-time game. |
| `py/src/games/Board.py`, `Game.py`, `GameVisuals.py` | Delete | Narrow factories/protocols in `az/games/api.py`; no universal board/model hierarchy. |
| `py/src/games/chess/`, `checkers/`, `connect4/`, `hex/`, `tictactoe/` | Delete from active code | `PreRework` at `f8cb82a` remains the authoritative legacy snapshot. Later chess is a new game package. |
| `py/src/Network.py`, `Encoding.py`, `value.py` | Delete | Go-owned model/encoding/output types; shared code does not impose them. |
| `py/src/self_play/SelfPlayCpp.py` | Adapt concepts, replace file | Native worker/coordinator with one responsibility each. Preserve proven batching/model-refresh ideas only after review. |
| `py/src/self_play/SelfPlayDataset.py`, `SelfPlayDatasetStats.py` | Replace | Universal replay envelope/storage plus Go-owned payload/codec/metrics. |
| `py/src/self_play/curriculum.py` | Replace | Typed elapsed-time budget/model schedule strategies. |
| `py/src/self_play/model_refresh.py` | Adapt | Focused runtime model-publication subscriber. |
| `py/src/self_play/resignation.py`, `value_target.py` | Move semantics to Go | Go game/self-play configuration and Go loss target construction. |
| `py/src/train/LegacyIterationReplayBuffer.py` | Delete | Credit-only replay. |
| `py/src/train/TrainingArgs.py` | Delete | Frozen resolved configuration; no mutable parallel settings layer. |
| `py/src/train/RollingReplayBuffer.py`, `CreditTrainingLedger.py`, `CreditPublication.py` | Retain capability, rewrite/adapt | `az/replay/storage.py`, `credits.py`, and checkpoint publication with game-opaque payloads. |
| `py/src/train/DistributedTraining.py`, `Trainer.py`, `TrainingStats.py` | Retain capability, rewrite/adapt | Focused DDP, trainer, checkpoint, and telemetry modules using game factories. |
| `py/src/train/ReplayReanalysis.py` | Defer | Later typed replay transformation; excluded from the initial comparison. |
| `py/src/cluster/` | Retain capability, replace decomposition | `az/runtime`, `az/self_play`, `az/training`, and `az/evaluation`; eliminate broad process classes and duplicated cached/non-cached clients. |
| `py/src/experiment/run_configuration.py` | Replace | Split `az/config` graph/resolution/manifest. |
| `py/src/experiment/artifact_retention.py`, `cost_accounting.py`, `evaluation_schedule.py`, `resource_telemetry.py` | Adapt | Preserve validated behavior behind v2 types. |
| `py/src/experiment/credit_telemetry.py`, `progress_telemetry.py`, `evaluation_protocol.py` | Adapt/split | Typed telemetry records and evaluation/statistics/reporting. |
| `py/src/experiment/plateau.py` | Defer | Not part of fixed elapsed-time primary experiment. |
| `py/src/eval/` other than Python-MCTS files | Delete or defer | Replace research evaluation in `az/evaluation`; GUI/human/tournament surfaces are out of scope. |
| `py/src/uci/`, `py/web_play/` | Delete from active platform | Chess deployment is out of scope; history remains in Git. |
| `py/src/util/communication.py`, `background_worker.py`, `save_paths.py`, `tensorboard.py`, `timing.py`, `log.py` | Adapt selectively | Move only required lifecycle, checkpoint, telemetry, timing, and logging behavior into focused v2 modules. |
| `py/src/util/mcts_graph.py`, `ZobristHasherNumpy.py`, `ZobristHasherTorch.py` | Delete | No Python tree; native Go owns production hashing. |
| Remaining `py/src/util/` profiling/download helpers | Defer or delete | Reintroduce only for a demonstrated v2 requirement. |

### 11.2 Native code

| Current path | Disposition | Replacement or reason |
| --- | --- | --- |
| `cpp/src/Board.*`, `BoardEncoding.*`, `MoveEncoding.*`, `GameHistory.*`, `common.hpp` | Delete after Go replacement | Chess/Stockfish-specific state becomes `v2/games/go`; shared constants become typed configuration. |
| `cpp/src/MCTS/` | Retain algorithms as reference, rewrite | `v2/search` with explicit budget/stopping/FPU/discount/RNG and no game constants. |
| `cpp/src/DirectInference.*`, `InferenceModel.hpp`, `InferenceResultProcessing.*` | Adapt | Focused `v2/inference` with typed game-independent shapes supplied at construction. |
| `cpp/src/InferenceClient.*`, `NonCachingInferenceClient.*`, related types | Replace/simplify | One explicit batching interface and optional cache strategy, without parallel semantic paths. |
| `cpp/src/InteractiveEngine.*`, `InteractiveSearch.*` | Delete | Chess interactive/UCI deployment is out of scope. |
| `cpp/src/binding.cpp` | Replace | Small bindings split by native domain and typed result objects. |
| `cpp/src/util/BlockingQueue.hpp`, `ShardedCache.hpp`, `ThreadPool.*` | Retain selectively | Reuse only after focused tests prove they meet v2 lifecycle/determinism needs. |
| `cpp/src/util/CollisionCheckedCache.hpp`, logging/timing utilities | Adapt or delete | Keep only used, focused infrastructure. |
| Stockfish fetch, patch, sources, and include paths in `cpp/CMakeLists.txt` | Delete | Native Go has no Stockfish dependency. |
| Existing C++ chess tests/benchmarks | Delete after replacement | New Go-rule, generic-search, inference, and lifecycle tests/benchmarks. |

### 11.3 Entry points, configuration, deployment, and tests

| Current path | Disposition | Replacement or reason |
| --- | --- | --- |
| `py/train.py` | Replace | One v2 experiment CLI: validate, resolve, run, resume, and report. |
| `py/eval.py`, `opt.py`, shell entry points | Delete or defer | Evaluation is scheduled from the experiment CLI; population optimization is later. |
| `py/configs/chess-*.json` | Archive as legacy documentation, remove from active configs | New resolved Go configurations and matrix authoring inputs. |
| `py/AlphaZeroCpp.pyi`, built `.so` | Regenerate | Bindings/types for the new native module; build artifacts are not source. |
| `py/test/` | Replace incrementally | Keep tests only for retained v2 behavior; remove tests coupled to deleted chess/Python-MCTS/UCI/web APIs. |
| `py/test_helpers/` | Adapt selectively | Typed Go fixtures and native-build markers. |
| `py/tools/` | Delete by default, port selectively | Rebuild only configuration validation, system smoke, throughput, evaluation, and report tools needed by v2. |
| `deployment/lichess/`, `web/` | Delete from active repository at cleanup | Deployment product surfaces are not research-platform requirements. |
| `containers/training.Dockerfile` | Adapt late | Reproducible Go build/runtime image after dependencies stabilize. |
| `documentation/benchmarks/`, historical plans/results | Retain as historical evidence | Mark legacy; do not make them runtime dependencies or evidence for new Go strength. |
| Root and language READMEs | Replace at cleanup | Build, configuration, validation, run, resume, evaluation, and report instructions for v2. |

### 11.4 Essential capability replacement map

| Essential capability today | Authoritative replacement |
| --- | --- |
| Run configuration, approval, source/config hashes | `az/config/models.py`, `resolution.py`, `manifest.py` |
| Hardware and process topology validation | `az/runtime/topology.py` |
| Wall-clock/cost enforcement and shutdown | `az/experiment/runner.py`, `az/runtime/shutdown.py` |
| Native high-throughput self-play/inference | `v2/self_play`, `v2/search`, `v2/inference`, `az/self_play` |
| Credit-based replay ratio and exact reuse | `az/replay/credits.py`, `sampling.py`, `storage.py` |
| Distributed training | `az/training/distributed.py`, `trainer.py` |
| Atomic model/optimizer publication and resume | `az/training/checkpoints.py` |
| Artifact retention | `az/experiment/artifact_retention.py` |
| Model refresh during self-play | focused subscriber in `az/self_play/native_worker.py` |
| Fixed elapsed-time checkpoint/evaluation schedule | `az/experiment/evaluation_schedule.py` |
| Paired evaluation and bootstrap intervals | `az/evaluation/protocol.py`, `statistics.py` |
| Resource, progress, and search diagnostics | `az/runtime/resource_telemetry.py`, native `SearchTelemetry`, reporting |
| Old game settings, model, dataset, and losses | `az/games/go/*` for Go; future games supply their own package |

## 12. Review rules for implementation

Every stage review must verify behavior and dependency direction, not just passing
tests. Reject changes that:

- preserve hidden legacy defaults or mutable shadow configuration;
- make shared replay/training inspect Go fields;
- introduce raw structured dictionaries, dynamic attribute access, or untyped
  binding tuples;
- implement search decisions in Python;
- duplicate random-number ownership or omit provenance;
- combine orchestration, storage, model semantics, and evaluation in one module;
- add a compatibility layer without a scheduled deletion stage; or
- claim research conclusions from correctness, smoke, calibration, or throughput
  pilots.

The final readiness audit traces every resolved configuration field into its
consumer, every replay field into its producer and consumer, every reported
metric to recorded evidence, and every retained operational capability to an
integration or system test.
