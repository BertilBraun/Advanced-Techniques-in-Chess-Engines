# Multi-game AlphaZero rework handoff

Status: superseded clean-slate implementation handoff

The active direction is now the `PreRework`-based integration described in
`documentation/high-throughput-multigame-rework-plan.md`. This document remains
historical context for the abandoned Go-first clean-slate path and must not be
used as the current implementation sequence.

Reference snapshot: `PreRework` / `f8cb82a`

Controlling plan: `documentation/high-throughput-multigame-rework-plan.md`

Runtime status: target-hardware benchmarks, end-to-end runs, training, and
full evaluation are deferred until the user confirms compute access.

## 1. Mission

Build the strongest from-scratch AlphaZero-style engine obtainable within a
fixed modest hardware and wall-clock budget.

Use 7x7 Go for rapid screening, selected 9x9 Go for scale confirmation, and
then transfer demonstrated improvements to chess. The final target is a
from-scratch chess run on four RTX 4090 GPUs, 120 GiB host RAM, and 128 CPU
cores for less than 48 hours.

Go supports every square board size of at least 3 that fits the signed native
action representation. A resolved Go session uses one fixed board size; 7x7
and 9x9 are experiment profiles, not implementation limits.

The immediate task is restoration and architectural cleanup, not a research
feature sprint.

## 2. Documents and authority

Read in this order:

1. repository `AGENTS.md` for mandatory implementation behavior;
2. this handoff for context and reference behavior;
3. `high-throughput-multigame-rework-plan.md` for architecture and stages;
4. component READMEs for build and local details;
5. `THINGS_TO_TRY.md` only when promoting a feature into the registry.

If implementation pressure conflicts with an architectural invariant, stop
and update the plan explicitly rather than quietly working around it.

## 3. What went wrong

The Go-first rewrite retained several useful experiment concepts but removed
proven high-throughput machinery and chess:

- multi-process, multi-threaded native self-play;
- native LibTorch inference and batching;
- centralized bounded tree storage and subtree reuse;
- the mature rolling replay lifecycle;
- the established playing-evaluation suite;
- four-GPU phased resource coordination.

The replacement also accumulated mixed responsibilities, long functions, deep
nesting, Python orchestration in hot paths, and per-step trainer overhead. The
rework restores proven behavior selectively and places it behind game-agnostic
infrastructure.

Do not wholesale revert to `PreRework`: it contains useful concepts alongside
Python MCTS, chess-specific globals, HDF5 coupling, and maintainability debt.

## 4. Reference components to inspect

When restoring or replacing a proven component, inspect these reference
components:

| Concern | `PreRework` reference |
|---|---|
| bounded tree slots, generations, rerooting | `cpp/src/MCTS/SearchTree.*` |
| multi-root batched search | `cpp/src/MCTS/DirectSelfPlaySearch.*` |
| LibTorch inference buffers and pipelines | `cpp/src/DirectInference.*` |
| native search facade used by self-play | `py/src/self_play/SelfPlayCpp.py` and native bindings |
| process supervision and pause/resume | `py/src/cluster/CommanderProcess.py` |
| per-worker lifecycle | `py/src/cluster/SelfPlayProcess.py` |
| persistent DDP training | `py/src/train/DistributedTraining.py` |
| rolling replay | `py/src/train/RollingReplayBuffer.py` |
| reanalysis overlays | `py/src/train/ReplayReanalysis.py` |
| evaluation ladder | `py/src/cluster/EvaluationProcess.py` |
| efficient evaluation search | `cpp/src/MCTS/EvalMCTS*`, `cpp/src/MCTS/EvalSearchTree.*` |
| legacy match wrapper/provenance | `py/src/eval/ModelEvaluationCpp.py` |
| chess rules and history | `cpp/src/Board.*`, pinned Stockfish dependency and patch |
| chess encoding and action mapping | `cpp/src/BoardEncoding.*`, `cpp/src/MoveEncoding.*` |

Use `git show PreRework:<path>` to inspect the snapshot without disturbing the
current worktree. Reference comparison harnesses and paired execution are
postponed for the current foundation phase and are not prerequisites for new
contracts, build structure, or game abstractions.

## 5. Required production topology

The initial implementation reproduces the known-good four-GPU topology.

### Self-play-dominant phase

- 4 GPUs.
- 4 self-play worker processes per GPU.
- 4 CPU search threads per worker.
- 16 active workers and 64 search threads total.
- Each worker owns its model replica, inference pipeline, active games, tree
  arenas, replay output, and health state.
- Four persistent DDP ranks remain idle.

### Optimizer phase

- Drain and pause three self-play workers per GPU.
- Leave one self-play worker active per GPU.
- Run one persistent DDP rank on each GPU.
- Total: four active self-play workers, four DDP ranks, and 16 search threads.
- Publish the next model generation through an atomic prepare/warm/validate
  and acknowledgement sequence.
- Resume all paused workers.

The supervisor owns the transition. A pause is complete only after the worker
has stopped accepting new roots, drained inference, removed virtual loss,
sealed or safely retained replay state, and acknowledged. Failure handling
must restore the intended worker count or terminate clearly.

Do not consolidate four processes into one device process merely because it
looks cleaner. Model duplication, CUDA context overhead, batching, isolation,
and contention are empirical tradeoffs. Restore and benchmark the working
topology first.

## 6. Search scheduling

### Reference baseline: cohort barrier

`PreRework::DirectSelfPlaySearch` schedules leaves across many roots and keeps
multiple inference batches outstanding. It is already asynchronous at the
leaf/inference level. Its external `search(boards)` operation, however, returns
only when every root in the cohort is complete.

That behavior is the first restoration target because it is understood and was
fast in practice. Capture its throughput, batch histogram, selection/wait
timing, memory, and deterministic fixtures before changing lifecycle semantics.

### Candidate: root-asynchronous advancement

Mixed search budgets expose a barrier cost: a completed fast root waits while
slower roots in its cohort finish. The candidate scheduler makes each game a
native state machine:

```text
ready root
  -> selecting
  -> inference outstanding
  -> budget met
  -> drain in-flight leaves
  -> freeze target and select move
  -> apply move and reroot
  -> next ready root or terminal game
```

Required invariants:

- do not select more work once completed plus reserved in-flight work reaches
  the resolved limit;
- do not advance while `in_flight != 0`;
- do not reroot with virtual loss outstanding;
- tag pending leaves with game ID and root-generation token;
- reject stale completion as an internal invariant failure;
- keep per-game randomness independent of completion order;
- keep replay ordering separate from search completion ordering;
- drain all roots at pause, model refresh, and shutdown boundaries;
- keep one model generation per root search.

This complexity belongs in a focused native scheduler, not in Python workers
or callbacks. Implement it as a typed scheduler variant after parity and
benchmark it against the cohort barrier with identical games and budgets.

Promotion requires better games/hour, positions/hour, or strength/hour without
lower target quality, pathological partial batches, nondeterminism, or
lifecycle failures.

## 7. Ownership boundaries

### Generic native infrastructure

- arena-backed search tree;
- search-policy composition;
- multi-game scheduler;
- native inference batching and model generation lifecycle;
- complete evaluation match runner and opponent adapters;
- batched-match scheduling on the shared MCTS engine;
- worker pause/drain/resume;
- replay shard container and publication;
- aggregate telemetry.

### Game-specific native code

- state and history;
- rules and terminal semantics;
- legal actions;
- action/policy mapping;
- network input encoding;
- symmetry declarations;
- typed replay schema fields.

### Generic Python infrastructure

- frozen typed configuration and resolution;
- process supervision;
- single Layer A intake and Layer B materialization service;
- persistent DDP loop;
- replay catalog and deterministic sampling;
- checkpoints and model publication;
- coarse native evaluation-job scheduling and report aggregation;
- experiment manifests and reports.

### Game-specific Python code

- model construction;
- batch type;
- primary and auxiliary losses;
- valid augmentation;
- artifact export;
- game-specific evaluation opponents or diagnostics.

## 8. One-game configuration boundary

One serialized experiment configuration describes exactly one game. Model this
as a discriminated union of complete variants:

- `GoExperimentConfiguration` contains Go rules/board, Go model and objective,
  Go Layer A/B schemas, self-play, training, and Go evaluation.
- `ChessExperimentConfiguration` contains chess rules/history, chess model and
  objective, chess Layer A/B schemas, self-play, training, and chess
  evaluation.

Shared topology, optimizer, search-policy, and telemetry structures may be
reused inside the variants. Do not expose independent unions that allow a Go
game to select a chess model, loss, replay schema, or Stockfish evaluator.

The game literal is the serialization discriminator. Every worker, trainer
rank, replay manifest, checkpoint, and evaluation task in a run carries and
validates the same game identity. Resume fails before launch on a mismatch.

Each game variant has a constrained native match-search configuration.
Low-budget matches use the same general batched MCTS implementation as
self-play while structurally disabling virtual loss and intra-root parallel
leaves. Timed interactive search and its deadline, parallel-leaf, and
virtual-loss settings are deferred.

## 9. Two-layer replay direction

### Layer A: fast native producer spool

Self-play workers always generate compact Layer A shards. They group samples
by completed game and store the compact game-specific state/input payload
already available to native inference plus sparse variable-length root search
observations. They do not construct dense action tensors or mmap layout. A
background writer batches several completed games, performs sequential atomic
publication, and returns to search quickly.

Layer A contains terminal metadata, model and environment identity, and
per-ply compact state, priors, visits, child Q, selected/played action, budget,
search type, weights, and diagnostics. Every fast and full search is retained.
It is representation-specific to the current one-game run rather than a
universal archive. Full initial-state/action reconstruction data is included
only when a named feature needs it.

The writer queue is bounded. Backpressure stops admission of new games rather
than blocking leaf completion or dropping samples. Layer A shards remain
immutable until acknowledged as fully materialized.

### Layer B: large trainer-ready mmap shards

The trainer control process owns exactly one replay maintainer. While it waits
for enough presentation credits, that maintainer:

1. discovers Layer A through incremental manifests;
2. claims each producer shard idempotently;
3. decodes compact state payloads;
4. derives dense policy/value/auxiliary targets and deterministic
   augmentation;
5. fills large fixed-shape columnar mmap shards;
6. atomically commits a Layer B manifest and source-to-row ledger;
7. awards credits for newly committed eligible rows;
8. acknowledges Layer A retention or deletion.

DDP ranks do not materialize data. They read only committed Layer B through
deterministic rank-partitioned sampling and persistent pinned batches.

Crashes cannot expose partial shards or duplicate credits. Layer A deletion is
allowed only after every contained game maps durably to Layer B and no
diagnostic/reanalysis lease pins it.

Preserve from the old replay design:

- atomic manifests and checksums;
- logical versus physical segment identity;
- deterministic global/rank partitioning;
- batch leases and safe eviction;
- bounded capacity and freshness metrics;
- vectorized reads and read-amplification accounting;
- immutable reanalysis sidecars when reanalysis is enabled.

A different representation or objective still receives a new end-to-end run.
The two layers decouple producer latency from trainer I/O; they are not a reason
to reuse fixed self-play data for ordinary ablations.

## 10. Native evaluation with shared batched MCTS

The existing `PreRework` evaluation tree, batching, opponents, match semantics,
and reports are references. Do not preserve its Python ownership boundary.
The target native evaluation executable owns complete games:

- game states, legal moves, terminal decisions, and both players' turns;
- evaluation trees, leaf selection, batching, inference, and move choice;
- random and model opponents;
- external Stockfish UCI communication for chess;
- paired colors/openings and raw game/match results.

Python resolves and submits coarse evaluation jobs, assigns devices, monitors
processes, rotates historical checkpoints, and aggregates completed native
reports. It never advances a game or handles a move/leaf callback.

Cleanup should strengthen types without changing reference match semantics or
replacing the efficient native search. Add Go state/action support at the
native match boundary. Stockfish and chess opening settings remain present only
in `ChessExperimentConfiguration`.

### Low-budget training and offline matches

Run roughly 100 games concurrently. Each root has at most one outstanding leaf
and uses no virtual loss. Sequential MCTS semantics are preserved within each
tree while inference batches are filled across independent games. Completed
visits exactly match the fixed budget; root noise and self-play temperature
are disabled.

At tiny visit budgets, intra-root virtual loss would consume a significant
fraction of the search and bias results. One hundred independent games already
provide sufficient batching parallelism.

Timed interactive/user play, UCI, deadline scheduling, intra-root parallel
leaves, and virtual-loss tuning are deferred. The current implementation uses
one general native batched MCTS/tree/inference implementation for self-play and
evaluation; match evaluation constrains its scheduling rather than introducing
a second search implementation.

At fixed elapsed checkpoints run the low-search progress ladder. The default
plan uses roughly 100 games per opponent under a frozen small search budget:

- current MCTS model versus seeded legal random;
- current policy only versus random;
- current versus the preceding evaluated checkpoint;
- current versus configured milestone/anchor checkpoints;
- chess only: pinned Stockfish skill levels, initially 0-3;
- chess only: a pinned fixed-node Stockfish condition.

Use alternating colors or paired openings. Retain raw game outcomes, seeds,
opening identities, model hashes, engine binary/options, search limits,
terminal reasons, W/D/L, score, and uncertainty intervals.

These matches are the main progress and strength indicator. Loss curves,
samples/second, GPU utilization, and games/hour explain results but cannot
claim playing improvement on their own. Larger common-search paired matches
remain the confirmatory publication protocol.

Evaluation is asynchronous and resource-budgeted. It may lag, but every due
checkpoint stays queued and visible until completed or explicitly failed.

## 11. Implementation sequence

Do not skip directly to feature experiments.

1. Establish the architecture/build/game/config foundation: generic game,
   artifact, session, topology, and replay contracts plus complete one-game Go
   and chess configuration variants.
2. Restore chess rules, history, encoding, and tests within that foundation.
3. Restore generalized native LibTorch inference.
4. Restore arena-backed cohort MCTS and the 4x4 worker topology.
5. Restore fast Layer A producer shards, trainer-side Layer B materialization,
   and the persistent trainer hot path.
6. Restore the complete native progress-evaluation ladder using the shared
   batched search engine.

Each item is a sequence of coherent reviewed commits, not one large rewrite
commit.

Stable baselines, root-asynchronous scheduling, search/training ablations,
timed interactive play, UCI, chess transfer pilots, and final training are
deferred beyond the current implementation scope.

### Current structural status

- Deferred reference benchmarking: intentionally postponed.
- Joint foundation and chess restoration: native contracts, a typed
  preallocated arena, split core/search/Go/chess targets, complete Go/chess
  configuration branches, and the initial Stockfish-backed chess
  rules/encoding/policy restoration are implemented. Arena integration into
  search and broader chess differential parity remain.
- Native LibTorch inference and generic native self-play: not started; the
  predecessor still uses a Python inference callback and Python game lifecycle.
- Two-layer replay: not started; the predecessor replay remains a useful
  checksummed storage and sampling reference.
- Trainer/runtime: explicit four-process-per-GPU, four-thread-per-process
  assignments and optimizer active/paused worker IDs are resolved. Layer B
  consumption and the drain/pause/acknowledge transition itself remain.
- Experiment lifecycle/evaluation: partial predecessor foundation; complete
  native matches, chess opponents, and coarse Python job ownership remain.
- Stage 8 and later: deferred.

## 12. Definition of done for restored infrastructure

A restored component is not complete merely because it compiles.

- It has parity fixtures against the relevant reference behavior.
- Hot-path benchmark harnesses and frozen inputs exist; measurements are
  captured before promotion when the target hardware becomes available.
- Both Go and chess fit the contract where the stage requires them.
- C++ formatting, clang-tidy, warnings, compilation, and tests pass.
- Python formatting, lint, typing conventions, and relevant tests pass.
- The native compile database includes all added sources and targets.
- Pause/resume, refresh, failure, and shutdown behavior are tested.
- Telemetry exposes throughput and backpressure rather than hiding them.
- No long compute claim is made from a smoke test.
- The coherent change is committed without unrelated workspace files.

Until runtime access returns, a component may be labeled only `structurally
complete; runtime validation pending`. It cannot replace or delete its
reference path on the strength of unexecuted benchmarks.

## 13. Known decisions still requiring measurement

Do not silently decide these during an unrelated implementation:

- cohort-barrier versus root-asynchronous default;
- active games and tree-arena capacity per worker;
- inference batch size, outstanding slots, and CUDA streams;
- four model replicas per GPU versus later process consolidation;
- optimizer quantum duration and self-play/training duty cycle;
- Layer A shard size/compression and Layer B row/shard layout/dtype tradeoffs;
- CPU affinity and NUMA placement;
- exact low-search evaluation budget after reference parity.
- timed-interactive leaf parallelism, virtual-loss magnitude, and batch wait
  after interactive play enters scope.

Resolve each with a typed configuration, benchmark, and recorded decision.

## 14. Deferred execution period

For the next few days, prepare code for later execution:

- compile and run static analysis;
- add focused unit and deterministic parity tests;
- define schemas, manifests, state machines, and failure recovery;
- create benchmark executables/scripts and frozen configurations;
- record exact commands, expected artifacts, and acceptance thresholds.

Do not execute or claim target-hardware benchmarks, training runs, complete
self-play integrations, full evaluation suites, or playing-strength results
until the user confirms the compute environment is available.

## 15. Starting a new implementation session

1. Read `AGENTS.md`, this handoff, and the relevant plan stage.
2. Run `git status`; identify and preserve unrelated changes.
3. Inspect `PreRework` only when restoring or replacing the relevant proven
   component.
4. State the smallest stage-scoped objective.
5. Capture the relevant structural/parity fixture; prepare performance
   comparison commands when restoring a proven hot-path component.
6. Implement without mixing a restoration and a speculative optimization.
7. Format, lint, compile, test, and run only a short smoke if needed.
8. Regenerate `cpp/compile_commands.json` for native target changes.
9. Record validation and performance evidence.
10. Commit the coherent change.
