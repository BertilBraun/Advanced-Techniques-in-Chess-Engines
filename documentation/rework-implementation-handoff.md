# Multi-game AlphaZero rework handoff

Status: architecture and implementation planning

Reference snapshot: `PreRework` / `f8cb82a`

Controlling plan: `documentation/high-throughput-multigame-rework-plan.md`

## 1. Mission

Build the strongest from-scratch AlphaZero-style engine obtainable within a
fixed modest hardware and wall-clock budget.

Use 7x7 Go for rapid screening, selected 9x9 Go for scale confirmation, and
then transfer demonstrated improvements to chess. The final target is a
from-scratch chess run on four RTX 4090 GPUs, 120 GiB host RAM, and 128 CPU
cores for less than 48 hours.

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

Before implementing a replacement, inspect and benchmark these reference
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
| chess rules and history | `cpp/src/Board.*`, pinned Stockfish dependency and patch |
| chess encoding and action mapping | `cpp/src/BoardEncoding.*`, `cpp/src/MoveEncoding.*` |

Use `git show PreRework:<path>` to inspect the snapshot without disturbing the
current worktree.

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
- persistent DDP loop;
- replay catalog and deterministic sampling;
- checkpoints and model publication;
- evaluation scheduling and result storage;
- experiment manifests and reports.

### Game-specific Python code

- model construction;
- batch type;
- primary and auxiliary losses;
- valid augmentation;
- artifact export;
- game-specific evaluation opponents or diagnostics.

## 8. Replay direction

The production baseline has one durable training path:

1. native self-play keeps an unfinished game's training observations in
   bounded memory;
2. terminal completion supplies outcome targets;
3. the worker writes exact fixed-shape rows into immutable mmap shards;
4. atomic manifests expose sealed shards to the rolling replay catalog;
5. deterministic samplers group reads by shard and feed persistent pinned
   batches.

The replay schema is resolved with the game, representation, model heads, and
objective. A different representation or auxiliary objective gets a new
end-to-end run and a different schema. Do not build a mandatory canonical
trajectory/materialization pipeline.

Optional logical trajectory capture may be introduced only for a named
consumer such as adaptive-stop calibration, difficult-state mining, restart
states, or reanalysis. Measure its cost and keep it disabled otherwise.

Preserve from the old replay design:

- atomic manifests and checksums;
- logical versus physical segment identity;
- deterministic global/rank partitioning;
- batch leases and safe eviction;
- bounded capacity and freshness metrics;
- vectorized reads and read-amplification accounting;
- immutable reanalysis sidecars when reanalysis is enabled.

## 9. Evaluation is not optional

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

## 10. Implementation sequence

Do not skip directly to feature experiments.

1. Freeze reference benchmarks and parity fixtures.
2. Define generic game, artifact, session, topology, and replay contracts.
3. Restore chess rules, history, encoding, and tests.
4. Restore generalized native LibTorch inference.
5. Restore arena-backed cohort MCTS and the 4x4 worker topology.
6. Restore direct mmap replay and the persistent trainer hot path.
7. Restore the complete progress-evaluation ladder.
8. Establish stable 7x7, 9x9, and short chess baselines.
9. Implement root-asynchronous scheduling as an isolated measured variant.
10. Begin the fixed/progressive/mixed/adaptive search ablations.

Each item is a sequence of coherent reviewed commits, not one large rewrite
commit.

## 11. Definition of done for restored infrastructure

A restored component is not complete merely because it compiles.

- It has parity fixtures against the relevant reference behavior.
- Hot-path benchmarks were captured before and after.
- Both Go and chess fit the contract where the stage requires them.
- C++ formatting, clang-tidy, warnings, compilation, and tests pass.
- Python formatting, lint, typing conventions, and relevant tests pass.
- The native compile database includes all added sources and targets.
- Pause/resume, refresh, failure, and shutdown behavior are tested.
- Telemetry exposes throughput and backpressure rather than hiding them.
- No long compute claim is made from a smoke test.
- The coherent change is committed without unrelated workspace files.

## 12. Known decisions still requiring measurement

Do not silently decide these during an unrelated implementation:

- cohort-barrier versus root-asynchronous default;
- active games and tree-arena capacity per worker;
- inference batch size, outstanding slots, and CUDA streams;
- four model replicas per GPU versus later process consolidation;
- optimizer quantum duration and self-play/training duty cycle;
- exact replay row/shard layout and dtype tradeoffs;
- CPU affinity and NUMA placement;
- exact low-search evaluation budget after reference parity.

Resolve each with a typed configuration, benchmark, and recorded decision.

## 13. Starting a new implementation session

1. Read `AGENTS.md`, this handoff, and the relevant plan stage.
2. Run `git status`; identify and preserve unrelated changes.
3. Inspect the `PreRework` component and existing benchmark evidence.
4. State the smallest stage-scoped objective.
5. Capture or identify the parity fixture and performance baseline.
6. Implement without mixing a restoration and a speculative optimization.
7. Format, lint, compile, test, and run only a short smoke if needed.
8. Regenerate `cpp/compile_commands.json` for native target changes.
9. Record validation and performance evidence.
10. Commit the coherent change.
