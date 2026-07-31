# Repository agent contract

These instructions apply to the entire repository. Follow them for every
change without waiting for the user to repeat them.

## Read before changing code

1. Read `documentation/rework-implementation-handoff.md`.
2. Read the relevant sections of
   `documentation/high-throughput-multigame-rework-plan.md`.
3. Read the nearest component README and any nested `AGENTS.md`.
4. Inspect `git status` and preserve all unrelated or user-owned changes.
5. Inspect the equivalent component at `PreRework` (`f8cb82a`) before replacing
   or restoring it. Use `git show PreRework:<path>`; do not rewrite that
   reference branch.

The rework plan controls architecture. The handoff records implementation
intent and reference behavior. `THINGS_TO_TRY.md` is an idea backlog, not
authorization to implement every item.

## Scope and workflow

- Prefer simple, explicit, maintainable implementations.
- Make the smallest coherent change that advances the current stage.
- Backward compatibility is not required when a cleaner typed interface is
  available.
- Do not modify unrelated code or absorb existing workspace changes.
- Define and prepare a benchmark for a proven `PreRework` hot-path component
  before replacing it; execute both sides before promoting the replacement.
- Restore behavior first, then optimize it in a separate change with evidence.
- Commit each validated feature-sized change before starting the next one.
- Do not begin research ablations before the stable baseline stages pass.
- Until the user explicitly says the target compute environment is available,
  do not run performance benchmarks, training, end-to-end self-play, or full
  evaluation. Prepare their harnesses and frozen configurations. Compilation,
  static checks, unit tests, deterministic fixtures, and very short
  non-performance smoke tests are allowed.

## Non-negotiable architecture

- Go and chess are first-class games selected by typed configuration. One
  experiment configuration is exactly one game.
- C++ owns self-play and evaluation game progression, MCTS, tree storage,
  opponent turns, multi-game scheduling, leaf batching, LibTorch inference,
  subtree reuse, native match results, and replay emission.
- Production self-play never calls Python once per leaf and never uses Python
  MCTS or a Python inference broker.
- Python owns typed configuration, supervision, training/autograd, coarse
  native evaluation-job scheduling, experiment lifecycle, result aggregation,
  and reporting. It never advances evaluation games or handles moves/leaves.
- Game rules, encodings, action maps, models, losses, augmentations, and
  auxiliary targets may be game-specific. Do not force them into a nullable
  universal structure.
- Native self-play writes compact Layer A producer shards quickly. The single
  trainer-side replay maintainer materializes them into large fixed-shape mmap
  Layer B shards while waiting for presentation credits. DDP ranks train only
  from committed Layer B.
- Playing strength from the fixed low-search evaluation ladder is the primary
  progress signal. Training loss and throughput do not replace it.
- Use the `PreRework` evaluation tree, batching, opponents, semantics, and
  reports as references, but keep complete match execution in C++. Generalize
  it for Go and clean responsibility boundaries; do not replace its efficient
  native search concepts without evidence.

The root configuration is a discriminated union of complete
`GoExperimentConfiguration`, `ChessExperimentConfiguration`, and future
per-game variants. Do not model game, model, objective, replay, and evaluation
as independently mixable unions.

## Reference hardware topology

Restore this working `PreRework` topology before experimenting with alternatives:

- four self-play worker processes per GPU;
- four CPU search threads per self-play worker;
- four GPUs, therefore 16 workers and 64 search threads in self-play phases;
- one persistent DDP rank per GPU;
- during optimizer quanta, drain and pause three workers per GPU;
- train with four DDP ranks while one self-play worker remains active per GPU;
- resume all paused workers after the training/publication boundary.

Worker-to-device mapping and paused worker IDs must be explicit typed
configuration. Alternative process counts, GPU splits, consolidation, or
scheduler policies require benchmark evidence against this baseline.

## Native C++ rules

- Target C++20 and use the repository CMake targets.
- `cpp/src/common.hpp` is the project precompiled header. Use it and its
  `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, and `int64`
  aliases. Do not introduce `std::uint8_t`-style names in project code.
- Follow `cpp/.clang-format` and `cpp/.clang-tidy`; do not weaken either to
  make new code pass.
- Use `CamelCase` for classes, structs, and enums; `camelBack` for functions
  and variables; `_camelBack` for private members; and `UPPER_CASE` for global
  constants, as configured by clang-tidy.
- Use the configured warning-as-error targets.
- Regenerate `cpp/compile_commands.json` after adding sources, targets, or
  relevant compiler options so IDE and clang-tidy coverage is complete.
- Search trees use centrally owned, preallocated node and edge arenas with
  typed indices. No per-node owning pointers, child-vector heap allocation, or
  general allocator calls are allowed during search after warm-up.
- Do not add virtual dispatch, filesystem access, Python calls, synchronous
  logging, or avoidable allocation to leaf selection/expansion/backup.
- Use assertions for internal invariants and exceptions for invalid external
  configuration or I/O failures.
- Use project integer aliases in public structs, serialized schemas, and hot
  loops unless a third-party API requires its exact type.
- Prefer focused ownership types and explicit state machines over shared
  ownership.
- Treat CMake, `.clang-format`, `.clang-tidy`, and `.gitignore` as versioned
  project code. Commit intentional changes to them with the feature that needs
  them; keep the generated compile database ignored.

Default maintainability limits:

- C++ function: at most 100 logical lines;
- nesting depth: at most 4;
- cyclomatic complexity: at most 15;
- source file target: at most 500 lines.

A proven hot loop or declarative table may exceed a limit only with a concise
documented reason and dedicated tests.

## Python rules

- Target Python 3.10 or newer and fully annotate every function parameter and
  return type.
- Use precise library types. Do not use `Any`, `object`, raw dictionaries, or
  string keys for structured component data.
- Use dataclasses for internal structures, frozen Pydantic models for
  serialization boundaries, and enums or discriminated unions for variants.
- Do not use `getattr`, `setattr`, `hasattr`, import fallbacks, or silent
  defaults for required configuration.
- Keep imports at module scope.
- Prefer polymorphism or `match` over repeated dynamic type branching.
- Keep public entry points at one abstraction level and extract focused
  validation, persistence, transformation, and reporting components.
- Never put `.item()`, `.cpu()`, synchronous logging, checkpoint
  serialization, directory scans, or Python UUID accumulation in the tight
  optimizer loop.
- Keep production fakes and canned modes out of runtime code; inject test-local
  fakes.

Default maintainability limits:

- Python function: at most 80 logical lines;
- nesting depth: at most 4;
- cyclomatic complexity: at most 15;
- source file target: at most 500 lines.

## Search and concurrency rules

- Restore the `PreRework` cohort-barrier scheduler first.
- Implement root-asynchronous game advancement only as a separate configured
  variant after parity.
- A root may advance only after its budget is met and `in_flight == 0`.
- Pending inference work carries game and root-generation identity.
- Never reroot with outstanding virtual loss or inference.
- Model refresh, pause, replay flush, and shutdown are drain-and-acknowledge
  transitions.
- Per-game random streams must make results independent of scheduler completion
  order.
- Do not trade full inference batches for lower barriers without measuring
  games/hour, positions/hour, batch occupancy, and strength/hour.

## Replay and evaluation rules

- Layer A schemas are game-specific, representation-specific compact
  state/input payloads plus sparse search observations, grouped by completed
  game and optimized for sequential native writes. Workers do not build dense
  action tensors or mmap layout.
- Layer B schemas are resolved with the one-game experiment's representation,
  model, and objective and are optimized for large shuffled mmap reads.
- Layer A and B shards are immutable after atomic publication and include
  identities, source revisions, and checksums. Layer B additionally declares
  shapes, dtypes, offsets, and valid rows.
- A single trainer-control replay maintainer consumes Layer A idempotently
  while waiting for credits. Credits are awarded only for uniquely committed
  eligible Layer B rows.
- Preserve the old replay buffer's deterministic rank sampling, leases,
  logical/physical separation, freshness, bounded capacity, and
  read-amplification telemetry.
- Layer A may be deleted only after its complete Layer B mapping is durable and
  no lease pins it.
- Every elapsed evaluation checkpoint runs the configured low-search ladder:
  MCTS versus random, policy-only versus random, previous/milestone models, and
  chess-only Stockfish opponents.
- The normal ladder uses roughly 100 games per opponent with fixed search,
  alternating colors or paired openings, raw outcomes, W/D/L, score, and
  uncertainty.
- Evaluation jobs may lag but cannot be silently skipped.
- Stockfish configuration and adapters exist only in the chess experiment
  variant.
- Low-budget match evaluation batches across many complete native games. It
  permits at most one in-flight leaf per root and uses no virtual loss.
- Timed interactive/user play is a separate typed search mode. It may use
  multiple in-flight leaves per root and virtual loss, and must drain safely at
  the deadline.
- Do not reuse interactive parallel-search settings for the training-progress
  ladder or implement either mode as Python per-move orchestration.

## Validation

Run only checks relevant to the changed component, but run all of those checks
before declaring the change complete.

For C++ changes, from the repository root:

```powershell
cmake -S .\cpp -B .\cpp\build-clang `
    -DAZ_BUILD_PYTHON=ON `
    -DCMAKE_BUILD_TYPE=RelWithDebInfo `
    -DCMAKE_CXX_COMPILER=clang++
cmake --build .\cpp\build-clang --parallel
ctest --test-dir .\cpp\build-clang --output-on-failure
Copy-Item .\cpp\build-clang\compile_commands.json .\cpp\compile_commands.json
```

Run `clang-format` on changed C++ files and `clang-tidy` using
`cpp/compile_commands.json`. If the relevant environment cannot run a tool,
report that explicitly; do not claim it passed.

For Python changes, format and lint the changed files:

```powershell
ruff format <changed-files>
ruff check --fix <changed-files>
```

Run tests from `py` and retain `--import-mode=importlib`:

```powershell
python -m pytest --import-mode=importlib .\test -q
```

Narrow test selection is acceptable for a focused intermediate commit. Before
recording a stage as structurally complete, run the complete relevant C++ and
Python unit suites. Mark hardware/runtime gates as pending rather than claiming
the stage accepted. Always report commands run, failures, and intentionally
deferred benchmarks or long-running tests.
