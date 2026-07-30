# AlphaZero research platform

This repository is an experiment platform for equal-hardware, equal-wall-clock
AlphaZero-style research. The active implementation is currently Go-only:

- C++20 owns Go rules, state mutation, MCTS, search budgets, stopping, FPU, and
  search telemetry.
- Python owns typed experiment configuration, batched model inference,
  self-play orchestration, replay, training, evaluation, calibration, and
  reporting.
- Go board sizes are configuration values. Every square board of side length at
  least 3 is accepted when its action space fits the native signed 32-bit action
  type.

The initial confirmatory matrix compares fixed, progressive, mixed fast/full,
and adaptive search under common wall-clock and hardware constraints.
Playing-strength claims, Go screening runs, and the final sub-48-hour chess run
are intentionally deferred until separate compute is rented.

The checked-in implementation is currently Go-only, but that is no longer the
target boundary. The controlling rework plan restores chess and the native
multi-game C++ search/LibTorch inference path while retaining Go as the fast
experimental game.

## Build and validate

See [cpp/README.md](cpp/README.md) for the native build and
[py/README.md](py/README.md) for Python configuration and tests.

The controlling architecture and implementation plan is
[documentation/high-throughput-multigame-rework-plan.md](documentation/high-throughput-multigame-rework-plan.md).
The completed Go-first architecture and migration record remains in
[documentation/go-first-rework-architecture.md](documentation/go-first-rework-architecture.md)
as superseded implementation history.
The `PreRework` branch at `f8cb82a` is the authoritative snapshot of the removed
chess, Python-MCTS, UCI, web-play, and deployment implementation.

## Repository layout

```text
cpp/src/          Go rules, search, inference contracts, and Python bindings
cpp/test/         Native correctness and deterministic-search tests
py/src/az/        Experiment platform
py/configs/go/    Go experiment authoring and resolved configurations
py/test/          Go-focused unit and integration tests
documentation/    Current architecture plus explicitly historical evidence
```

Historical chess results and optimization benchmarks remain for provenance.
They are not active runtime dependencies and are not evidence for Go strength.
