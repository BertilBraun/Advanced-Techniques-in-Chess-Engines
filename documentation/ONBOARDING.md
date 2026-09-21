# Onboarding

Read-in-this-order path for a new contributor or agent.

## Read first

1. `CLAUDE.md` (5 min) — the working agreement: authority, evidence rules, conventions.
2. [`CURRENT-STATE.md`](CURRENT-STATE.md) (5 min) — what the system is today.
3. [`chess-final-config.yaml`](../py/configs/production/chess-final-config.yaml) (20 min) — the complete readable
   recipe for the final chess run. Published results additionally pin an exact revision and resolved configuration.
4. [`py/README.md`](../py/README.md) (20 min) — entry points, setup, validation.
5. [`cpp/README.md`](../cpp/README.md) + [`cpp/AGENTS.md`](../cpp/AGENTS.md) (20 min) — native build and
   runtime boundary.
6. [Experiment platform](operations/experiment-platform.md) (30 min) — with its supersession banners in mind.
7. [Experiment catalog](experiments/README.md) and [technical report](report/README.md) — research decisions and
   their evidence. The dated [chess recovery plan](plan/chess-recovery-plan-20260820.md) is historical context, not
   current operating authority.

## Build and validate

Python validation, from `py/`:

```powershell
uv run ruff format
uv run ruff check --fix
python -m pytest --import-mode=importlib .\test -q
```

Always keep `--import-mode=importlib`. Tests that need the native extension or CUDA are marked and skip when
unavailable; real Stockfish/KataGo smoke tests are opt-in and need provisioned external artifacts. Run
`ruff format` and `ruff check` on touched files before committing, with all warnings resolved.

Native build and tests, from the repository root:

```powershell
cmake -S .\cpp -B .\cpp\build -DCMAKE_BUILD_TYPE=Release
cmake --build .\cpp\build --parallel
ctest --test-dir .\cpp\build --output-on-failure
```

Native-facing Python tests require the freshly built extension. For routine compile checks use the
`CompileCheck` build type in a persistent build directory with ccache; Release is required for anything
deployed or measured. All native tests run through the single `NativeTests` executable (see `cpp/AGENTS.md`).

## Provisioning a training node

[`deployment/setup_remote.sh`](../deployment/setup_remote.sh) is the authoritative bootstrap for a fresh
training node. It clones the requested revision, installs the hashed training environment, builds the Release
extension, exports `ENGINE_SOURCE_REVISION`, and executes the supplied command. Set `ENGINE_REPOSITORY_REF`,
`ENGINE_REPOSITORY_DIRECTORY`, `ENGINE_VIRTUAL_ENVIRONMENT` or `ENGINE_REPOSITORY_URL` to override the
checkout and environment locations. The script intentionally does not install Stockfish, KataGo, or their
model and configuration artifacts — see [evaluation engines](operations/evaluation-engines.md).

After bootstrap, runs are started, stopped, inspected and archived exclusively through
[`deployment/run_control.sh`](../deployment/run_control.sh) — see [run control](operations/run-control.md).

The checked-in `py/configs/*-experiment-template.yaml` files are validation templates, not approved production
runs: hardware, artifact paths and hashes, output paths, source revision and approval must be resolved
explicitly before a run.

## Playing against a trained model

The native interactive chess engine is shared by both deployments and is retained production code:
[web play](operations/web-play.md) uses the typed FastAPI backend and browser client (deployed at
[chess.bertil-braun.de](https://chess.bertil-braun.de)), and the
[Lichess/Vast path](../deployment/lichess/README.md) invokes `python -m src.games.chess.uci` through the
checked-in UCI launcher.

## What you may and may not do

The user owns approvals, launches, stops and phase acceptance. Agents prepare, validate and report. Never
start, stop or reconfigure a run without explicit instruction; never spend GPU time the user has not
authorised. Do not push to `master`; one branch per work unit.

## Where evidence goes

- Measurements: `documentation/benchmarks/<topic>-<hardware>-<date>/README.md` with config SHA, full source
  SHA, node, and raw numbers.
- Run and node records: dated files (see `operations/README.md`); fetch archives via
  `deployment/run_control.sh` before a node is released — nothing on a node is durable.
- Configurations are resolved and hashed (`experiment_configuration_sha256`); record the SHA with every
  measurement.

## Conventions in ten lines

- Use Python 3.12.
- Add `from __future__ import annotations`.
- Fully annotate functions and values.
- Use frozen dataclasses for value types.
- Define Pydantic configuration models without implicit defaults.
- Make atomic writes through `src/util/atomic_file.py`.
- Use `src/util/log.py` for logging.
- Keep MCTS, rules, and encoding native; do not add a Python MCTS path.
- Use C++20, clang-format style, and the single `NativeTests` executable.
- Comment only where a deliberate choice or boundary is non-obvious, and use imperative commit subjects of at
  most 60 characters.

## Glossary

- **WP** — work package in the recovery plan.
- **Generation** — 500 optimizer steps at batch 2048.
- **Quantum** — one funded training slice between checkpoint publishes.
- **Credit** — replay-sample budget that funds a quantum at the configured replay ratio.
- **Replay ratio** — samples ingested per sample trained on.
- **Yardstick** — the per-generation pass/fail table from the four-day run.
- **r3/r4** — the four-day run's config revisions (tag `four-day-baseline`).
- **Freeze** — the archived evidence bundle of that run.
- **`extends`** — configuration inheritance; lists replace wholesale.
- **Progressive sizing** — staged model growth during a run
  ([architecture guide](architecture/progressive-model-sizing.md)).
