# Onboarding

Read-in-this-order path for a new contributor or agent.

## Read first

1. `CLAUDE.md` (5 min) — the working agreement: authority, evidence rules, conventions.
2. [`CURRENT-STATE.md`](CURRENT-STATE.md) (5 min) — what the system is today.
3. [Chess recovery plan](plan/chess-recovery-plan-20260820.md) §0–§2 (30 min) — the current plan and
   yardstick; skim the [regression analysis](plan/chess-post-four-day-regression-analysis-20260820.md) for
   why it exists.
4. [`py/README.md`](../py/README.md) (20 min) — entry points, setup, validation.
5. [`cpp/README.md`](../cpp/README.md) + [`cpp/AGENTS.md`](../cpp/AGENTS.md) (20 min) — native build and
   runtime boundary.
6. [Experiment platform](operations/experiment-platform.md) (30 min) — with its supersession banners in mind.

## Build and validate

- Python (from `py/`): `python -m pytest --import-mode=importlib ./test -q` — always keep
  `--import-mode=importlib`. Tests needing the native extension or CUDA are marked and skip when unavailable.
- Lint: `ruff format` and `ruff check` on touched files before committing; all warnings resolved.
- Native: CompileCheck build in a persistent build directory with ccache for routine checks; Release for
  anything deployed or measured; all native tests run through the single `NativeTests` executable
  (see `cpp/AGENTS.md`).

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

Python 3.12 · `from __future__ import annotations` · full type hints · frozen dataclasses for values ·
pydantic configs with no implicit defaults · atomic writes via `src/util/atomic_file.py` · logging via
`src/util/log.py` · no Python MCTS (search/rules/encoding are native) · C++20, clang-format style, single
`NativeTests` executable · comments only where a deliberate choice or boundary is non-obvious — one short
line explaining the why · commit subjects imperative, ≤ 60 chars.

## Glossary

**WP** — work package in the recovery plan. **Generation** — 500 optimizer steps at batch 2048.
**Quantum** — one funded training slice between checkpoint publishes. **Credit** — replay-sample budget that
funds a quantum at the configured replay ratio. **Replay ratio** — samples ingested per sample trained on.
**Yardstick** — the per-generation pass/fail table from the four-day run. **r3/r4** — the four-day run's
config revisions (tag `four-day-baseline`). **Freeze** — the archived evidence bundle of that run.
**`extends`** — config inheritance; lists replace wholesale. **Progressive sizing** — staged model growth
during a run (`architecture/progressive-model-sizing.md`).
