# CLAUDE.md — working agreement for this repository

AlphaZero-style engine: Python supervises experiments (`py/`), a C++20 extension is the authoritative runtime for
rules, encoding, batched TorchScript inference and tree search (`cpp/`). Games: chess and Go (7x7, 9x9). Runs are
rented on Vast.ai; the node filesystem is ephemeral. `AGENTS.md` is the older Codex-era instruction file and stays
valid where this file is silent.

## Read before you act

- `documentation/CURRENT-STATE.md` — what the system is today.
- `documentation/plan/chess-recovery-plan-20260820.md` — the current plan; work is referenced by WP number.
- `documentation/plan/chess-post-four-day-regression-analysis-20260820.md` — why the plan exists.
- `documentation/operations/run-control.md` — before starting, stopping or archiving any run.
- `documentation/operations/experiment-platform.md` — before any experiment configuration, provisioning, queue,
  monitoring or export work (read its supersession banners first).
- `documentation/architecture/python-runtime-rework.md` — before architectural changes.
- `cpp/AGENTS.md` and `cpp/README.md` — before touching native code.

## Authority and handoffs

The user owns approvals, launches, stops and phase acceptance. Agents prepare, validate and report. Never start,
stop or reconfigure a production or test run without an explicit instruction; never spend GPU time the user has not
authorised. At every handoff report: completed work, outstanding work, commits (hashes and subjects), validation run
and its result, changes that need special review, unresolved decisions. Do not describe work as done that was not
validated.

## Evidence rules

- A test run that has no fetched archive under `.codex-diagnostics/` did not happen. Use `deployment/run_control.sh`
  (start / stop / status / preserve / fetch) for every run.
- Configurations are resolved and hashed (`experiment_configuration_sha256`); record the SHA with every measurement.
- Benchmarks go under `documentation/benchmarks/<topic>-<hardware>-<date>/README.md` with config SHA, source SHA, node
  and raw numbers. Do not compare numbers across different hardware without saying so.
- The per-generation yardstick in the recovery plan is the pass/fail reference for new runs.

## Python conventions

- Python 3.12, `from __future__ import annotations`, full type hints, `@dataclass(frozen=True)` for values, pydantic
  models for configuration with **no implicit defaults** (a missing key is an error; the only defaults are the ones
  listed in the plan). Configuration files use `extends`; lists replace wholesale.
- `ruff` is the formatter and linter (`ruff.toml`: line length 120, single quotes). Run `ruff format` and
  `ruff check` on touched files before committing.
- Atomic file writes only through `src/util/atomic_file.py`. Logging through `src/util/log.py`.
- Tests live in `py/test`, one module per subject, fixtures under `py/test/fixtures`. Run from `py`:
  `python -m pytest --import-mode=importlib ./test -q` (always keep `--import-mode=importlib`). Tests that need the
  native extension or CUDA are marked and skipped when unavailable; never make a unit test depend on a network.
- No Python MCTS, no second search implementation: search, rules and encoding are native.

## Native conventions

- C++20, `clang-format` file style, `cpp/run-clang-tidy.sh` before large changes. Routine compile check via the
  `CompileCheck` build type in a persistent build directory with ccache (see `cpp/AGENTS.md`); Release build for
  anything deployed or measured.
- All native tests belong to the single `NativeTests` executable (`test/TestRunner.hpp`, `test/TestMain.cpp`);
  no per-suite executables. Binding contract changes need a Python-side test too.
- Encoding and action-id changes require the colour-symmetry harness
  (`cpp/test/flip-harness/`) to pass again.

## Commits and branches

- Feature-sized commits after validation; imperative subject ≤ 60 characters in the existing style ("Add …",
  "Fix …", "Bound …"); body states what was validated. Do not mix unrelated changes; preserve unrelated local
  changes you find in the tree.
- One branch per work package (`wp1-heads`, `wp2-ingestion`, `wp8-run-control`, …), rebased on `master`, merged
  only after its acceptance criterion in the plan is met.
- Pushing to `origin` is fine, `master` included; nodes fetch what they run, so unpushed work cannot be deployed.
  Push validated commits, not a broken tree, and never force-push a branch someone else may have based on.

## Remote nodes

- Run every command on a node through `deployment/remote_command.sh <HOST[:PORT]> <command …>`; never
  hand-assemble `ssh`. The script owns the key, user and connection options, takes the destination from
  `documentation/operations/current-node.md`, and returns the remote exit status (255 = connection failed).
  Runs are still started, stopped and archived only through `deployment/run_control.sh`.
- Provision with `deployment/setup_remote.sh` (clones the revision, installs the locked dependencies, builds the
  Release extension, smokes the engines). KataGo must be the CUDA build; never the Eigen fallback.
- Record `nvidia-smi`, driver, GPU model/count, effective CPUs, RAM, disk and the locked PyTorch CUDA/cuDNN runtime
  in the provisioning note. Read `/etc/vast-agents-guide.md` on the node before changing it.
- SSH keys are local-only and never enter the repository or the node's filesystem.
- Copy evidence off the node before it is destroyed; nothing on a node is durable.

## Style of communication

Short, specific, numbers with units and sources. Say what was verified and how. When something is uncertain, say
so with the reason. No filler, no restating the request.
