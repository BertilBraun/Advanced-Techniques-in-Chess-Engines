# Run control

`deployment/run_control.sh` is the only supported way for humans and agents to start, stop, inspect and archive a
production or test run. It wraps the existing approval/validation flow (`py/run_approved_experiment.py` →
`py/train.py`) in a supervisor program and guarantees evidence preservation on every exit path.

**The rule: a test run without a fetched `.codex-diagnostics` archive did not happen.** Node filesystems are
ephemeral; the failed runs of 18–19 August 2026 were lost because nothing preserved their TensorBoard state. Every
run started through `run_control.sh` is archived on the node whether it completes, is stopped, or crashes, and the
archive only counts as evidence once `fetch` has verified it on the workstation.

## Environment

All paths are overridable; defaults target the standard Vast.ai node layout.

| variable | default | meaning |
|---|---|---|
| `ENGINE_REPOSITORY_DIRECTORY` | script's parent repository | repository root |
| `ENGINE_VIRTUAL_ENVIRONMENT` | `<repository>-venv` | locked venv from `setup_remote.sh` |
| `RUN_CONTROL_ROOT` | `/workspace/run-control` | registry, stop files, per-run logs |
| `RUN_CONTROL_APPROVAL_DIRECTORY` | `/workspace/approvals` | approval JSONs, outside Git |
| `RUN_CONTROL_TENSORBOARD_ROOT` | `/workspace/tensorboard` | curated TensorBoard root the node service watches |
| `RUN_CONTROL_ARCHIVE_ROOT` | `<repository>/.codex-diagnostics` | node-side archive root |
| `RUN_CONTROL_SUPERVISOR_CONF_DIR` | `/etc/supervisor/conf.d` | supervisor program definitions |
| `RUN_CONTROL_STOPWAITSECS` | `600` | supervisor `stopwaitsecs` (≥ 300 s: checkpoint + copy) |
| `RUN_CONTROL_STOP_TIMEOUT_SECONDS` | `1800` | how long `stop` waits for a checkpoint-safe exit |

`fetch` (workstation side) additionally needs `RUN_CONTROL_SSH_DESTINATION` (`root@HOST`), `RUN_CONTROL_SSH_PORT`,
optionally `RUN_CONTROL_SSH_KEY` (a local key path; keys never enter the repository or the node filesystem), and
optionally `RUN_CONTROL_REMOTE_ARCHIVE_ROOT`.

## Approval file

`<approval-directory>/<config-stem>.json` has exactly these five fields; unknown fields are rejected:

```json
{
  "approved_by": "USER_NAME",
  "approved_at_utc": "UTC_TIMESTAMP_WITH_OFFSET",
  "source_revision": "APPROVED_REVISION",
  "configuration_sha256": "RESOLVED_CONFIGURATION_SHA256",
  "maximum_cost": null
}
```

`source_revision`, `configuration_sha256` and `maximum_cost` are compared against the run; the resolved
configuration hash already pins the run name, hardware offer, price and wall-time limit, so those are no longer
carried separately. Approval files written before this change are invalid and must be re-issued.

## Commands

### `run_control.sh start <config.yaml>`

Validates before anything runs: clean checkout (`git status --porcelain` empty), the configuration resolves through
`load_experiment_configuration`, a non-null `training.limits.manual_stop_file`, the approval JSON at
`<approval-directory>/<config-stem>.json` matches `experiment_configuration_sha256` and `HEAD`, and no
stale stop file exists. It then writes the run registry entry (`$RUN_CONTROL_ROOT/runs/<run-name>.env`), installs a
supervisor program named after the run, and starts it via `supervisorctl reread` / `update` / `start`. The deep
validation (hardware contract, runtime image, dependency lock, evaluation artifacts) still happens inside
`py/train.py`; `start` only fails faster.

Expected output (three lines, then the run is supervised):

```
run name:              vast-chess-8gpu-run1
tensorboard directory: /workspace/tensorboard/vast-chess-8gpu-run1
state directory:       /workspace/alphazero-engine/py/training_data/validation/vast-chess-8gpu-run1
```

The supervised command is `run_control.sh supervised-runner <run-name>` (internal): it raises the open-file limit,
exports `TRAINING_RUNTIME_IMAGE`, `LD_LIBRARY_PATH`, `TRAINING_TENSORBOARD_LOG_PATH` and `TRAINING_LOG_PATH`, and
runs the Python runner as a child. It traps `EXIT`, `SIGTERM` and `SIGINT`: a signal touches the manual stop file
(checkpoint-safe stop), and `preserve` runs on every exit — including crashes. The supervisor definition uses
`stopwaitsecs=600`, `autorestart=unexpected` and `killasgroup=true`, so even `supervisorctl stop` leaves either a
trap-preserved or a `stop`-preserved archive.

### `run_control.sh stop <run-name>`

Touches the run's `manual_stop_file`; the coordinator finishes the current quantum, writes a checkpoint, persists
the credit ledger and exits with a `stopped` outcome. `stop` polls the supervisor until the process leaves
`RUNNING` (default timeout 1,800 s), then runs `preserve`. On timeout it still preserves, exits non-zero, and tells
you to escalate with `supervisorctl stop <run-name>` — never Ctrl+C into the process group. The stop file is left in
place as evidence; `start` refuses to reuse it, so remove it deliberately before any restart.

### `run_control.sh status <run-name>`

Prints: the supervisor state line, last generation and optimizer steps (from `credit-ledger.json`), available
credits, the run outcome if terminal, the newest evaluation result (definition, generation, score or accuracy),
inbox/staging depth under `completed-games/`, and per-GPU utilisation/memory from `nvidia-smi`.

### `run_control.sh preserve <run-name>`

Copies, point-in-time, into `<archive-root>/<run-name>-<UTC timestamp>/`:

- `tensorboard/` — the run's TensorBoard directory
- `run/` — `run_manifest.json`, `run_manifests/`, `resolved-experiment.json`, `run-outcome.json`,
  `resource-telemetry.jsonl`, `credit-ledger.json`, `evaluations/`, `checkpoint_*.json` manifests
- `run/search-budget-labels/` and compact replay journals — finalized learned-budget reports, calibration state,
  checksummed shard manifests, cleanup receipts, label-source cohort locators, and replay write-back receipts
- `run/models/` — the latest retained model and optimizer weights for evaluation or resumption
- `logs/` — the run log directory and supervisor stdout/stderr
- `config/` — the authored configuration, the approval JSON, the registry entry

It writes `SHA256SUMS` over every file and self-verifies it. Idempotent: if the newest existing archive for the run
has identical content, no new directory is created. Full replay stores, completed-game payloads, restart states,
and live inbox/staging shards stay on the node. Safe to run against a live run; expect a newer archive later.

### `run_control.sh fetch <run-name> <local-dir>` (workstation)

Rsyncs every `<run-name>-*` archive from the node's archive root into `<local-dir>/.codex-diagnostics/` and runs
`sha256sum --check SHA256SUMS` in each. Fails if verification fails or nothing was fetched. Archives live on the
node until fetched — fetch before the node is destroyed, and only after a verified fetch may the run be reported.

## Preservation guarantees

A run is archived by at least one of three independent paths:

1. the runner's `EXIT`/`SIGTERM`/`SIGINT` trap (crashes, supervisor stop, clean exit),
2. `stop`, which calls `preserve` explicitly after the process exits,
3. a manual `preserve`, safe at any time.

The only unprotected case is `SIGKILL` of the whole group after `stopwaitsecs` expires; the following `stop` or
manual `preserve` still archives everything on disk, so run one before releasing the node.
