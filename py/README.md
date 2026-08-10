# Python runtime

Python owns experiment configuration, run preparation, coordinator lifecycle, replay ingestion, DDP training,
evaluation scheduling, and reporting for chess and Go. Native C++ owns game rules, state transitions, network input
encoding, and batched search.

## Production entry points

- `python py/train.py --run-config ... --expected-source-revision ... --approval-file ...` starts an explicitly
  approved training run from the repository root.
- `python py/queue_experiments.py run --queue-config ...` validates and runs an ordered experiment queue on Linux.
- `python py/queue_experiments.py status --summary ...` validates and prints its durable JSON summary.
- `python -m src.games.chess.uci --model ...` runs the native interactive chess engine behind the UCI protocol from
  `py`.
- `deployment/web/backend` uses the same native interactive chess engine for browser play.

The UCI and interactive packages are deployment-owned production code. They are independent of the removed Python
chess rules and training implementations.

## Environment and build

`deployment/setup_remote.sh` is the authoritative fresh-node setup path. It clones the requested revision, installs
the hashed environment from `requirements-training.lock`, builds the Release extension, exports
`ENGINE_SOURCE_REVISION`, and starts the supplied command. It intentionally does not provision Stockfish or KataGo;
external evaluation artifacts remain explicit configuration inputs.

For local development, install `requirements-training.lock` with `uv` and build the extension with CMake:

```powershell
uv pip install --require-hashes --requirements .\py\requirements-training.lock
cmake -S .\cpp -B .\cpp\build -DCMAKE_BUILD_TYPE=Release
cmake --build .\cpp\build --parallel
```

The checked-in files under `configs/` are validated templates. Resolve their hardware, external artifacts, hashes,
paths, and approval record before a production run.

### Experiment overrides

An experiment YAML may inherit another experiment with a top-level `extends` path. The path resolves relative to
the inheriting file. Mapping fields merge recursively, while lists replace the inherited list. A discriminated
mapping such as a schedule replaces the complete inherited mapping when its `kind` changes:

```yaml
extends: ../../baselines/vast-go-7x7-2gpu-2h.yaml
run:
  run_name: go7-learning-rate-decay
training:
  save_path: py/training_data/screening/go7-learning-rate-decay
  trainer:
    learning_rate: {kind: linear, start_generation: 0, end_generation: 50, start_value: 0.005, end_value: 0.001, rounding: none}
```

Inheritance may be chained, but cycles are rejected. The fully resolved configuration is validated through the
canonical chess/Go Pydantic union. Approval records, resolved run JSON, and queue fingerprints use that effective
configuration, so changing a base invalidates approvals and queue summaries for every dependent experiment.

## Experiment queue

The queue is a resource wrapper above `train.py`; it does not add a training mode. Its command prefix owns all
arguments other than the experiment YAML path, so source revision and approval stay explicit. The following is a
schema sketch; replace every angle-bracket placeholder with the approved run and node values:

```yaml
schema_version: 1
runner:
  command:
    - python
    - py/run_approved_experiment.py
    - --approval-directory
    - <approval-directory>
  experiment_path_argument: --run-config
slots:
  - slot_id: <slot-id>
    cuda_devices: [<cuda-index>, ...]
    cpu_affinity: [<cpu-index>, ...]
    ram_capacity_bytes: <ram-capacity-bytes>
    working_directory: <repository-directory>
    log_directory: <slot-log-directory>
experiments:
  - experiment_id: <experiment-id>
    experiment_file: <experiment-yaml>
    resources:
      cuda_device_count: <exact-device-count>
      cpu_core_count: <requested-core-count>
      ram_limit_bytes: <requested-ram-limit-bytes>
summary_path: <queue-summary-path>
poll_interval_seconds: 0.1
termination_grace_seconds: 10.0
```

Paths resolve relative to the queue file. CUDA and CPU sets must not overlap between slots. A job consumes one
complete matching slot; its exact CPU affinity and aggregate RAM limit may be smaller than the slot capacity.
`run_approved_experiment.py` resolves the current repository revision at child launch and selects
`<approval-directory>/<experiment-yaml-stem>.json`; the approval remains revision- and configuration-specific.
The current four-GPU screening queue is committed at
[`configs/queues/vast-go-7x7-screening.yaml`](configs/queues/vast-go-7x7-screening.yaml).

### Memory monitoring

The queue samples the resident memory of the runner and every discovered descendant. If aggregate process-tree RSS
exceeds `ram_limit_bytes`, it terminates that experiment and records the observed usage and limit in the summary.
This works inside unprivileged rented containers without delegated cgroups. It is intentionally a sampled safety
monitor rather than a kernel-hard reservation, so the polling interval controls how quickly an overshoot is caught.
Queue children must not daemonize away from the supervised process tree.

### Starting and observing a queue

Run from an environment containing the locked Python dependencies:

```text
python py/queue_experiments.py run --queue-config <queue-yaml>
```

Keep that supervisor running. It launches every currently compatible job, captures separate stdout and stderr logs,
records exits, releases empty slots, and immediately schedules the next compatible pending job. Before each
scheduling pass it reloads the desired queue and every authored experiment. Pending entries may therefore be added,
removed, reordered, or changed without interrupting running work. A running, completed, or failed experiment ID is
immutable; changing one rejects that desired update and suspends new launches until the file is corrected. Set
`wait_for_updates_when_empty: true` to keep the supervisor alive after the current desired queue is empty. In another
terminal, inspect the atomic summary without modifying the queue:

```text
python py/queue_experiments.py status --summary <queue-summary-json>
```

To change future work, commit and pull changes to the queue YAML or any still-pending experiment YAML. The active
supervisor reloads them before its next scheduling pass. Running entries keep the exact configuration hash already
recorded in the summary. Stop the supervisor with `SIGINT` or `SIGTERM` only when active runs should also be stopped;
ordinary queue edits do not require a restart.

After selected queue entries are terminal, export their durable evidence without replay or completed games:

```text
python py/export_experiment_results.py --queue-config <queue-yaml> --output <results-zip>
```

The complete Vast workflow and archive contents are documented in
[`documentation/operations/experiment-result-export.md`](../documentation/operations/experiment-result-export.md).

The run command exits zero only when every experiment completed successfully and exits one when any experiment
failed. On `SIGINT` or `SIGTERM`, it sends `SIGTERM` to each active process group, waits the configured grace period,
forcibly kills any remaining tracked descendants, and records failures before exiting. Completed and failed entries
are terminal and are not run again; there is no automatic retry. Only `pending` entries are scheduled.

If a summary contains `running` entries after a supervisor restart, the wrapper marks them failed and stops without
signalling or adopting possibly stale process IDs. Verify the recorded process groups have ended, then invoke the
same queue again to continue entries still marked `pending`.

The summary fingerprint covers immutable slot, summary, and termination ownership. Each launched entry records the
exact runner command and independently resolved canonical configuration hash. The runner command, pending entries,
polling interval, and empty-wait control may change at runtime. Deleting the summary intentionally creates a fresh
queue and will make every configured experiment eligible to run again.

## Validation

Run Python validation from `py`:

```powershell
uv run ruff format
uv run ruff check --fix
python -m pytest --import-mode=importlib .\test -q
```

Retain `--import-mode=importlib`; the repository's `py` package otherwise conflicts with pytest's historical `py`
dependency during collection.

Native-facing changes also require the extension-backed Python suite and CTest target. External-engine integration
tests remain opt-in and require configured Stockfish or KataGo artifacts.

## Optional tools

`tools/` contains explicit manual utilities for benchmark evidence, UCI validation, Lichess model retrieval, and
Cute Chess compatibility. They are not alternate training or evaluation runtimes. In particular:

- `fetch_hf_model.py` and `validate_uci_transcript.py` are called by the Lichess deployment;
- `benchmark_interactive_engine.py` exercises the same engine used by web and UCI deployment;
- `run_cutechess_gauntlet.py` is a retained manual interoperability tool;
- benchmark scripts reproduce historical performance evidence under `documentation/benchmarks`.
