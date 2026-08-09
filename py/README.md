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

## Experiment queue

The queue is a resource wrapper above `train.py`; it does not add a training mode. Its command prefix owns all
arguments other than the experiment YAML path, so source revision and approval stay explicit. The following is a
schema sketch; replace every angle-bracket placeholder with the approved run and node values:

```yaml
schema_version: 1
runner:
  command:
    - python
    - py/train.py
    - --expected-source-revision
    - <revision>
    - --approval-file
    - <approval-file>
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
complete matching slot; its exact CPU affinity and RAM limit may be smaller than the slot capacity. On `SIGINT` or
`SIGTERM`, the wrapper terminates every active process group and records failures before exiting.

If a summary contains `running` entries after a supervisor restart, the wrapper marks them failed and stops without
signalling or adopting possibly stale process IDs. Verify the recorded process groups have ended, then invoke the
same queue again to continue entries still marked `pending`.

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
