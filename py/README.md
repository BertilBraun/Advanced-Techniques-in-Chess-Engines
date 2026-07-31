# Python multi-game control-plane foundation

Python provides the typed control plane for the multi-game rework. The root
configuration is a discriminated union of complete Go and chess experiment
branches, so one run selects exactly one compatible game/configuration stack.
The active Go predecessor modules cover self-play, inference batching, replay
credits/storage, training and checkpoint resume, evaluation, adaptive
calibration, experiment matrices, and research reports.

Chess configuration and validation exist as an architectural foundation.
Native chess execution, model training, replay materialization, and evaluation
remain pending. The current `az_go_native` callback path is retained only as
the predecessor Go implementation while native multi-game sessions and
LibTorch inference are restored.

Run commands from this directory so the repository's `src` package is
importable.

## Install

```powershell
python -m pip install -r .\requirements.txt
```

Build the native extension first using the instructions in `..\cpp\README.md`,
then add its build directory to `PYTHONPATH`.

## Configuration

The top-level `game` discriminator selects the complete Go or chess branch.
Validate and resolve the checked-in Go authoring configuration:

```powershell
python -m src.az.config.cli validate .\configs\go\go-7x7-fixed.authoring.json
python -m src.az.config.cli resolve .\configs\go\go-7x7-fixed.authoring.json `
    --output .\configs\go\go-7x7-fixed.resolved.json
```

The fixed, progressive, and mixed files are experiment definitions, not
strength recommendations. Adaptive configurations are produced only after an
authenticated calibration artifact exists.

## Tests

```powershell
python -m pytest --import-mode=importlib .\test -q
```

Native-extension tests skip clearly when `az_go_native` is not importable.
CUDA and multi-rank integration tests skip unless their required infrastructure
is configured. A short system smoke is the local readiness check; full
multi-GPU training is deliberately not part of local validation.

## Runtime execution is deferred

Do not launch training, performance benchmarks, prolonged self-play, or full
evaluation until the target compute environment is explicitly available. The
commands below document the existing Go lifecycle for later execution; they do
not imply that the multi-game rework or native chess runtime is complete.

## Experiment lifecycle

`src.az.experiment.cli` is the existing Go predecessor entry point. A run
directory owns one immutable run ID, resolved-configuration digest, source
revision, checksummed artifact lineage, and resumable phase state.

```powershell
python -m src.az.experiment.cli validate .\configs\go\go-7x7-fixed.authoring.json
python -m src.az.experiment.cli resolve .\configs\go\go-7x7-fixed.authoring.json `
    --output .\configs\go\go-7x7-fixed.resolved.json
python -m src.az.experiment.cli freeze .\configs\go\go-7x7-fixed.resolved.json `
    --run-directory 'C:\runs\go-7x7-fixed' `
    --artifact-root 'C:\' `
    --run-id 'replace-with-a-new-uuid' --repository-root .. `
    --dependency-lock .\requirements-training.lock --build-id 'rental-node-build'
python -m src.az.experiment.cli run --run-directory 'C:\runs\go-7x7-fixed'
```

The training run performs native self-play, replay publication, exact-credit
training, checkpoint publication, and worker model refresh under the same
monotonic wall-clock deadline. At configured elapsed times it claims immutable
model snapshots with requested and actual publication timing. Evaluation and
reporting are separate resumable phases, so their cost is reported but does not
extend the training clock:

```powershell
python -m src.az.experiment.cli evaluate --run-directory 'C:\runs\go-7x7-fixed'
python -m src.az.experiment.cli report --run-directory 'C:\runs\go-7x7-fixed'
python -m src.az.experiment.cli status --run-directory 'C:\runs\go-7x7-fixed'
```

`stop` writes an authenticated request checked by the active runtime. After it
has stopped at a safe boundary, `resume` authenticates the frozen configuration,
source/run identity, replay credits, checkpoint, elapsed schedule, and retained
artifacts before continuing the remaining duration.

Adaptive stopping is enabled only from a typed calibration request with an
explicit candidate grid and acceptance rule. The command authenticates the run,
durable replay commit journal, and registered trace files, then prints a
`CalibrationArtifactReference`:

```powershell
python -m src.az.experiment.cli calibrate `
    --run-directory 'C:\runs\go-7x7-fixed' `
    --request .\calibration-request.json
```

The printed reference can be inserted directly into a later adaptive-search
configuration. Freeze that later run with the source run as
`--reference-artifact-root`; freeze copies and authenticates the calibration
below the new run's `reference-artifacts` directory.

### Very short CPU readiness smoke

This uses the real 3x3 Go rules, native MCTS, replay codec, trainer, checkpoint,
paired evaluation, and report builder. It is only a wiring check and makes no
playing-strength or throughput claim.

```powershell
python -m src.az.experiment.cli write-smoke-config --output .\smoke.resolved.json
python -m src.az.experiment.cli freeze .\smoke.resolved.json `
    --run-directory 'C:\runs\go-local-readiness' `
    --artifact-root 'C:\' `
    --run-id '00000000-0000-0000-0000-000000000711' `
    --repository-root .. --dependency-lock .\requirements-training.lock `
    --build-id 'local-smoke'
python -m src.az.experiment.cli run --run-directory 'C:\runs\go-local-readiness'
```
