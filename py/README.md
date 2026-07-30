# Python Go experiment platform

Python provides the typed control plane around the native `az_go_native`
extension. The active modules cover configuration, self-play, inference
batching, replay credits/storage, training and checkpoint resume, evaluation,
adaptive calibration, experiment matrices, and research reports.

Run commands from this directory so the repository's `src` package is
importable.

## Install

```powershell
python -m pip install -r .\requirements.txt
```

Build the native extension first using the instructions in `..\cpp\README.md`,
then add its build directory to `PYTHONPATH`.

## Configuration

Validate and resolve a Go authoring configuration:

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
