# Experiment result export

Use `py/export_experiment_results.py` after queue experiments are terminal to create one portable ZIP without
altering their run directories. By default, every experiment in the durable queue summary is selected; the command
fails if any selected experiment is pending or running.

For successful worktree-isolated runs, export reads the queue-preserved authored configuration chain under the
central runtime directory rather than depending on the disposable worktree or current control checkout. Failed runs
retain their worktree, which remains the configuration fallback for diagnostic exports.

The archive contains authored and resolved configuration, run identity and outcome, resource telemetry, evaluation
results and manager state, TensorBoard events, queue logs, every elapsed evaluation checkpoint's model and inference
artifacts, and the latest checkpoint's optimizer state. It excludes replay storage, completed self-play games,
generated evaluation inputs, engines, and unrelated checkpoints. `archive-manifest.json` records every included
source/archive path, reason, size, and SHA-256. Missing required files, changed identities, unsafe paths, and artifact
hash mismatches abort the export and remove the partial ZIP.

## Vast export

Run from the repository root after the queue summary reports terminal results:

```bash
mkdir -p /workspace/exports
/workspace/alphazero-engine-venv/bin/python py/export_experiment_results.py \
  --queue-config py/configs/queues/QUEUE.yaml \
  --output /workspace/exports/SCREENING-results.zip \
  --tensorboard-log-root logs \
  --queue-stdout-log /workspace/run-control/queue.stdout.log \
  --queue-stderr-log /workspace/run-control/queue.stderr.log
```

Use repeated `--experiment-id ID` arguments to export only named terminal entries from a still-active queue. If a
terminal experiment was removed from the live desired queue, restore it or supply its immutable authored file with
`--experiment-config ID=PATH`. This does not make running experiments exportable.

The queue supervisor's own log arguments are optional because supervisors launched interactively may not have
files. Their absence is recorded in the manifest. Per-experiment stdout and stderr paths from the durable summary
are always required.

## Download and verify

From the Windows development host:

```powershell
scp -i C:\Users\berti\.ssh\codex_vast_ed25519 -P 56488 `
  root@171.101.230.38:/workspace/exports/SCREENING-results.zip `
  C:\Projects\Papers\projects\advanced-techniques-chess-engines\evidence\
```

Inspect `archive-manifest.json` after download and retain the ZIP as the immutable experiment backup. Export does
not delete or modify remote run data; remote cleanup is a separate, explicit operation.
