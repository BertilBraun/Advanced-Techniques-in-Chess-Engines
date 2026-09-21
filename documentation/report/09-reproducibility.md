# 9. Reproducibility

## Two reproducibility targets

The project maintains two distinct targets:

1. **Recipe reproduction:** use the current fully expanded
   [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) as the supported entry point.
2. **Result reproduction:** use the frozen source revision, resolved config hash, manifest, checkpoints, engines,
   datasets, and archive recorded for the final result.

The first may evolve; the second must not.

## Required final evidence bundle

The publication bundle should contain or identify:

- Git source revision and clean/dirty state;
- resolved YAML and SHA-256;
- dependency lock and hash;
- operating image, Python, PyTorch, CUDA, cuDNN, driver, GPU, CPU, RAM, and disk facts;
- Stockfish and KataGo versions and archive hashes where applicable;
- evaluation dataset and opening-suite identities and hashes;
- run manifest, approval record, coordinator logs, TensorBoard events, and resource telemetry;
- replay schema, final capacity/occupancy, and reconciled volume counters;
- selected training checkpoint and trimmed inference artifact hashes;
- ONNX, TensorRT template/engine provenance, calibration positions, and fidelity reports;
- raw terminal match records, aggregate reports, commands, and confidence-interval method;
- one digest covering the fetched archive or a checksummed artifact manifest.

The authoritative values belong in [the final result record](../results/final-chess-run.md) and Chapter 7.

## Reproducing the software

Local setup and validation begin in the root [README](../../README.md), the [Python guide](../../py/README.md), and
the [native runtime guide](../../cpp/README.md). Production nodes are provisioned by
[`deployment/setup_remote.sh`](../../deployment/setup_remote.sh), which installs locked dependencies, builds the
Release extension, installs pinned evaluation engines, and runs engine smokes. Run lifecycle operations go through
[`deployment/run_control.sh`](../../deployment/run_control.sh).

This chapter intentionally does not duplicate commands from current operational documentation. The
[experiment platform](../operations/experiment-platform.md), [run control](../operations/run-control.md), and
[result export](../operations/experiment-result-export.md) documents are the executable authorities.

## Reproducing evaluation

Use the preserved checkpoint and inference artifact rather than re-exporting it with a newer toolchain. Reuse the
same paired opening suite, colors, opponent binary, node limit, threads, hash, candidate search budget, parallelism,
batching, and adjudication rules. Report every game and recompute aggregates independently.

For latency, separate:

- isolated model-forward throughput;
- saturated many-position search throughput;
- single-game interactive latency.

Only compare like with like. The [Stockfish gauntlet](../operations/stockfish-gauntlet.md) defines the current match
procedure, while [evaluation engines](../operations/evaluation-engines.md) owns binary identity.

## Reproducing plots and tables

Plots should be generated from archived JSON, CSV, TensorBoard, or manifest data. Every figure should name the source
archive and extraction script or command. Derived tables should preserve enough raw columns to recompute totals,
rates, Elo transformations, and uncertainty intervals. Hand-copied live-dashboard values are not publication data.

## Minimum validation before publication

- verify every internal link and anchor;
- verify hashes against artifacts rather than copied prose;
- reconcile generation, optimizer-step, game, position, and presentation counts;
- check that every Elo row identifies its calibration and protocol;
- ensure no live-node address, private key, or unarchived path is treated as permanent evidence;
- render the report and figures and inspect layout if a PDF edition is produced;
- preserve the Markdown report as the canonical editable source.
