# 9. Reproducibility

## Two reproducibility targets

The project maintains two distinct targets:

1. **Recipe reproduction:** use the current fully expanded
   [`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) as the supported entry point.
2. **Result reproduction:** use the frozen source revision, resolved config hash, manifest, checkpoints, engines,
   datasets, and archive recorded for the final result.

The first may evolve; the second must not.

## Result identity and provenance

The frozen local result record and its linked evidence identify these layers of provenance:

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

The authoritative values belong in [the final result record](../results/final-chess-run.md) and Chapter 7. The
large run archives and some exact evaluation inputs are currently local rather than published with the Git
repository. The public recipe and model artifact support inspection and a new run, but they are not yet a
self-contained package for bitwise or exact-match reproduction of the reported experiment.

## Reproducing the software

Local setup and validation begin in the root [README](../../README.md), the [Python guide](../../py/README.md), and
the [native runtime guide](../../cpp/README.md). Production nodes are provisioned by
[`deployment/setup_remote.sh`](../../deployment/setup_remote.sh), which installs locked dependencies, builds the
Release extension, installs pinned evaluation engines, and runs engine smokes. Run lifecycle operations go through
[`deployment/run_control.sh`](../../deployment/run_control.sh).

This chapter intentionally does not duplicate commands from current operational documentation. The
[experiment platform](../operations/experiment-platform.md), [run control](../operations/run-control.md), and
[result export](../operations/experiment-result-export.md) documents are the executable authorities.

Original project code and documentation, including this report, are available under the repository
[MIT License](../../LICENSE). The published final model artifacts carry the same license in the
[Hugging Face model repository](https://huggingface.co/BertilBraun/alphazero-chess). External dependencies,
reference sources, and third-party data retain their own terms.

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

The plots derive from archived JSON, CSV, TensorBoard, or manifest data. Figure inputs and extraction methods are
retained beside the final evidence archive. Derived tables preserve the raw columns needed to recompute totals,
rates, Elo transformations, and uncertainty intervals; live-dashboard values are not treated as publication data.
