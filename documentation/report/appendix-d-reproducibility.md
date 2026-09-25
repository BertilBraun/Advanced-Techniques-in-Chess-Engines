# Appendix D. Reproducibility and release boundary

## Two reproducibility targets

Reproducing the current system and reproducing the reported result require different starting points:

1. **Recipe reproduction:** use the current fully expanded `chess-final-config.yaml` as the supported entry point.
2. **Result reproduction:** use the frozen source revision, resolved config hash, manifest, checkpoints, engines,
   datasets, and archive recorded for the final result.

The recipe may evolve; the reported result remains fixed.

## Expanded chess recipe settings

The architectural settings summarized in Chapter 7 include a key-size-128 chess from-to policy head and a
two-channel WDL head with a 48-unit hidden layer. The training-only next-searched-policy and remaining-game-length
heads have loss weights 0.15 and 0.1. Primary policy and value losses each have weight 1.0. Terminal outcome
targets are discounted by 0.998 per ply; the search-root-value blend rises from zero to 0.1 over its configured
schedule, while search backup uses a separate 0.99 per-ply discount.

Each 500-step training quantum uses eight ranks processing 256 positions each, for a global batch of 2,048.
Nesterov SGD uses momentum 0.9, weight decay 0.0001, and gradient clipping at norm 1.0. The learning rate warms
from zero to 0.1 over the first 1,000 optimizer steps and then follows the configured linear schedule to 0.01.
Training uses bfloat16 and persistent trainer processes; `torch.compile` is disabled. QAT calibration uses 516
real evaluation positions and is refreshed at every publication boundary.

The 32 self-play actors run four per GPU, with 512 interleaved games per actor. Native inference uses batches of
320 with two outstanding batches per worker. Search uses exploration constant 1.5, reduced-parent FPU with
reduction 0.2, forced playout coefficient 1.5, and Dirichlet epsilon 0.25 with alpha 0.3. Restart-state
selection retains a 30% uniform component. Eight materializers convert completed games into the fixed-layout
memory-mapped replay store; training credit is committed only against durable admitted rows.

The successor network trains on a captured replay snapshot for an average of 1.5 optimizer quanta per
active-model quantum, with its own catch-up learning-rate clock. The configured match gate requires the successor
to score at least 0.48 in two consecutive paired evaluations. The fully expanded config [10] specifies the
stage-specific ladder-plateau thresholds and window.

## Result identity and provenance

The frozen local result record identifies these layers of provenance:

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

The published INT8 ONNX has SHA-256 `d634abacae3c874eac6ded89f6af861eb81b509da638b5ad710587b1a08be658` at
the immutable model-repository revision `dc8fccccb67ab5ec9e36267a165a9700b7dbf55f` [11]. The source release's
evidence index [10] records additional artifact hashes. The frozen training-source revision, resolved configuration,
large run archives, and some exact evaluation inputs remain in the local result archive. The public recipe and model
support inspection and a new run, but are not yet a self-contained package for exact-match reproduction of the
reported experiment.

## Reproducing the software

Local setup and validation begin in the public source-code release [10]. Production nodes are provisioned by
`deployment/setup_remote.sh`, which installs locked dependencies, builds the
Release extension, installs pinned evaluation engines, and runs engine smokes. Run lifecycle operations go through
`deployment/run_control.sh`.

Original project code and documentation, including this report, are available under the MIT License in the
source release [10]. The published final model artifacts carry the same license in the model repository [11].
External dependencies, reference sources, and third-party data retain their own terms.

## Reproducing evaluation

Use the preserved checkpoint and inference artifact rather than re-exporting it with a newer toolchain. Reuse the
same paired opening suite, colors, opponent binary, node limit, threads, hash, candidate search budget, parallelism,
batching, and adjudication rules. Report every game and recompute aggregates independently.

For latency, separate:

- isolated model-forward throughput;
- saturated many-position search throughput;
- single-game interactive latency.

Only compare like with like. Appendix B states the terminal opponent, openings, game count, and inference artifact.

## Reproducing plots and tables

The plots derive from archived JSON, CSV, TensorBoard, or manifest data. Figure inputs and extraction methods are
retained beside the final evidence archive. Derived tables preserve the raw columns needed to recompute totals,
rates, Elo transformations, and uncertainty intervals; live-dashboard values are not treated as publication data.
