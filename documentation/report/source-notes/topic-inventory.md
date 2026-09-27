# Technical-report topic inventory

This inventory defines the minimum conceptual coverage required before publication writing resumes. A checked item
means that an evidence-linked source dossier contains the mechanism, alternatives, measurements, decision rationale,
pitfalls, and remaining unknowns. It does not merely mean that a benchmark is linked somewhere.

The check marks below were re-audited against the current tree, the experiment ledgers, the benchmark-coverage
ledger, the final configuration, relevant implementation paths, Git history, and the six preserved release tags.
The owner-memory questions are resolved. The two unchecked items are terminal figure/accounting work; they gate those
publication claims, not narrative writing from the completed dossiers. The audit and exact boundary are recorded in
[the completeness audit](completeness-audit.md).

## Search and target generation

- [x] Fixed search budgets and budget schedules
- [x] Randomized full and cheap searches, including the KataGo rationale and why the trade differed in chess
- [x] Hand-written adaptive stopping rules
- [x] Predicted per-position search budgets
- [x] Learned in-search stopping
- [x] Monte Carlo tree search selection, expansion, backup, root noise, and policy-target construction
- [x] Monte Carlo graph search and exact-history transpositions
- [x] Neural-inference caching and the attempted shared-cache worker topology
- [x] Parallel leaf selection, virtual loss, collision handling, and batch occupancy
- [x] Tree retention and root advancement
- [x] First-play urgency
- [x] Forced playouts and policy-target pruning
- [x] Search-backup discounting, replay-target discounting, root-value blending, and draw/value treatment
- [x] Search cutoffs, terminal handling, and late-game target poisoning

## Model architecture and representations

- [x] Every policy representation and head, including dense, plane-based, and from-to/action-attention variants
- [x] Policy/value trunk sharing; owner confirms split trunks were neither tested nor a serious design path
- [x] Convolutional, attention, and hybrid trunks
- [x] Residual-block variants and activation placement
- [x] Global-context and pooling mechanisms
- [x] Value-head representations and widths
- [x] Input feature planes, history, repetition, and rule-state encoding
- [x] Quantization-driven architectural constraints
- [x] Progressive model sizing, failed loss-based promotion, match gating, and function-preserving growth
- [x] Auxiliary policy and remaining-length heads
- [x] Distillation and smaller deployment models
- [x] Terminal student saturation under extended passes over one fixed replay buffer

## Training, replay, and curriculum

- [x] Optimizers, learning-rate schedules, warmup, clipping, and weight decay
- [x] Causal bounds on optimizer, replay-regime, objective, and quantized-serving explanations of the terminal gain
- [x] Initialization, calibration, and deterministic seeding failures
- [x] Replay capacity growth and the difference between capacity, freshness, and unique data
- [x] Replay reuse and presentation-credit accounting
- [x] Row admission, row selection, sampling probability, and loss weighting as separate mechanisms
- [x] Policy-surprise and difficult-state prioritization
- [x] Random openings and opening diversity
- [x] Restart-state extraction, weighting, reservation, and correction
- [x] Resignation calibration, continuation games, and false-resignation control
- [x] Cut-game value targets and censored remaining-length targets
- [x] Reanalysis designs and why they were removed or not revived
- [x] Synchronous quanta, overlapping self-play, and the boundary with fully asynchronous learning
- [x] Auxiliary-target materialization and eligibility
- [x] State canonicalization, board-symmetry augmentation, and transformed policy/legal targets
- [x] Model-publication cadence, refresh cost, and target-freshness trade-offs

## Inference and throughput engineering

- [x] Python-to-native ownership transfer and the discarded pipe, queue, client/server, and asyncio designs
- [x] Native tree arenas and direct inference submission
- [x] Search batching, queueing, and CUDA graph capture
- [x] Actor counts, inference-worker sharing, GPU partitioning, and contention
- [x] TorchScript
- [x] `torch.compile`, including attempted modes and why it was not retained where applicable
- [x] TensorRT FP16 compilation and serving
- [x] Post-training INT8 failure modes
- [x] Quantization-aware training and pre-fold deployment
- [x] TensorRT templates, refitting, equality-scale failure, and fidelity validation
- [x] Replay storage, materialization, compaction, memory mapping, and prefetch
- [x] Persistent distributed training and process lifetime
- [x] Evaluation throughput versus interactive latency
- [x] Hardware-transfer limits and shape-dependent performance

## System integration and reliability

- [x] Python and C++ ownership boundaries
- [x] Coordinator, self-play, replay, trainer, checkpoint, publisher, and evaluator interactions
- [x] Commands, messages, data products, and synchronization points between components
- [x] Training and inference artifact lifecycles
- [x] Evaluation callbacks into native search and external opponents
- [x] Credit flow and backpressure
- [x] Crash recovery, pending work, quarantine, and discarded-data boundaries
- [x] Configuration, manifest, and artifact identity
- [x] Multi-layer system interaction SVG specification and source mapping

## Evaluation and interpretation

- [x] Policy-only evaluation
- [x] Fixed-search ladder evaluation
- [x] Deep-search evaluation
- [x] External-engine calibration and uncertainty
- [x] Strength, target fidelity, throughput, admitted-data rate, training cadence, and wall-clock learning as distinct
  outcomes
- [ ] Final training curves, learning-rate curves, replay statistics, and systems telemetry (terminal summary exists;
  archive-derived figures remain)
- [x] Cross-generation progress figure using descriptive checkpoint/campaign labels rather than internal run identifiers
- [x] Matched-estimator plateau comparison, including transfer uncertainty and policy/search decomposition
- [x] Adaptive-rung transitions as a source of false terminal improvement
- [ ] Cost, hardware, runtime, and reproducibility accounting

## Reader-facing incident studies

- [x] Late-game target poisoning and drawn-out unconvertible games
- [x] Invalid or misleading attention comparisons
- [x] Non-deterministic initialization in supposedly matched runs
- [x] Misleading inference-cache hit-rate measurements
- [x] Adaptive-search proxy improvement without strength improvement
- [x] TensorRT conversion/refit passing mechanically while changing model behavior
- [x] Search-work savings failing to become wall-clock savings
- [x] Replay throughput gains failing to become fresh training information
