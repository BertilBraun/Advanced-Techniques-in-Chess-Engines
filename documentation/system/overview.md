# End-to-end system overview

## What the system does

The project is a configuration-driven AlphaZero training and evaluation platform for chess and small-board Go. The
current final-strength effort is chess. Python owns configuration, process supervision, replay, training,
checkpointing, evaluation, and reporting. A C++20 extension owns authoritative game rules and state transitions,
packed input encoding, batched neural inference, Monte Carlo tree search, and retained-root analysis. The two sides
meet at a coarse typed action-ID boundary; there is no second Python search implementation.

The relevant source maps are [`py/README.md`](../../py/README.md), [`cpp/README.md`](../../cpp/README.md), the
canonical experiment union in
[`py/src/experiment/configuration.py`](../../py/src/experiment/configuration.py), and the native game/search layout
under [`cpp/src`](../../cpp/src).

## Runtime topology

```text
Coordinator process
├── replay manager and credit ledger
├── training session
│   └── one persistent DDP trainer group per eligible model
├── 32 persistent self-play worker processes
├── eight persistent replay-materialization worker processes
└── short-lived evaluation job processes
```

The coordinator loop in
[`py/src/training/coordinator.py`](../../py/src/training/coordinator.py) appends sealed replay shards, reconciles
training credits, applies self-play backpressure, collects and schedules evaluations, supervises workers, checks run
limits, and starts a blocking optimizer quantum when both data and credits permit. Training is synchronous at the
quantum boundary. Self-play is partly overlapped: the final configuration pauses 16 of 32 self-play workers while
the eight-rank DDP group trains, while the other workers may continue producing completed games. Replay is immutable
for the captured training snapshot.

One generation is one 500-step optimizer quantum. Completed optimizer steps are the persisted progress coordinate;
the generation is derived from them. The final run has no configured wall-time, cost, or optimizer-step terminal
limit and therefore stops through its explicit manual stop file and the run-control workflow.

## Final chess recipe at a glance

[`chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml) specifies:

- eight RTX 4070 SUPER GPUs for both eight-rank NCCL training and distributed self-play;
- a global batch of 2,048, split into local batches of 256;
- SGD with Nesterov momentum, weight decay `0.0001`, gradient clipping at `1.0`, and a linear learning-rate schedule
  from `0.1` to `0.01` through generation 1,000;
- a progressive convolutional ladder of 12×128, 14×160, and 19×176 networks;
- pre-fold INT8 quantization-aware training and TensorRT inference;
- fixed self-play search budgets increasing from 300 to 800 visits;
- a replay ratio of four, policy-surprise sampling, and a replay capacity growing from 600,000 to 20 million
  positions;
- mixed random-opening and archived restart-state starts, forced playouts, retained roots, and calibrated
  resignation;
- elapsed 20-minute policy-only and 64-search Stockfish evaluation ladders.

This list is descriptive. The YAML is the authoritative owner of every exact value and schedule.

## The production data path

1. A native search batch evaluates one position for every active self-play game and returns sparse visits, a policy
   target with forced exploration removed, WDL-derived root values, and diagnostics.
2. The Python worker chooses a move, advances the retained root, and eventually atomically publishes a typed
   completed-game JSON record.
3. A bounded dispatcher renames game files into per-worker directories. Long-lived materializers reconstruct and
   validate trajectories, encode columnar replay shards, quarantine isolated bad games, and seal manifests last.
4. The coordinator appends all sealed shards to a preallocated memory-mapped FIFO and awards presentation credits
   only after the append is flushed.
5. Every DDP rank maps the same immutable replay snapshot read-only, deterministically samples disjoint local rows,
   applies chess augmentation, prefetches batches, and trains the complete model.
6. Rank zero writes model, optimizer, QAT state, and a trimmed deployment artifact, then writes the checkpoint
   manifest last.
7. The coordinator publishes exactly one active checkpoint, refreshes self-play, and makes the checkpoint available
   to elapsed-time evaluation.

The live implementations are
[`self_play/worker.py`](../../py/src/self_play/worker.py),
[`replay/manager.py`](../../py/src/replay/manager.py),
[`training/session.py`](../../py/src/training/session.py), and
[`evaluation/manager.py`](../../py/src/evaluation/manager.py).

## Artifact and restart boundaries

Bulk data crosses processes through files rather than Python object graphs:

- completed trajectories are atomic JSON files;
- replay is one schema-checked columnar memory map;
- checkpoints contain training weights, optimizer state, QAT state, inference artifact, and a manifest written last;
- `credit-ledger.json`, `progressive-training.json`, evaluation manager state, resignation state, and per-worker
  restart databases provide focused durable control state;
- evaluation successes and failures are atomic typed result artifacts.

Restart recovery is intentionally bounded rather than fully transactional. A pending progressive quantum resumes
from the first incomplete model on the exact recorded replay identity. In-flight self-play games are suspended and
restored when possible. The replay store prevents immediate duplicate shard application, but a crash at an append
boundary may still lose or duplicate a small bounded shard set; this is an accepted tradeoff documented in
[`replay-pipeline-rework.md`](../architecture/replay-pipeline-rework.md).

## Reproducing a reported run

Use the final YAML as the readable entry point, but do not treat its current contents as the immutable identity of a
past result. A report must retain:

- the exact source revision and clean-workspace provenance;
- authored and canonical resolved configuration plus its SHA-256;
- approval and run outcome;
- hardware, runtime image, locked dependency identity, and external-engine hashes;
- checkpoint manifest, model/inference hashes, and progressive-model state;
- evaluation inputs and result artifacts;
- TensorBoard events, logs, and resource telemetry.

The operational owner of launch, stop, preservation, and fetch is
[`run-control.md`](../operations/run-control.md). The archive contract is documented in
[`experiment-result-export.md`](../operations/experiment-result-export.md).
