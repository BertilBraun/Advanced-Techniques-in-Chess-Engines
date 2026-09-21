# Current system

These pages describe the system that is implemented now. They are organized by responsibility rather than by the
order in which the project was developed:

- [End-to-end overview](overview.md) — process ownership, the production loop, artifacts, and reproducibility;
- [Search and self-play](search-and-self-play.md) — native MCTS, batching, start states, move selection, resignation,
  and game publication;
- [Replay and data](replay-and-data.md) — materialization, the columnar replay store, sampling, credits, and restart
  behavior;
- [Training and model](training-and-model.md) — the final network ladder, objective, DDP training, QAT, promotion,
  and checkpoint publication;
- [Inference and evaluation](inference-and-evaluation.md) — TorchScript/TensorRT deployment and the asynchronous
  Stockfish evaluation ladder.

The living recipe is
[`py/configs/production/chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). It is fully
expanded and intentionally has no `extends` chain. These guides explain that configuration; they are not another
source of defaults.

The living recipe and a published result have different stability contracts. The recipe may be updated if the
project's settled configuration changes. A reported run must instead pin its exact Git revision, canonical resolved
configuration and SHA-256, approval, run manifest, checkpoint hashes, engine artifacts, and hardware/runtime
identity. Final-run measurements are still pending and are tracked in
[`documentation/results/final-chess-run.md`](../results/final-chess-run.md); this section makes no terminal strength
claim.

## Authority and historical material

The current implementation and final configuration take precedence when an older design record disagrees with
these pages. In particular:

- [`python-runtime-rework.md`](../architecture/python-runtime-rework.md) records the runtime redesign but includes
  replay and adaptive-search descriptions that were later replaced;
- [`replay-pipeline-rework.md`](../architecture/replay-pipeline-rework.md) records the accepted columnar replay and
  per-worker materialization design;
- [`progressive-model-sizing.md`](../architecture/progressive-model-sizing.md) explains the core sizing mechanism,
  but some example architectures and plateau constants predate the final configuration;
- [`learned-search-budget.md`](../architecture/learned-search-budget.md) is a retained design record for a removed
  system, not current behavior;
- [`platform-rework.md`](../architecture/platform-rework.md) is a closed historical ledger.

Operational commands, node setup, approvals, stopping, preservation, and export remain in
[`documentation/operations`](../operations/README.md). They are deliberately not duplicated here.
