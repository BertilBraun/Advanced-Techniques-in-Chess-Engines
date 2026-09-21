# Architecture

- [Current system](../system/README.md): reader-facing description of the architecture that is implemented now.
- [Python runtime rework](python-runtime-rework.md): accepted runtime topology and process-model design record. Its
  fixed-row replay and learned-budget sections were superseded; use the current-system guides for current behavior.
- [Columnar replay and shard ingestion](replay-pipeline-rework.md): accepted replay-store and materialization design.
- [Replay materialization analysis](replay-materialization-rework.md): analysis that motivated the current
  per-worker-directory materializer. Retained as rationale rather than an additional runtime design.
- [Progressive model sizing](progressive-model-sizing.md): current progressive training and promotion policy, with
  historical KataGo rationale retained explicitly.
- [Learned adaptive search budget](learned-search-budget.md): superseded design record for the removed adaptive
  search system. Current search uses fixed visit limits.
- [Platform rework](platform-rework.md): the **closed** R1–R12 execution ledger of the multi-game platform
  rework. Historical record only; it authorises nothing. Current work is tracked in
  [Current state](../CURRENT-STATE.md) and the [final-run result record](../results/final-chess-run.md).

The checked-in final recipe is
[`py/configs/production/chess-final-config.yaml`](../../py/configs/production/chess-final-config.yaml). Historical
roadmaps elsewhere in Git history or under `documentation/history/` do not authorize work.
