# V10 training-quality implementation

Implemented behavior:

- Credit evaluation has an inspection tier at the configured inspection interval and a full tier at its
  configured multiple. Inspection evaluates the fixed dataset, paired policy and MCTS games against random,
  one paired checkpoint offset, and a smaller color-swapped teacher-budget search. The scheduler retains one
  deferred evaluation of the other tier so a slow full evaluation cannot continually erase odd inspection
  checkpoints.
- Self-play model refreshes discard every retained MCTS root while preserving each game's board and move
  history.
- Search telemetry retains entropy and KL-to-uniform and additionally reports visit-policy KL against the raw
  pre-Dirichlet prior, top-move disagreement, and the prior rank of the search-selected move.
- Producer shards aggregate compatible canonical encoded rows before credits are issued. Policies and scalar
  targets are averaged, occurrence multiplicity is retained, and training uses normalized capped
  square-root multiplicity weights. Conflicting hard WDL targets remain separate and are counted in telemetry.
- Compaction repeats the same compatible-target aggregation across all producer shards selected for each
  roughly 100,000-position container after materializing reanalysis overrides. A merged position belongs to
  its newest contributing shard for FIFO retention, while already-issued presentation credits remain
  cumulative and the live replay population shrinks to the compacted unique count.
- Material adjudications no longer become hard WDL labels. Their continuous perspective-correct score is
  stored separately and trained with Huber loss.
- Schema 6 stores a lossless starting FEN and complete UCI history for generated chess samples. After each
  publication, worker zero re-searches a bounded configured fraction of one recent replay payload with the
  published model and full self-play search budget. Immutable, source-hash-bound sidecars refresh policy and
  MCTS scalar targets during replay decode without altering terminal outcome/material supervision or credits.
  Disjoint sidecars are folded in model order, repeated rows use the newest target, and compaction materializes
  all active overrides before retiring producer payloads and their sidecars.
- Each self-play process keeps a bounded in-memory archive of complete encoded move prefixes at plies 1-10.
  A configured minority of new games starts from the archive using a smoothed, capped distribution favoring
  genuine policy-search disagreement.

Known limits:

- Duplicate aggregation spans producer shards within each compacted container, but not separate containers or
  the uncompacted tail of the replay window. Truly window-global aggregation still needs a persistent
  row-level canonical index with representative promotion during FIFO eviction.
- Reanalysis is deliberately bounded and synchronous inside one designated self-play worker at model refresh.
  It does not consume trainer GPU time or pause credit accounting, but that worker acknowledges the publication
  only after its bounded pass. Sidecars survive disjoint periodic passes and the newest target wins when a row
  is refreshed repeatedly. The designated worker currently selects producer payloads in the replay root;
  compacted containers retain materialized targets but are not selected for another reanalysis pass.
