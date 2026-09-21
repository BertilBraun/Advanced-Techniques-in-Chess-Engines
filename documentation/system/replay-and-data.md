# Replay and training data

## Durable flow from games to replay

Self-play publishes atomic completed-game JSON files. Replay materialization is a separate, continuously running
pipeline:

1. a bounded dispatcher renames inbox files into eight stable per-worker directories;
2. one long-lived process owns each directory and consumes its files in local counter order;
3. each game is reconstructed through the authoritative native chess state contract and validated;
4. valid games are encoded into typed columnar shards; an isolated invalid game is moved to `rejected/`;
5. shard data is written first and its manifest is written last;
6. the coordinator appends every sealed shard to the live memory map, flushes once, applies resignation evidence,
   and deletes the shard files.

The current implementation is
[`replay/manager.py`](../../py/src/replay/manager.py),
[`replay/dispatch.py`](../../py/src/replay/dispatch.py), and
[`replay/materialization_worker.py`](../../py/src/replay/materialization_worker.py). Its design rationale is
[`replay-pipeline-rework.md`](../architecture/replay-pipeline-rework.md).

The dispatcher reads at most 4,096 inbox entries per pass, so its Python work is bounded independently of backlog
depth. Materialization groups at most 32 games and approximately 16 MiB of source JSON into a shard. Sealed staging
is bounded at 96 shards. A rolling 512-game rejection window fails the run if more than 5% of games are rejected;
individual corrupt games therefore do not wedge the run, while systemic schema or materialization failures remain
loud.

## Columnar FIFO store

Replay is one preallocated schema-checked binary memory map with a 20-million-row physical maximum. Its logical FIFO
capacity is scheduled from 600,000 positions through 1.2, 2.0, 2.8, 4, 6, 8, 12, and 16 million before reaching 20
million at generation 1,000. Increasing the logical capacity does not change the file layout.

[`ReplayLayout`](../../py/src/replay/layout.py) is the canonical schema. Separate fixed-width columns store packed
network inputs, sparse primary policy visits and legal actions, WDL and root value, auxiliary targets and
eligibility, sample weight, policy surprise, source generation, and source timestamp. The final recipe retains at
most 60 nonzero policy entries, sorted deterministically by visit count and action ID; batch construction normalizes
the retained mass.

The store is positional rather than globally time-sorted. Per-worker shard order is stable, but different
materializers may finish out of order. This is deliberate: eviction follows append order and the bounded cross-worker
reordering is acceptable for the replay distribution. The source generation and timestamp remain available for age
telemetry.

## Materialized targets

The materializer reconstructs the complete trajectory before emitting rows. This allows it to derive targets whose
meaning depends on future positions. Every eligible searched position carries:

- a sparse MCTS policy target with forced-playout excess pruned;
- a final-game WDL target transformed to that position's player-to-move perspective;
- the search root value for the scheduled outcome/root-value blend;
- next-ply search policy when a later searched observation exists;
- normalized remaining game length, censored for games cut at the configured ply cap;
- sample weight and search diagnostics used for replay priority and reporting.

Chess augmentation is selected after sampling and applied consistently to the encoded state, legal actions, primary
policy, and next-policy target. The materialization and batch boundaries are
[`replay/materialization.py`](../../py/src/replay/materialization.py) and
[`replay/batch_loader.py`](../../py/src/replay/batch_loader.py).

## Sampling and replay ratio

Each global batch contains distinct rows. The final policy-surprise sampler draws 30% of proposals uniformly and
70% proportional to policy surprise, capped at `2.0`; it removes duplicate draws until the 2,048-row global batch is
full. Policy surprise is the disagreement between the network prior and the completed search policy. Sampling is
deterministic for a given run seed and source optimizer step, and each of the eight DDP ranks receives a disjoint
256-row slice plus independently selected augmentation indices.

The configured replay ratio is four. Each newly appended position earns four presentation credits. One training
quantum consumes `2,048 × 500 = 1,024,000` credits, equivalent to 256,000 newly materialized positions at this
ratio. Credits are reconciled from the replay store's absolute appended-row total only after a flushed append, so
the ledger can be late by one coordinator iteration but cannot train against uncommitted data. Five funded quanta
of surplus trigger self-play backpressure on the configured pause subset.

## Training snapshot and prefetch

Training holds the replay-manager boundary while it captures a `ReplayDescription`, preventing a concurrent append
from changing head, size, or logical capacity. Every rank then opens the same file read-only and checks that those
values still match. A single CPU preparation thread fills a bounded prefetch queue of depth four. CUDA batches use
reusable pinned host slots, a persistent transfer stream, and events so CPU decoding, host-to-device transfer, and
GPU optimization overlap without reusing a buffer too early.

## Restart behavior and accepted limits

Atomic rename means one game exists at exactly one producer, worker, or rejected path. Worker directory counters and
deterministic shard identities make interrupted materialization repeatable. Sealed manifests are the shard commit
point, and the replay header's last transaction identity prevents immediate duplicate application.

The system intentionally does not journal an exact multi-shard replay transaction. A crash during the append/flush/
cleanup boundary can lose or duplicate a small bounded set of games. Credits are recovered from the store's absolute
appended total, resignation observations are identity-idempotent, and pending worker files and sealed shards resume
through the ordinary pipeline. This bounded recovery contract should not be described as exact replay
reconstruction.
