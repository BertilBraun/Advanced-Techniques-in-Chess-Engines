# Columnar replay and shard ingestion

Status: accepted implementation design, 2026-08-22.

This decision supersedes the old fixed-row replay design in `python-runtime-rework.md`. The producer-facing
completed-game JSON contract and per-game atomic inbox publication remain unchanged. Normal operation is deterministic
and processes every game; an unclean boundary restart may lose or duplicate a small bounded shard set.

## Canonical representation

`ReplayLayout` is the sole semantic schema. Its digest covers the packed state representation, target layout,
retention widths, ordered column descriptors, shapes, and dtypes. Materialization shards, the live replay store, and
the mapped loader use those same descriptors and `ReplayColumnViews`; they do not define row DTOs or adapters.

The live replay is one preallocated schema-4 binary file. It has a fixed 64 KiB header and aligned fixed-capacity
column slabs. The header records maximum and logical capacities, FIFO head and size, total appended and evicted rows,
append sequence, and the last transaction identity. Logical indices are converted to physical indices once and whole
columns are gathered with NumPy indexing. Each append copies a source column into at most two ring slices.

Columns retain packed encoded states, sparse visit policies and legal actions, WDL/root/weight/source metadata, and
the configured auxiliary targets. Storage-boundary validation enforces action ranges, uniqueness, legal subsets,
positive visit mass, finite/ranged scalar values, eligibility, padding, shapes, and precise dtypes.

## Materialization lifecycle

1. Self-play atomically publishes one typed completed-game JSON file.
2. The manager snapshots inbox files in `(mtime_ns, canonical filename)` order and durably assigns contiguous shard
   sequences before dispatch.
3. A worker receives a bounded ordered game batch, validates source size/hash/identity, materializes every game with
   the existing game-neutral semantics, and encodes samples directly into replay columns.
4. The worker flushes and atomically seals one uncompressed shard data file, writes its typed manifest last, and only
   then removes the consumed inbox files.
5. Worker completion order does not affect replay order. Only the maximal contiguous sealed sequence is appendable;
   a missing or invalid earlier sequence blocks later shards.

The default bounds are 32 games and a soft 16 MiB of source JSON per shard. An indivisible oversized game becomes a
singleton. Transient worker failures retry the same durable claim; invalid game data is surfaced as a fatal run error
without dropping or bypassing the game.

## Boundary append and restart

At a quantum boundary the manager opens the contiguous sealed prefix, validates layout and structural headers,
appends each shard directly, and flushes once. Normal trusted-boundary opens avoid a redundant whole-shard hash pass.
Zero-row shards still advance the append sequence and transaction identity without inventing rows.

After the replay flush, resignation observations are applied through their identity-idempotent SQLite sink, committed
claims are atomically removed from the queue, and shard/inbox leftovers are deleted. On startup, queued claims below
the store append sequence are recognized as committed cleanup leftovers; sealed uncommitted claims remain appendable
and unsealed claims are resubmitted.

The design intentionally does not infer torn header states or journal every boundary instruction. A process crash
during append may lose or duplicate the small in-flight shard set. Invalid store headers fail clearly. This bounded
risk is preferable to carrying a larger transaction and reporting-receipt subsystem for a run that restarts rarely.

Schema 4 is intentionally incompatible with earlier replay files. Resuming an older run requires the blocked,
offline-only [schema-4 migration procedure](../operations/replay-schema4-migration.md); it must never be attempted
against a running experiment.

Credits are reconciled from the absolute materialized total: committed store rows plus unique sealed uncommitted shard
rows. Completion callbacks never add row deltas. Reporting metadata is accumulated in memory between training quanta;
losing a small amount on an unclean restart is accepted.

## Vectorized training path

Each rank retains the existing deterministic global sampling RNG calls and slices its local 256 indices. The loader
then converts logical indices once, gathers columns in bulk, decodes packed states by vectorized NumPy operations,
applies cached validated plane/action permutations, and constructs the existing dense policy, legal, WDL, and
auxiliary targets through vectorized scatter. The dense objective and sampling distribution remain authoritative.

Prefetch uses one preparation worker and a bounded FIFO queue. CUDA pinned slots have explicit free, filling, ready,
transfer-in-flight, and reusable ownership. Completed events are queried before blocking; if all slots are busy, the
oldest transfer is synchronized. Consumer streams wait on transfer events and record tensor ownership before slots can
be reused. The configured default depth is four; benchmark inputs support depths 1, 2, 4, and 8.

Production hot loops contain no per-batch clocks or profiling callbacks. Focused benchmark tools own stage timing.
Sparse policy targets remain an isolated experiment and cannot replace dense targets without forward, gradient, and
end-to-end performance evidence.

## Required invariants

- Every completed game is processed in deterministic FIFO order during normal operation without skipping.
- Shard and replay order is independent of process completion order.
- Shard/store layout digests must match exactly before mutation.
- A shard is complete only when its final typed manifest validates.
- Sealed uncommitted and committed-cleanup shard states have simple restart handling.
- A dirty boundary restart may lose or duplicate a small bounded number of games; no exact recovery journal is kept.
- Chess and Go share the representation and lifecycle; game contracts own only representation-specific transforms.
- The eight-rank DDP topology, self-play worker count, and training pause topology are unchanged.
