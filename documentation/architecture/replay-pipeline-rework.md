# Columnar replay and exact shard ingestion

Status: accepted implementation design, 2026-08-22.

This decision supersedes the approximate replay-recovery policy in
`python-runtime-rework.md`. Completed-game rows, replay credits, and downstream reporting metadata are now recovered
exactly once. The producer-facing completed-game JSON contract and per-game atomic inbox publication remain unchanged.

## Canonical representation

`ReplayLayout` is the sole semantic schema. Its digest covers the packed state representation, target layout,
retention widths, ordered column descriptors, shapes, and dtypes. Materialization shards, the live replay store, and
the mapped loader use those same descriptors and `ReplayColumnViews`; they do not define row DTOs or adapters.

The live replay is one preallocated schema-4 binary file. It has a fixed 64 KiB header and aligned fixed-capacity
column slabs. The header records maximum and logical capacities, FIFO head and size, total appended and evicted rows,
append sequence, and the last transaction identity. Logical indices are converted to physical indices once and whole
columns are gathered with NumPy indexing. Append plans copy each source column into at most two ring slices.

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

## Boundary transaction and recovery

At a quantum boundary the manager opens the contiguous sealed prefix, validates layout and structural headers, plans
one append transaction per shard, and writes an append-recovery manifest before copying. Normal trusted-boundary opens
avoid a redundant whole-shard hash pass; startup recovery verifies full shard hashes.

The store prevalidates the complete plan chain and every source column before mutation. It reapplies every planned
destination from the recorded starting geometry, writes linked intermediate headers, reaches the exact final state,
and flushes once. Reapplication repairs partial data even when the final header was already persisted. Zero-row shards
advance the append sequence and transaction identity without inventing rows.

After the replay flush, resignation observations are applied through their identity-idempotent SQLite sink and one
durable ingestion receipt is written. Only then are shard files, leftover matching inbox files, durable claims, and the
append manifest removed. Startup repeats any incomplete suffix safely. Capacity changes persist a queue-owned resize
record so evictions remain attributable across a crash between resize and append.

Schema 4 is intentionally incompatible with earlier replay files. Resuming an older run requires the blocked,
offline-only [schema-4 migration procedure](../operations/replay-schema4-migration.md); it must never be attempted
against a running experiment.

Credits are reconciled from the absolute durable total: committed store rows plus unique sealed uncommitted shard
rows. Completion callbacks never add row deltas. Reporting receipts are deduplicated by identity, replayed after
restart, and acknowledged only after training reporting succeeds.

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

- Every completed game remains in exactly one durable lifecycle state and is eventually ingested without skipping.
- Shard and replay order is independent of process completion order.
- Shard/store layout digests must match exactly before mutation.
- A shard is complete only when its final typed manifest validates.
- Reapplying a recovery plan is byte-idempotent for its planned destinations and final state.
- Rows, append sequences, credits, resignation evidence, and reporting receipts cannot be duplicated or lost.
- Chess and Go share the representation and lifecycle; game contracts own only representation-specific transforms.
- The eight-rank DDP topology, self-play worker count, and training pause topology are unchanged.
