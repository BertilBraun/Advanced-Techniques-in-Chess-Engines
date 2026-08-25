# Columnar replay and shard ingestion

Status: accepted implementation design, 2026-08-22. Its materialization dispatch was replaced on 2026-08-25 by
the per-worker-directory pipeline described below; the analysis behind that replacement is
`replay-materialization-rework.md`. The columnar store, shard file format and boundary-append rules are unchanged.

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

1. Self-play atomically publishes one typed completed-game JSON file into `completed-games/inbox/`.
2. A dispatcher pass renames each inbox game into a per-worker directory, round-robin, prefixing a fifteen-digit
   per-worker counter. The pass reads at most `materialization_inbox_rename_cap` entries from a lazy `scandir`, so its
   cost is independent of inbox depth and of how many games were already ingested. A game exists at exactly one path
   at a time, so `rename(2)` alone prevents double ingestion; there is no queue, no hash and no timestamp cache.
3. One long-lived process per worker index owns one directory for the life of the run and consumes it in counter
   order. It takes a bounded batch, materializes each game independently, and encodes samples into replay columns.
4. The worker flushes and atomically seals one uncompressed shard data file, writes its typed manifest last, and only
   then removes the consumed sources. The shard identity is `sha256(layout digest, worker index, first counter, last
   counter)`, so a worker killed between sealing and unlinking re-derives the same identity on restart and adopts the
   existing shard instead of producing a duplicate.
5. Order across shards is approximate by decision: eviction is positional and sampling uniform, so round-robin
   dispatch plus per-worker FIFO is sufficient. Order within a shard is exact and follows the dispatch counter.

The default bounds are 32 games and a soft 16 MiB of source JSON per shard. An indivisible oversized game becomes a
singleton. Any game that cannot be read, parsed or materialized is moved to `completed-games/rejected/` and the rest
of its batch still seals; a shard that fails to encode or seal rejects its whole batch. Nothing about bad game data
stalls or kills the run. A rolling rejection-rate alarm
(`materialization_rejection_window_games`, `materialization_rejection_rate_ceiling`) fails the run loudly if discarding
becomes systematic. Worker backpressure is `materialization_staging_shard_limit` sealed shards in staging.

## Boundary append and restart

At a quantum boundary the manager opens the contiguous sealed prefix, validates layout and structural headers,
appends each shard directly, and flushes once. Normal trusted-boundary opens avoid a redundant whole-shard hash pass.
Zero-row shards still advance the append sequence and transaction identity without inventing rows.

At a boundary every sealed shard in staging is appendable; there is no sequence cursor and no head-of-line block.
After the replay flush, resignation observations are applied through their identity-idempotent SQLite sink and the
appended shard files are deleted. Training credit is reconciled from `total_appended_rows` strictly after that flush:
late credit costs one coordinator iteration, early credit would let the ledger over-earn against the store. On
startup, games left in a worker directory are re-materialized in the same counter order, sealed shards in staging are
appended by the normal path, and per-worker counters are reseeded from the highest prefix present. A worker directory
left behind by a reduced `materialization_processes` has its games renamed back into the inbox.

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

- Every completed game is seen exactly once by the dispatcher, because it is renamed out of the inbox on first sight.
- Ingestion order is approximate across shards and exact within a shard.
- A game that cannot be materialized is discarded individually and quarantined, never stalling the pipeline.
- A systematic rejection rate fails the run instead of silently discarding every game.
- Shard/store layout digests must match exactly before mutation.
- A shard is complete only when its final typed manifest validates.
- Sealed uncommitted and committed-cleanup shard states have simple restart handling.
- A dirty boundary restart may lose or duplicate a small bounded number of games; no exact recovery journal is kept.
- Chess and Go share the representation and lifecycle; game contracts own only representation-specific transforms.
- The eight-rank DDP topology, self-play worker count, and training pause topology are unchanged.
