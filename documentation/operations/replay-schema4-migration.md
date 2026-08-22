# Replay schema-4 offline migration

Status: blocked pending user approval, converter implementation, and validation.

No replay converter has been implemented or run for this migration. No production replay, completed-game directory,
credit ledger, receipt, queue, process, or configuration has been modified. The currently running experiment must
remain untouched. This procedure becomes executable only after the user approves a quiesced snapshot or maintenance
window and the proposed tools pass the validation described below.

## Compatibility boundary

Schema 4 is intentionally incompatible with earlier replay files. It is a versioned columnar store whose header and
column descriptors are bound to the complete `ReplayLayout` digest. The schema-4 runtime must reject an earlier
`replay.bin`; it must not reinterpret, resize, or repair one in place.

A lossless schema-3 conversion is possible only when the source descriptor and run configuration contain enough
information to reconstruct every configured schema-4 column exactly. Before authorizing conversion, an audit tool
must prove all of the following:

- the source revision, resolved experiment configuration, game, board/action dimensions, target layout, packed-plane
  layout, policy retention widths, legal-action width, scalar dtypes, and source schema are known;
- every live source row has the complete primary and auxiliary targets required by the destination `ReplayLayout`,
  including exact legal-action lists and eligibility values;
- packed state bytes and all narrow integer fields can be converted without truncation, inference, or changed
  validation;
- source FIFO geometry and counters are readable independently of Python object deserialization; and
- no target field would need to be guessed from policy support, reconstructed by replaying unavailable game history,
  or filled with a synthetic default.

Local schema-1 snapshots that omit legal-action lists do not meet these requirements. Policy support is not the legal
move set. Filling legal lists from policy entries, as an earlier local supervised-testbed conversion did, is an
approximation and is not eligible for training-resume migration. Such snapshots may remain explicitly labelled
research inputs, but they cannot be activated as a schema-4 production replay. A file called “schema 3” is not
sufficient evidence by itself; the descriptor audit decides whether conversion is lossless.

## Preconditions and snapshot

Migration is offline. The approved procedure must identify the exact source revision and use that revision's reader
to interpret the old store. Do not open the live replay with the new runtime.

1. Obtain explicit user approval for a maintenance window or an immutable, application-consistent snapshot. Record
   the run path, source revision, resolved configuration SHA-256, checkpoint generation, replay source SHA-256, and
   snapshot timestamp.
2. Quiesce the approved copy at a quantum boundary. No self-play publisher, materializer, coordinator, trainer, or
   reporting process may write to it. A filesystem copy taken while those processes are active is not a valid
   snapshot unless the storage platform provides an atomic snapshot of the complete run directory.
3. Inspect the old append/recovery state with the old revision. A pending append must be recovered by that revision
   on the offline copy, or migration must stop. Do not synthesize completion from file names or row counts.
4. Inventory `replay.bin`, the credit ledger, completed-game inbox, staged files,
   resignation state, checkpoints, and reporting artifacts. Persist this inventory and checksums beside the snapshot.
5. Compute the schema-4 destination size with the exact destination `ReplayLayout` and maximum capacity. Available
   space on the conversion filesystem must cover the immutable source snapshot, the complete destination file,
   conversion scratch/manifests, and at least 20% additional headroom. If the source snapshot is retained on another
   verified filesystem, local free space must still cover the destination, scratch space, and headroom.

The source snapshot is read-only from this point onward. All trials write to a separate conversion directory on the
same filesystem intended for activation.

## Legacy staged files

Schema-4 startup deliberately rejects legacy `staging/*.rows.npy` and `staging/*.meta.json`. They must never be
deleted merely to make startup pass.

The preferred prerequisite is a source-revision boundary that has durably appended all complete staged pairs before
the approved snapshot. If legacy files remain, the migration tool must inventory them in deterministic established
order and require one valid row/metadata pair per game. It must preserve the game's identity, row span, termination
and resignation metadata, observations, and policy-mass telemetry while lowering the rows into canonical schema-4
columns. Inbox duplicates may be removed only after their identity and content are proven to match a durably converted
game. A missing pair, orphan non-temporary file, identity disagreement, or metadata that cannot populate the typed
schema-4 shard manifest blocks migration.

Temporary files may be ignored only after they are identified as unsealed old-runtime temporaries on the offline
copy. The inventory must retain their names and checksums. There is currently no validated legacy-staging converter,
so a snapshot with complete legacy staged games is not deployable yet.

## Deterministic conversion

The converter must create a new file; it must never update the source in place.

1. Construct the destination `ReplayLayout` from the exact approved destination revision and resolved configuration.
   Record its digest and full ordered column descriptors.
2. Read and validate the source header and descriptors with the source revision. Record maximum capacity, logical
   capacity, head, size, total appended rows, evicted rows, and every available transaction counter.
3. Traverse live rows in FIFO order, oldest to newest, using the source physical mapping. Convert bounded chunks
   directly into schema-4 column arrays and write each destination column slab without creating per-row domain
   objects or concatenating the complete replay in memory.
4. Preserve maximum capacity, logical capacity, head, live size, total appended rows, and evicted rows exactly.
   Preserve source append/transaction counters when they exist. When schema 3 lacks a schema-4 append sequence, the
   converter must create one documented deterministic migration-baseline transaction identity and sequence; it must
   not claim that this newly introduced counter existed in the source.
5. Initialize a schema-4 shard queue with the destination layout digest and `next_sequence` equal to the converted
   store's append sequence. It must contain no fabricated claims. Convert any approved legacy staged pairs through a
   separately validated shard append after the baseline, not by changing baseline counters.
6. Preserve the credit ledger. Validate that absolute materialized rows account for earned credits and that consumed
   credits remain legal. Do not add callback deltas or reset credits.
7. Preserve the resignation journal and verify its game identities against migrated metadata.

Chunk size is an operational parameter, not a semantic one. Repeating conversion with different chunk sizes must
produce identical headers, logical column checksums, queue state, and migration manifest.

## Validation gate

The conversion tool must fail before activation unless all checks pass:

- source and destination revision/configuration hashes and destination layout digest equal the approved values;
- source and destination FIFO geometry and total/live/evicted counters agree exactly;
- packed decoded states are byte-identical for every live row;
- primary policy, legal actions, WDL, root value, sample weight, source generation/timestamp, and every configured
  auxiliary target are bitwise equal after lossless dtype conversion;
- per-column SHA-256 digests in logical FIFO order match an independently generated expected manifest;
- deterministic sampled rows cover the head, tail, wrap boundary, repeated physical pages, and every auxiliary target;
- dense targets and all supported augmentations are equivalent for chess or Go as selected by the configuration;
- a read-only schema-4 open passes complete header, descriptor, padding, dtype, and semantic validation;
- a copied credit ledger reconciles to the converted absolute materialized total without changing earned or consumed
  credit; and
- an isolated restart smoke on the converted copy opens the store and queue without errors.

Record tool version, commands, stdout/stderr, elapsed time, peak memory, source and destination file SHA-256 digests,
the logical per-column manifest, and the complete validation result. Two conversions of the same snapshot must have
identical semantic manifests even if filesystem allocation bytes differ.

## Activation and rollback

Activation requires a second explicit user decision after reviewing the conversion evidence. Keep the source
snapshot and its checksums until the converted run has passed the approved canary and rollback window.

The activation tool must prepare the complete replay subsystem in a sibling directory, fsync every file and both
directories, and write its typed completion manifest last. It must verify that the target remains quiesced and still
matches the snapshotted source hashes. Activation then uses a same-filesystem atomic exchange of the prepared and
active run directories, or another prevalidated atomic pointer switch. If the platform cannot provide that primitive,
deployment remains blocked; a sequence of unrelated file replacements is not an atomic activation plan.

After activation, run only the approved read-only open/recovery smoke before allowing producers or training to start.
Rollback atomically switches back to the untouched source directory while everything remains stopped. Never copy
rows backward from schema 4 into the old store. Preserve the failed destination, logs, and manifests for diagnosis.

## Planned tools and commands

The names below are interface placeholders, not commands that are currently available. Do not run them until the
tools exist at the approved revision and their focused tests pass.

```powershell
# NOT IMPLEMENTED: audit source completeness without writing.
uv run python .\tools\audit_replay_schema3.py --run-directory <offline-snapshot> --output <audit.json>

# NOT IMPLEMENTED: create a new schema-4 conversion directory.
uv run python .\tools\convert_replay_schema3_to_schema4.py --source <offline-snapshot> --destination <prepared-run> --resolved-configuration <resolved.yaml> --manifest <conversion.json>

# NOT IMPLEMENTED: independently validate the prepared copy.
uv run python .\tools\validate_replay_schema4_migration.py --source <offline-snapshot> --destination <prepared-run> --manifest <conversion.json> --output <validation.json>

# NOT IMPLEMENTED: atomically activate only after a separate user approval.
uv run python .\tools\activate_replay_schema4_migration.py --active <quiesced-run> --prepared <prepared-run> --validation <validation.json> --approval <approval.json>
```

Until those tools, tests, conversion evidence, atomic activation mechanism, and approvals exist, schema-4 deployment
against an earlier replay is blocked. A fresh run may use schema 4 only through the ordinary revision-bound run
approval process; this document does not authorize launching one.
