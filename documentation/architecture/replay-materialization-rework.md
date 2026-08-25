# Replay materialization rework — analysis

Status: analysis, not an accepted design. 2026-08-25.
Base revision: branch `replay-claim-skip-bad-sources`, commit `09986f20`. All `py/src/replay/manager.py` line
numbers below are **post-`09986f20`**; the pre-fix file differs by ±8 lines in `_allocate_claims`.
Supersedes nothing yet. `documentation/architecture/replay-pipeline-rework.md` is the currently accepted design and
stays authoritative until the owner accepts a replacement.

This document answers five questions: what every mechanism in the current pipeline defends against, which invariants
are real, how the owner's proposed design works out in detail, what migrating costs, and what the simplification
risks. It does not change code.

---

## 0. Summary of conclusions

1. **Global FIFO order across shards is not required.** Eviction is positional in a ring buffer
   (`store.py:639-643`), sampling is uniform over the whole live window (`batch_loader.py:129-133`), and the two
   recency columns are read only by TensorBoard telemetry (`distributions.py:93-94`). Approximate age ordering is
   sufficient. This removes the largest single justification for the current machinery.
2. **The credit ledger needs eventual consistency, with one direction constraint:** credit may be late, never early.
   Crediting on *append* instead of on *seal* satisfies this and deletes the entire sealed-callback path.
3. **The replay store, its file format and its schema-4 header are untouched by the rework.** An existing run's
   `replay.bin` continues without a clean restart. Only the code that feeds `append_columns` changes.
4. **Duplicate ingestion is prevented by construction in the proposed design** — a game file exists at exactly one
   path at a time and `rename(2)` is atomic — which is strictly stronger than today's lock-guarded name sets.
5. **The owner's design as stated needs four additions to stay correct** (§3.7): deterministic re-seal identity,
   credit-after-append, an early-break bounded `scandir`, and a rejection-rate alarm. Everything else is sufficient
   as described.
6. **Measured**: `InboxScanner.invalidate` is O(n) per call with a full tuple rebuild; a full `_allocate_claims`
   pass at a 100k inbox costs ~15 s of pure Python and I/O and is thrown away wholesale if the commit block bails
   (§5.2). This is a sufficient explanation for the observed symptoms without needing a further trigger.

---

## 1. Inventory: every mechanism and what it defends against

### 1.1 `ReplayShardQueue` and `validate_pending` (`manager.py:76-99`)

A pydantic model persisted to `completed-games/shard-queue.json`. Its validator enforces four things:

| Check | Lines | Defends against |
| --- | --- | --- |
| Sequences unique and increasing | 84-86 | A corrupted or hand-edited queue file |
| Sequences contiguous and ending at `next_sequence - 1` | 87-90 | A gap that would permanently stall `_contiguous_sealed_claims` |
| `archive_key` and `file_name` globally unique across all pending claims | 91-96 | The same game landing in two shards, i.e. duplicate rows in the store |
| Order keys globally sorted | 97-98 | Cross-shard FIFO violation |

**If deleted:** the first two checks lose nothing that matters once the global sequence itself is gone (§2.1). The
uniqueness check is the only load-bearing one, and in the proposed design it is enforced by the filesystem instead:
a game lives at exactly one path, and the rename that moves it is atomic. The global-order check enforces an
invariant that §2.1 shows is not required.

Note the cost: this validator runs on **every** `ReplayShardQueue(...)` construction, i.e. on every
`_allocate_claims` commit (`manager.py:554`) and every `_finalize_committed_claims` (`manager.py:672`). It is
O(total pending sources) with several tuple builds and set constructions. At 32 claims × 32 games it is ~1024
`archive_key` property calls (each an f-string) per pass. Not the wedge, but not free.

### 1.2 `_allocate_claims` (`manager.py:469-565`)

The single largest and most fragile function in the pipeline. It does, per pass:

- snapshots the queue and computes `claimed` and `last_claimed_key` (474-478) — O(pending sources);
- calls `inbox_files_by_modification_time()` and filters it into a new tuple (479) — O(inbox depth), builds a
  full tuple of `Path` objects even when only 32 shards will be claimed;
- computes `claim_slots` (480);
- **outside the lock**, batches candidates: `stat()` each (492), possibly `os.utime` + `invalidate` (496-499),
  bound by game count and source bytes (501-505), parse the identity from the filename (506), and
  **SHA-256 the whole file** (511);
- **inside the lock**, re-checks the snapshot (526), re-`stat()`s every batched source (536), and commits.

**Defends against:** claiming a game twice; claiming a game that changed on disk between the scan and the commit;
claiming out of FIFO order; unbounded in-flight work.

**If deleted:** everything it defends against is either not required (FIFO order, §2.1), impossible by construction
in the new design (double-claim, §3.5), or already precluded upstream (source mutation — a completed game is
published by `write_bytes_atomically`, `atomic_file.py:12-23`, so the final name never appears until the content is
complete and fsynced, and nothing ever writes to that name again).

The optimistic-snapshot pattern at 473/526 is itself a hazard: the expensive work (up to 1024 SHA-256 reads) is done
outside the lock and then discarded if the coordinator thread committed an append in the meantime. There is no
progress guarantee. With a slow enough pass and a fast enough appender, this starves permanently.

### 1.3 `_claim_slots` (`manager.py:567-575`)

`min(materialization_limit - unsealed, staging_limit - outstanding)` with
`materialization_limit = max(32, 2 × processes)` and `staging_limit = 3 × materialization_limit`.

**Defends against:** (a) submitting more shards than the pool can chew, and (b) unbounded growth of `staging/` when
the appender stalls. The comment at 572-573 records a previous bug: counting sealed-but-unappended shards as
in-flight idled every worker during an append.

**If deleted:** (a) is not a real hazard — a `ProcessPoolExecutor` queues submissions, it does not blow up. (b) is
real but small: the appender runs on every coordinator loop iteration (`coordinator.py:125`) and self-play is already
throttled by credit backpressure (`coordinator.py:314-319`), not by shard slots. A single
`len(os.listdir(staging))` guard in the worker replaces it (§3.7).

### 1.4 `_contiguous_sealed_claims` (`manager.py:645-652`)

Walks from `store.state.append_sequence` upward while the next sequence is present and sealed.

**Defends against:** appending shards out of their assigned global order.

**If deleted:** nothing breaks, *given* §2.1. This is also a head-of-line block: one shard that fails to seal stalls
every later shard indefinitely, and `test_missing_earlier_sealed_sequence_blocks_later_shard_without_changing_totals`
(`test/test_replay_manager.py:978`) asserts exactly that behaviour. Under the new design that test inverts.

### 1.5 `InboxScanner` and its timestamp cache (`manager.py:747-793`)

Intended to avoid re-`stat()`ing the whole backlog every second. It keeps `_modified_at_by_name`, `_ordered_keys`
and `_ordered`, and takes a fast path returning the cached tuple when nothing arrived or departed (775-776).

**Two defects, both measured:**

- `_modified_at_by_name` is **never pruned for departed names** (compare 773-774 with 785: `departed` is used to
  filter `_ordered` but never removed from `known`). Therefore `departed = known.keys() - names` is non-empty
  forever after the first game is consumed, the fast path at 775 **never fires again for the rest of the run**, and
  every scan pays the full `heapq.merge` and both list rebuilds. Measured on a 4,000-file directory: steady-state
  scan 11.7 ms; after one single deletion, 19.2 ms and never faster again, with `known` holding 4,000 entries for
  3,999 files. The dict also grows for the whole life of the run.
- `invalidate` (757-762) is **O(n) twice**: `list.remove` on `_ordered_keys` plus a full tuple comprehension
  rebuilding `_ordered`. Measured per-call cost: 0.45 ms at n=2,000; 0.95 ms at n=4,000; 1.80 ms at n=8,000;
  5.68 ms at n=16,000. Extrapolated to n=108,000: ≈38 ms per call.

**If deleted:** nothing is lost. The proposed dispatcher needs no ordering and no cache at all.

### 1.6 The `os.utime` re-queue branch (`manager.py:493-500`)

If a candidate's `(mtime_ns, name)` key is not strictly greater than the newest already-claimed key, the file's
mtime is bumped to now and it is pushed back to the tail of the FIFO.

**Defends against:** a real artefact of atomic publication. `write_bytes_atomically` stamps the mtime when the
*temp* file is written and only then `os.replace`s it into position (`atomic_file.py:16-21`). A large game whose
write and fsync are slow can therefore appear in the inbox with an mtime older than files that appeared before it.
Without this branch, `PendingReplayShardManifest.create` → `_validate_source_games` (`shard.py:426-432`) would raise
on the unsorted key, and that raise propagates to `_set_fatal_materialization_error`.

**If deleted:** with mtime ordering gone (§2.1) the hazard disappears entirely. This branch is the tail of a
dependency chain that begins with a requirement that is not real.

**It is also an O(n²) landmine.** Each iteration calls `invalidate` (498), which costs O(inbox depth). If a
significant fraction of the inbox is behind `last_claimed_key`, one `_allocate_claims` pass is O(n²): at
n = 108,000 that is ≈38 ms × 108,000 ≈ **68 minutes for a single pass that commits nothing and raises nothing.**
That is a precise match for "spun at 100% CPU in pure Python for 45+ minutes". It is *not* the mechanism behind the
observed wedge, because the frozen queue showed `pending: []` and therefore `last_claimed_key is None` (475-478),
which disables this branch — but it is a live second landmine and I flag it explicitly.

**`09986f20` made this worse, not better.** It added two new `invalidate` call sites in the commit block
(`manager.py:538` and `546`), reachable up to 1,024 times per pass. At n = 108,000 that is ≈39 s of pure Python per
dispatch pass, at a 1 s poll interval, on the *fixed* branch. If the owner runs that branch in production against a
large inbox it will exhibit the same symptom. This should be addressed before the branch is used, independently of
the rework.

### 1.7 The SHA-256 of every source file (`manager.py:511`, `739-744`; verified at `parallel_materialization.py:140-146`)

**Defends against:** a source file whose content changed between claim and materialization, and against a leftover
inbox copy of an already-committed game that is *not* byte-identical (`_validate_leftover_sources`,
`manager.py:682-688`).

**If deleted:** the hazard it guards is already precluded by atomic publication (§1.2) and by
`publish_completed_self_play_game` refusing to overwrite an existing identity (`completed_game.py:184-185`). What is
genuinely lost is detection of an operator or a stray process mutating a file in the inbox. That is an acceptable
loss.

**Cost, measured**: SHA-256 of 1,024 × 200 KB source files took **14.8 s** in one pass on a local Windows disk. On
the production node with warm page cache it will be faster, but at a 108,000-file backlog the files are cold. This
is redone from scratch every pass for the same files whenever the commit block does not commit.

### 1.8 The sealed / unsealed distinction (`_is_sealed`, `manager.py:613-614`)

Sealed means `staging/<identity>.replay-shard.json` exists. `write_replay_shard` writes the data file first via a
temp + `os.replace` + `fsync_directory` (`shard.py:338-344`) and the manifest last (`shard.py:346`), so the manifest's
existence is the durable commit point of a shard.

**Defends against:** appending a half-written shard.

**If deleted:** the run corrupts. **This is a genuinely required invariant and must be kept exactly as is.**

### 1.9 `_retry_after` and `_notified` (`manager.py:108-109`, `203`, `213`, `224`, `234-236`)

`_retry_after` backs off a claim after a transient materialization failure. `_notified` prevents the sealed callback
firing twice for one sequence.

**Defends against:** a hot retry loop on a transient error, and double-crediting.

**If deleted:** the retry backoff becomes unnecessary because there is no durable claim to retry — a failed game is
dropped and a failed shard's games stay in the worker directory to be re-attempted on the next pass, naturally
paced. `_notified` disappears with the callback (§3.6).

### 1.10 The fatal-error flag (`_fatal_materialization_error`, `manager.py:279`, `593-596`, `423-427`)

Set from six places, checked at the top of the coordinator loop (`coordinator.py:121`) and from
`total_materialized_samples` (`manager.py:352`).

**Defends against:** silently continuing after data that cannot be materialized.

**If deleted / narrowed:** this is the mechanism the owner is explicitly overruling — "nothing about a bad game may
ever stall or kill the run". Two of its six triggers are game-data-driven and must go:
`_record_failure`'s `ValueError | FileNotFoundError` branch (228-232) and `_allocate_claims`' claim-order error
(513-514, 562-565). The other four are structural (executor submission failure, dispatcher loop failure, callback
failure, layout mismatch) and should be **kept**: a broken process pool is not a bad game. See §5.3 for the
replacement signal that must exist so that *systematic* materialization failure is still loud.

### 1.11 `ProcessPoolExecutor` and the inline-staging fallback (`manager.py:284-292`, `153-177`, `197-204`)

The pool exists because `materialize_completed_game` needs a live `GameStateContract` and a `TerminalOracle` — a
native game object built per worker process by `initialize_materialization_worker`
(`parallel_materialization.py:153-161`). The inline path exists so `materialization_processes == 1` works without a
pool (and so tests can run in-process).

**Defends against:** the GIL. Materialization is CPU-bound Python plus native calls.

**If deleted:** materialization serializes behind the coordinator. The pool must stay. **But `ProcessPoolExecutor`
is the wrong primitive for the proposed design** — see §3.8.

### 1.12 The store's transaction identity (`store.py:353-356`, `_TRANSACTION_IDENTITY_BYTES = 64`)

`append_columns` records the last transaction identity in the store header and returns without appending if the same
identity is presented again with a matching row count.

**Defends against:** re-appending the shard that was in flight when the process died between the `append_columns`
write and the queue-file update.

**If deleted:** an unclean restart duplicates rows. **Keep it** — but note it is only *one deep*. It protects the
most recent append, not a batch of them. §3.4 explains why that is still adequate.

---

## 2. Which invariants are real

### 2.1 Global FIFO order across shards — **not required**

Answered from the code, not from the design document:

- **Eviction is positional, not temporal.** `_append_state` (`store.py:634-650`) computes
  `evicted = max(0, size + row_count - logical_capacity)` and advances `head` by that many slots. There is no
  timestamp comparison anywhere in the eviction path. `set_logical_capacity` (`store.py:316-325`) evicts from the
  head by the same rule. "Oldest" means "earliest written into the ring", nothing else.
- **Sampling is uniform over the whole live window.** `generator.choice(self.replay.size, size=global_batch_size,
  replace=False)` (`batch_loader.py:129-133`). No recency weighting, no age-stratified sampling, no priority.
- **The two recency columns are telemetry only.** `source_model_generation` and `source_timestamp` are written by
  `encoding.py:128` / `materialization.py:193`, carried through `batch.py:26-27`, and consumed in exactly one place:
  `distributions.py:93-94`, which computes `generation_ages` and `replay_ages` for the TensorBoard histograms. No
  loss term, no sampling weight, no gate reads them. I grepped the whole of `py/src` for both names to confirm.

**Therefore:** what the store needs is that the live window approximates "the most recent N samples produced". With
16 workers × 32 games per shard, worst-case reordering is ~512 games ≈ tens of thousands of samples against a
capacity typically in the millions — a sub-1% perturbation of the eviction boundary, invisible to a uniform sampler.
The age histograms acquire slightly wider tails.

**One caveat that does need handling.** Blind round-robin lets a slow worker's games lag arbitrarily. If worker 7
stalls for an hour, its hour-old games are eventually appended into an otherwise fresh window. Uniform sampling does
not care about *position*, but it does care about *staleness*. §3.2 recommends depth-aware dispatch rather than blind
round-robin; that is one `len(os.listdir)` per worker per pass and makes worker lag self-correcting.

**Order *within* a shard is required** — `SealedReplayShardManifest.validate_identity_order_and_spans`
(`shard.py:135-141`) requires each game's `row_start` to be exactly the running row total, and `ReplayShardReader`
slices the columnar file on those spans. Consuming a worker directory in numeric filename order satisfies this
automatically. Keep the span check; delete the `(mtime_ns, file_name)` key check at `shard.py:426-432`.

### 2.2 The credit ledger — **eventual consistency, with a direction constraint**

- `has_quantum_credits` (`credit_ledger.py:90-92`) is a threshold test, re-evaluated on every coordinator loop
  iteration. Credit arriving one iteration late delays training by ~1 s.
- `reconcile_materialized_samples` (`credit_ledger.py:110-120`) recomputes `earned_credits` **absolutely** from the
  total, so it is idempotent and self-correcting. Duplicate callbacks are harmless (that is what
  `test_training_coordinator_replay.py:101` asserts).
- **But it raises** if `earned_credits > total_materialized × replay_ratio` (`credit_ledger.py:115-116`). So the
  total must be **monotone non-decreasing**. Today `total_materialized_samples` (`manager.py:351-358`) counts
  `store.total_appended_rows` **plus sealed-but-unappended shard rows**. That second term can *decrease* if a
  sealed shard is ever discarded — which is exactly what "discard invalid work individually" would do.

**Conclusion: credit strictly after append + flush.** Then `total = store.total_appended_rows` alone, which is
monotone by construction, the ValueError becomes unreachable, and the sealed-shard callback path disappears
entirely. Late credit is safe; early credit is not.

### 2.3 Crash / restart resumability — what is genuinely needed

The node is ephemeral and restarts happen. The required properties, and where each comes from:

| Property | Today | In the proposed design |
| --- | --- | --- |
| A game already materialized is not materialized again | queue file + `_recover_directories` (`manager.py:697-736`) | the source file is deleted after seal (`parallel_materialization.py:93`) |
| A game not yet materialized is not lost | queue file names it; it is still in `inbox/` | it is still in `workers/worker-k/` |
| A sealed shard is not appended twice | `_contiguous_sealed_claims` cursor + store transaction identity | deterministic re-seal identity + store transaction identity (§3.7 A1) |
| The ledger never over-earns | absolute reconcile | credit after append (§2.2) |
| A shard is never appended half-written | manifest-written-last | unchanged |

The queue file's only unique contribution is the ordering it imposes, which §2.1 removes.
**`shard-queue.json` holds nothing that is not reconstructible from the filesystem and can be deleted outright.**

Note what is *already* accepted and unchanged by the rework: `append_columns` mutates an mmap header before
`flush()`, so a crash mid-append can tear the store header. `replay-pipeline-rework.md` accepts this explicitly. It
is a pre-existing risk, not one the rework introduces.

### 2.4 What prevents the same game being ingested twice — today

Six overlapping mechanisms:

1. `claimed` / `claimed_now` name sets (`manager.py:474`, `530`, `540`) — within-pass and cross-claim.
2. `ReplayShardQueue.validate_pending` global uniqueness (`manager.py:91-96`) — a durable backstop.
3. `_remove_consumed_inbox_games` (`parallel_materialization.py:126-137`) — the source is deleted after seal.
4. `store.append_columns` transaction identity (`store.py:353-356`) — one-deep replay protection.
5. `_recover_directories` (`manager.py:697-736`) — on restart, deletes inbox copies of already-committed games,
   after verifying their hash.
6. `publish_completed_self_play_game` (`completed_game.py:184-185`) — a producer cannot republish an identity.

Six mechanisms for one invariant is the clearest signal in this codebase that the design is over-defended. In the
proposed design, (1) and (2) are replaced by *the filesystem*: a game is at exactly one path, and `rename(2)` within
a filesystem is atomic — either the dispatcher moved it or it did not, and the loser of any race gets `ENOENT`.
(3), (4) and (6) survive unchanged. (5) collapses to "whatever is in `worker-k/` gets materialized", with no hash
check needed because nothing else could have consumed it.

---

## 3. The proposed design, worked through

### 3.1 Directory layout

```
completed-games/
  inbox/                                   producers only; dispatcher drains it
  workers/worker-0/ … worker-{N-1}/        one owner each; dispatcher writes, worker k reads
  rejected/                                games that could not be materialized (see §5.3)
  staging/                                 sealed shards, unchanged format
```

`inbox/`, `workers/*` and `staging/` must be on **one filesystem** so `rename(2)` is O(1). They are all under
`run_path/completed-games`, so this holds unless someone mounts a subdirectory. Worth an assertion at startup
(`os.stat().st_dev` equality) because a cross-device rename silently degrades to copy+unlink and would reintroduce
exactly the I/O cost the design is trying to remove.

### 3.2 Filename scheme and the dispatcher loop

Rename target: `workers/worker-{k}/{counter:015d}-{original_name}`.

The original name is preserved verbatim after the prefix, so `GameIdentity.from_file_name` still parses it
(strip the 16-character prefix first) and the archive key is recoverable for resignation calibration. Fifteen digits
zero-padded means lexicographic `sorted()` equals numeric order up to 10^15 games per worker, which is not a real
limit. The counter is per-`k`, in memory, seeded at startup from the maximum existing prefix in `worker-k/` plus one.

Dispatcher pass (once per second, or on a condition variable):

```
for entry in islice(os.scandir(inbox), RENAME_CAP):     # no stat, no hash, no sort
    if not entry.name.endswith('.json'): continue
    k = worker with the fewest pending files            # depth-aware, refreshed once per pass
    try: os.rename(inbox/entry.name, workers/k/f'{counters[k]:015d}-{entry.name}')
    except FileNotFoundError: continue                  # lost a race; nothing to do
    counters[k] += 1
```

Four points about this loop:

- **`islice` on the `scandir` iterator, not on a materialized list.** `os.scandir` returns a lazy iterator; breaking
  out of it early never reads the rest of the directory. This makes the dispatcher's per-pass cost **O(RENAME_CAP),
  fully independent of inbox depth**. Building `list(os.scandir(...))` first would keep the O(n) cost that the whole
  rework exists to remove. This is the single most important line in the design.
- **The `.json` suffix filter is load-bearing** and it is sufficient. Producers write
  `.{name}.{hex}.tmp` and `os.replace` into the final name (`atomic_file.py:14-21`), so a name ending in `.json` in
  the inbox is always a complete, fsynced file. No stat, no size check, no hash is needed to establish this.
- **Depth-aware worker selection instead of blind round-robin**, per §2.1's caveat. Sixteen `len(os.listdir)` calls
  per pass; negligible, and it prevents unbounded worker lag.
- **`FileNotFoundError` is the only expected error and it is a no-op.** No other failure mode of this loop can lose
  or duplicate a game.

`RENAME_CAP` should be generous (a few thousand) so a backlog drains in seconds, but finite so the dispatcher pass
always terminates.

### 3.3 The worker loop

One process per `k`, `k` fixed for the life of the run:

1. `names = sorted(e.name for e in os.scandir(workers/worker-k) if e.name.endswith('.json'))`
2. Take a prefix of up to `materialization_shard_maximum_games`, stopping early at
   `materialization_shard_target_source_bytes` (one `stat` per candidate — the only stat in the whole pipeline, and
   it is bounded at 32 per shard, not 108,000 per pass).
3. For each: `read_bytes`, `CompletedSelfPlayGame.model_validate_json`, `materialize_completed_game`. Wrap each game
   in `try/except Exception`: on failure, `os.rename` it into `rejected/`, increment a counter, log once, `continue`.
   **Never raise out of the per-game loop.**
4. If every game in the prefix was rejected: unlink nothing further, seal nothing, loop.
5. `encode_replay_columns` → `write_replay_shard` (unchanged) → unlink the consumed `worker-k/` files.
6. If `len(os.listdir(staging)) > STAGING_CAP`: sleep instead of sealing (replaces `_claim_slots`, §1.3).

Step 3 is the whole of "materializes the ones it can, skips the ones that are invalid". Note that
`materialize_completed_game` raises `ValueError` from six places (`materialization.py:70, 288, 292, 314, 316, 329,
338, 340, 343`) and `model_validate_json` raises on any schema violation; a blanket `except Exception` around one
game is the correct granularity. Losing one game — or the whole 32-game shard if the encode step fails — is within
the owner's stated tolerance.

### 3.4 How sealed shards reach the appender

`staging/` is scanned by the coordinator on every loop iteration. `append_staged_games` becomes:

```
for manifest_path in sorted(staging.glob('*.replay-shard.json')):
    reader = ReplayShardReader.open(manifest_path, layout, verify_data_hash=False)
    store.append_columns(reader.columns, reader.manifest.shard_identity)
store.flush()
observe_resignation(all metadata)
credit(store.total_appended_rows)          # after the flush
delete the appended shard files
```

No sequence, no cursor, no queue. The `sorted()` is cosmetic. `verify_data_hash=False` is retained for the reason
already recorded at `manager.py:617-618`.

Ordering of the three tail steps matters and should be stated as a rule: **flush, then credit, then delete.** A
crash between flush and credit under-credits and the next `reconcile_materialized_samples` corrects it upward. A
crash between credit and delete re-appends up to the whole batch on restart — but only the *last* one is caught by
the store's one-deep transaction identity (`store.py:353-356`), so the rest duplicate. That is within the accepted
"a dirty boundary restart may lose or duplicate a small bounded number of games", it does not break the ledger
(duplicates inflate `total_appended_rows` too), and resignation calibration is idempotent by `archive_key`
(`INSERT OR IGNORE`, `resignation.py:184-197`). **This is acceptable and does not need fixing.** The alternative —
one shard per `append_columns` + `flush` — would pay an msync per shard, and the comment at `manager.py:380-381`
records that msync costs ~0.2 s per gigabyte of mapping.

### 3.5 What happens on restart

- **Games in `inbox/`**: the dispatcher renames them. Nothing special.
- **Games in `workers/worker-k/`**: worker k re-materializes them, in the same numeric order. **Zero loss, zero
  duplication, zero recovery code.** This is the strongest property of the design and it depends entirely on `k`
  being a stable index rather than a pid or a pool slot.
- **Sealed shards in `staging/`**: appended by the normal path. No "orphan" concept exists any more, because there
  is no queue to be an orphan relative to.
- **Counters**: re-seeded from the max prefix per directory. One `scandir` per worker directory at startup.
- **The store**: opened exactly as today; its header is the sole source of truth for what has been ingested.

### 3.6 What the coordinator does

`start_materialization`, the `on_sealed` callback and `_reconcile_materialized_shard` (`coordinator.py:119`,
`177-179`) all disappear. Crediting moves inside `_append_staged_games` (`coordinator.py:181-190`), after
`append_staged_games` returns. `raise_if_materialization_failed` (`coordinator.py:121`) is retained but now only
fires on structural failures (§1.10). Nothing else in `coordinator.py` changes.

### 3.7 Where the owner's design as stated needs an addition

Four, all small:

**A1 — deterministic re-seal identity.** If a worker dies *after* `write_replay_shard` sealed the manifest but
*before* it unlinked the consumed sources (`parallel_materialization.py:92-93`), the restarted worker re-materializes
the same games into a shard with a fresh identity, and both shards get appended. Fix: derive the shard identity
deterministically from `(worker_id, first_counter, last_counter)` — e.g.
`sha256(f'{layout_digest}|{k}|{first}|{last}')`, which keeps the 64-lowercase-hex form that `_validate_sha256`
(`shard.py:457-459`) and the store's 64-byte transaction field (`store.py:55`) both require — and change
`write_replay_shard`'s "output already exists" raise (`shard.py:335-336`) into "already sealed: unlink the sources
and return the existing manifest". Five lines; removes the entire duplicate class. Note the identity is now
**cheap** to compute: no file hashing at all, unlike `replay_shard_identity` (`shard.py:245-255`) which hashes every
source's stat and SHA.

**A2 — credit after append, never on seal.** Per §2.2. Without this, discarding a sealed shard makes
`reconcile_materialized_samples` raise at `credit_ledger.py:116` and kills the run — which is precisely the failure
mode the rework is trying to eliminate.

**A3 — early-break bounded `scandir`.** Per §3.2. Without it the dispatcher stays O(inbox depth) per pass and the
rework does not actually remove the class of failure that caused the wedge; it only removes one instance of it.

**A4 — a rejection-rate signal.** Per §5.3. "Never stall on a bad game" and "never silently drop every game" are
both required, and only A4 provides the second.

### 3.8 One change to the mechanism, not the design

`ProcessPoolExecutor` (`manager.py:284-292`) is the wrong primitive here. The design's core property is that worker
`k` owns directory `worker-k` across restarts; a pool assigns tasks to arbitrary free workers and replaces dead ones
with new ones. Use `N` long-lived `multiprocessing.Process` instances with `k` passed at construction, each running
its own loop from §3.3 and reporting nothing back except a heartbeat. This also removes the
`submit`/`Future`/`_collect_finished` machinery (`manager.py:107`, `162-175`, `206-214`) and the inline-staging
fallback branch, since `N=1` is just one such process.

The owner is right that 16 workers is not expected to be the fix. It is, however, free once the directories are
per-worker, and it removes the pool's shared-failure mode where one worker's death poisons the executor.

---

## 4. Migration path

### 4.1 Does an existing `replay.bin` continue?

**Yes, without a clean restart.** Nothing in `store.py` changes: same magic `AZRPLY02`, same
`_REPLAY_SCHEMA_VERSION = 4`, same header dtype, same layout digest, same `append_sequence` /
`total_appended_rows` / `last_transaction_identity` semantics. `test_replay_store.py` (715 lines, 20 tests) should
pass **unmodified** — that is the sharpest available check that the store is untouched, and I recommend treating any
required change to that file as a signal that the rework has grown out of scope.

The shard file format also does not change (`_SHARD_MAGIC = AZRSHD01`, `_SHARD_SCHEMA_VERSION = 1`), so sealed
shards written by the current code remain readable by `ReplayShardReader` after the rework. Only
`SealedReplayShardManifest.sequence` becomes unused, and an unused field costs nothing.

### 4.2 One-shot changeover, run offline against a stopped run

1. Append every sealed shard in `staging/` using the **old** code path (they carry valid `sequence` values, so
   `_contiguous_sealed_claims` works), flush, and delete them.
2. Any claim in `shard-queue.json` that is *not* sealed: its source games are still in `inbox/` — sources are only
   deleted after seal (`parallel_materialization.py:93`). Nothing to do.
3. Delete `shard-queue.json`.
4. `mkdir completed-games/workers/worker-{0..N-1}` and `completed-games/rejected`.
5. Start the new code. The inbox drains normally.

`_recover_directories` (`manager.py:697-736`) already implements roughly steps 1–2 and can be lifted almost verbatim
into a ~40-line migration script. This is offline-only and must never run against a live experiment, same rule as
`documentation/operations/replay-schema4-migration.md`.

If a worker directory exists for `k ≥ materialization_processes` after a config change, its games are orphaned
forever. Add a three-line startup step that re-renames leftovers from high-`k` directories into low-`k` ones. Cheap;
without it, a config change silently loses games, which is worse than anything the current design does.

### 4.3 Delete outright

`manager.py`: `ReplayShardQueue` + `validate_pending` (76-99); `_QUEUE_FILE` (45); `_load_queue` / `_save_queue`
(449-467); `_allocate_claims` (469-565); `_claim_slots` (567-575); `_unclaimed_inbox_files` (577-580);
`_has_claim_capacity` (582-584); `_has_unsealed_claims` (586-591); `_claim` (598-599);
`_contiguous_sealed_claims` (645-652); `_validate_leftover_sources` (682-688); `_file_sha256` (739-744);
`InboxScanner` + `_first` (747-797); `_retry_after` / `_notified` (108-109 and users); `_MINIMUM_PENDING_SHARD_LIMIT`
/ `_STAGED_SHARD_LIMIT_FACTOR` (48-49); the whole `Future`/`_collect_finished` path (107, 162-175, 206-214).

`shard.py`: `InboxGameOrder` (34-44); `PendingReplayShardManifest` (58-85); the `order` / `source_size` /
`source_sha256` fields of `ReplayShardSourceGame` (47-55); `_validate_source_games` (426-432);
`replay_shard_identity`'s source-hashing body (245-255, replaced by A1); `pending_replay_shard_manifest_path`
(272-274, already unused by `manager.py`).

`parallel_materialization.py`: `_read_claimed_source` (140-146) and its two call sites (60, 136);
`stage_replay_shard_path`'s pending-manifest argument.

`coordinator.py`: `start_materialization` call (119); `_reconcile_materialized_shard` (177-179).

### 4.4 Keep for compatibility or correctness

`ReplayStore` in full; `write_replay_shard` / `_write_data_file` / `_validate_header` / `ReplayShardReader` /
`SealedReplayShardManifest` (minus `sequence`); `materialize_completed_game` and `encode_replay_columns` unchanged;
`initialize_materialization_worker` (`parallel_materialization.py:153-161`); the legacy-artifact guard
(`manager.py:697-702`) and the `.*.tmp` sweep (703-705); `_observe_resignation_games` (654-666); the store's
transaction identity; the manifest-written-last seal ordering.

### 4.5 Tests that must change

`py/test/test_replay_store.py` — **no change** (see §4.1).

`py/test/test_replay_manager.py` — roughly half. Deleted with their mechanisms:
`test_materialization_claims_use_bounded_deterministic_game_batches` (667),
`test_replay_shard_queue_rejects_cross_claim_duplicates_and_reordering` (682),
`test_late_arriving_older_inbox_file_is_requeued_behind_claimed_games` (709),
`test_transient_inline_failure_retries_same_durable_claim` (869),
`test_queue_gap_and_orphan_sealed_shard_are_rejected` (938),
`test_repeated_materialized_totals_do_not_reopen_or_hash_sealed_shards` (958),
`test_leftover_inbox_file_for_ingested_game_is_never_restaged` (1259),
`test_inbox_rescan_stats_only_newly_arrived_files` (1280),
`test_inbox_rescan_drops_removed_and_refreshes_requeued_files` (1312),
`test_appending_a_sealed_shard_does_not_reparse_its_manifest` (1333),
`test_sealed_shards_awaiting_append_do_not_consume_materialization_slots` (1359),
`test_staged_shard_backlog_is_bounded_when_the_appender_stalls` (1376).

**Inverted** — these currently assert the behaviour the owner is deliberately reversing, and each one is a decision
being made explicitly, so each should be rewritten rather than deleted:
`test_missing_durable_claim_source_is_fatal` (898) → a missing source is skipped;
`test_missing_earlier_sealed_sequence_blocks_later_shard_without_changing_totals` (978) → no head-of-line block;
`test_replay_manager_keeps_malformed_game_for_inspection` (1178) → the malformed game is quarantined in `rejected/`
and the run continues.

Rewritten around worker directories: the four restart tests (1057, 1079, 1110, 1139).
Survive with edits: `test_replay_manager_stages_appends_and_reopens_fifo` (634),
`test_synthetic_flood_appends_every_game_exactly_once_with_bounded_inbox` (1003),
`test_dispatcher_thread_stages_flood_without_inbox_growth` (1027),
`test_replay_ingestion_updates_central_resignation_state` (1220),
`test_append_with_nothing_staged_does_not_flush_the_store` (1390).

`py/test/test_parallel_materialization.py`: `test_source_claim_mismatch_deletes_nothing_and_seals_nothing` (249)
deleted; `test_malformed_later_game_deletes_no_source_and_seals_nothing` (277) **inverts** to "the malformed game is
rejected and the others still seal"; `test_sealed_shard_with_undeleted_inbox_is_recovered_without_rematerializing`
(328) survives and becomes the test for A1.

`py/test/test_replay_shard.py`: `test_shard_identity_is_stable_and_changes_with_persisted_source_inputs` (266) and
`test_typed_manifests_reject_bad_spans_order_and_duplicate_games` (408) need updating for the new identity and the
dropped order key. The rest is unaffected.

`py/test/test_training_coordinator_replay.py`:
`test_duplicate_sealed_callback_reconciles_absolute_credit_without_double_counting` (101) deleted with the callback.

### 4.6 Tests to add — the ones that would have caught this

1. **Dispatcher cost is independent of inbox depth.** Populate an inbox with 50,000 files, assert one dispatcher
   pass completes under a fixed wall-clock bound, and assert the bound does not change at 100,000. This is the exact
   property that `InboxScanner` violated.
2. **Dispatcher cost is independent of how many games were already ingested.** Ingest 10,000 games, then time a pass
   against a 100-file inbox. This is the `_modified_at_by_name` leak (§1.5), which no existing test can see.
3. **A game that raises during materialization is quarantined and its shard-mates still reach the store.**
4. **A worker killed mid-shard loses nothing on restart**, and one killed between seal and unlink duplicates
   nothing (A1).

---

## 5. Risks

### 5.1 What the current complexity was silently handling

- **Source integrity between claim and materialization** (§1.7). Precluded upstream by atomic publication. What is
  genuinely lost is detection of external mutation of an inbox file. **Low risk, accept.**
- **Deterministic reproducibility of ingestion.** Today, order is a pure function of `(mtime_ns, name)` and shard
  identities are content-addressed, so replaying an inbox snapshot reproduces byte-identical shards. In the new
  design, order depends on `scandir` order and worker scheduling. Nothing in the codebase depends on this — the
  sampler's RNG is seeded from `(sampler_seed, source_optimizer_step)` (`batch_loader.py:127`), independent of
  ingestion order — but a production failure can no longer be reproduced by replaying an inbox. **Real loss for
  debugging.** Partially mitigated: the counter prefix in the filename permanently records the ingest order.
- **Bounded staging growth** (§1.3). Replaced by a `len(os.listdir(staging))` guard. **Low risk if implemented;
  a real leak if forgotten.**
- **Loud failure on bad data** — see §5.3, the largest residual risk.

### 5.2 The wedge: what the new design eliminates by construction

The root cause was never conclusively identified. Here is what the code and the measurements do establish.

**Established from the frozen artefacts.** `shard-queue.json` at `{"next_sequence": 2738, "pending": []}` means
`self._queue.pending` was empty in memory too — `self._queue` is only ever mutated together with a
`_save_queue` (`manager.py:559-561`, `677-678`), so file and memory cannot diverge. An empty `pending` state is
exactly what `_finalize_committed_claims` leaves behind after a successful full append. So the wedge began
immediately after a normal append drained the queue, and from then on `_allocate_claims` built batches every pass
and **committed none of them** — the pre-`09986f20` bare `return` in the commit block returns before
`_save_queue`, leaving the file byte-identical forever. That is a precise match for the observed frozen file.

**Established by measurement** (local disk, Python 3.10, 100,000-file inbox, 200 KB games; the production node
differs but the scaling does not):

| Per-pass cost | Measured |
| --- | --- |
| `InboxScanner.scan`, degraded (i.e. always, after the first deletion) | 0.33 s |
| SHA-256 of 1,024 source files at `manager.py:511` | 14.8 s |
| `invalidate` per call at n = 16,000 | 5.68 ms (≈38 ms extrapolated to n = 108,000) |

One `_allocate_claims` pass therefore costs **~15 s** at a 108,000-file inbox, against a 1.0 s poll interval
(`DISPATCH_INTERVAL_SECONDS`, `manager.py:44`). The dispatcher thread is saturated by construction, and if the
commit block bails, all 15 s is discarded and repeated. The process is **self-amplifying**: while the pass runs,
self-play keeps publishing, `n` grows, and the next pass is slower. This is a sufficient explanation of "100% CPU in
pure Python for 45+ minutes, no generation, no error" without needing any further trigger.

**Not established:** which specific condition inside the commit block took the bare `return`. With `pending: []`,
`last_claimed_key` is `None` (`manager.py:475-478`), so the `os.utime` branch cannot fire, and the candidates reduce
to an `OSError` or a size/mtime mismatch at `manager.py:536-547` on a file that nothing should have been touching.
**What would resolve it:** the fetched `.codex-diagnostics/` archive for that run — specifically a listing of the
inbox with `stat` timestamps, and whether any non-canonical `.json` name was present. I did not have that archive.
I would not spend more time on this: the design below removes every candidate.

**Eliminated by construction:**

| Candidate cause | Why it cannot recur |
| --- | --- |
| `InboxScanner` timestamp cache and its unbounded `_modified_at_by_name` | the scanner is deleted; the dispatcher is stateless |
| O(n²) `invalidate` via the `os.utime` re-queue branch | mtime ordering is gone, so the branch is gone |
| Re-hashing the same ~1,024 files every pass | no hashing anywhere in the dispatcher |
| All-or-nothing commit discarding a whole pass | a rename either happens or does not, per file; a completed rename is never redone |
| The `if self._queue != queue_snapshot: return` livelock | there is no queue and no snapshot |
| `_contiguous_sealed_claims` head-of-line block on one missing sequence | there is no sequence |
| A fatal-error flag reachable from game data | game-data failures are per-game and non-fatal |
| Any per-pass cost that scales with inbox depth | `islice(scandir, cap)` makes it O(cap) (A3) |

**Carried forward, not eliminated:**

| Residual | Note |
| --- | --- |
| `readdir` on a directory with 10^5+ entries is still slow at the syscall level | A3 bounds the *reading*, but if the backlog grows the directory itself stays large and dentry lookups degrade. Bounded, not removed. |
| Whatever caused the backlog to start growing | If materialization throughput is genuinely below production rate, the new design does not fix it — it makes it *visible* (worker directories grow) instead of invisible. The owner's report that materialization was keeping up until the instant it wedged points at the dispatcher, which is consistent. |
| The appender holds `manager._lock` across a whole-mapping `msync` | `manager.py:375-404`; the comment at 380-381 records ~0.2 s/GiB. Unchanged by the rework. |
| Store header tearing on a crash mid-`append_columns` | Pre-existing and explicitly accepted. |
| Pool failure modes | Addressed by §3.8, but only if `ProcessPoolExecutor` is actually replaced. |

### 5.3 The largest residual risk: silent total loss

"Skip anything that cannot be materialized, never stall" has a failure mode the current design does not have. If a
code change, a layout mismatch or a game-schema drift makes **every** game unmaterializable, the new pipeline drops
every one of them, cheerfully, forever. The run produces zero samples, earns zero credits, and never trains — and it
will look like a self-play stall, not a materialization bug. That is a *worse* outcome than the wedge, because the
wedge at least left an obvious 100%-CPU thread and a frozen queue file.

**This must not be left to a log line.** Required:

- A rejected-game counter and a rejected-bytes counter per worker, surfaced through `ReplayIngestionTelemetry`
  (`coordinator.py:280-285`) alongside the existing `materialization_failures`.
- A hard failure when the rejection rate over a rolling window of, say, 512 games exceeds a configured threshold
  (5% is a reasonable starting point; the production rate today should be ~0%). This is a *structural* failure, so
  it belongs in the retained fatal path of §1.10, not the per-game path.
- Rejected files moved to `rejected/`, not unlinked, so a post-mortem has the evidence. The disk cost is bounded by
  the threshold above.

With that, both requirements hold simultaneously: one bad game never stalls the run, and systematic breakage is
loud within a few hundred games.

### 5.4 Immediate, independent of the rework

`09986f20` added two `invalidate` call sites in the claim commit block (`manager.py:538`, `546`), each reachable up
to 1,024 times per dispatch pass, each O(inbox depth) with a full tuple rebuild. At a 108,000-file inbox that is
≈39 s of pure Python per pass at a 1 s poll interval. The commit is a correct fix for the bug it names, but it
should not be run in production against a large inbox as it stands. Two cheap options if that branch is needed
before the rework lands: make `InboxScanner._ordered` a list and mark entries tombstoned instead of rebuilding, or
simply drop the whole scanner cache and `stat()` on demand — at these backlog sizes the cache is a net loss anyway
(§1.5).

---

## 6. Open questions

1. **The exact trigger of the commit-block bail.** Resolvable only from the run's `.codex-diagnostics/` archive
   (inbox listing with timestamps, and whether a non-canonical `.json` name was present). I recommend *not*
   pursuing it: §5.2 shows the design removes every candidate, and the cost of the investigation exceeds its value.
2. **`RENAME_CAP` and `STAGING_CAP` values.** Need one measurement each on the production node: sustained games/s
   from self-play, and shard seal rate at 16 workers. Both are cheap to obtain from an existing run's telemetry.
3. **Whether `inbox/`, `workers/` and `staging/` are guaranteed same-device on the Vast.ai node layout.** Should be
   asserted at startup rather than assumed; a cross-device rename would silently reintroduce the copy cost.
4. **The rejection-rate threshold.** 5% is a guess. The right value is "clearly above the observed steady-state
   rate", which nobody has measured because today any rejection is fatal.
