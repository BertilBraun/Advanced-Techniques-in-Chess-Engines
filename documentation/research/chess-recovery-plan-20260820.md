# Chess recovery plan: from the post-rework stall back to a trustworthy training loop

Date: 2026-08-20. Companion to
[`chess-post-four-day-regression-analysis-20260820.md`](chess-post-four-day-regression-analysis-20260820.md), which
holds the evidence. This document holds the decisions and the work. It is meant to be referenced phase by phase;
each work package has a scope, an owner slot, and an acceptance criterion that can be checked without judgement calls.

## 0. Decisions taken on 2026-08-20

- The four-day run (r3/r4, ≈2,800 ladder Elo at 10k visits) was a clean, from-scratch result. The stall of the
  post-rework runs is explained by (1) the bare 1×1 `chess_76_plane_direct_v2` head plus ReLU-gain Kaiming init on every
  linear layer and no final normalisation on the attention trunk, and (2) the bounded-ingestion coordinator starving
  self-play. Everything else that changed (auxiliary targets, adaptive budget, Syzygy, progressive sizing, attention)
  is kept and re-validated, not discarded.
- The policy-plane representation, including the black-side rank flip, is verified end to end by a compiled harness
  (2,834 positions, 83,651 moves, zero discrepancies; `cpp/test/flip-harness/`).
- File-mirror augmentation stays, but castling-right planes must be swapped under mirroring (pre-existing oversight).
- Replay ratio 10 is acceptable. `parallel_searches` stays as a throughput lever but is staged by visit count.
- The in-run yardstick moves to Stockfish 13 fixed-node rungs 30 / 100 / 300 / 1,000 (≈1,100 / 1,200 / 1,400 / 1,700
  on the Melonimarco SSDF-scale curve), 64 visits, 100 games each. Levels 0 and 4 are kept only during the test runs for
  comparability with the four-day yardstick.
- Early game termination in the first generations is preferred over forced fast searches; cut games are labelled by
  the existing material heuristic / Syzygy, never by the root search value.
- Development and all small-scale tests run on a 1–2 GPU node. Pipeline tests run on an 8-GPU node in 2–4 h runs.
  The long run happens only after Run 1 and Run 2 (section 3) have passed.

## 1. Per-generation yardstick (from the four-day run)

Hardware-independent targets, because generation = 500 optimizer steps at batch 2048 and replay ratio 8. Values are
20-minute evaluation bins nearest to the generation; 100 games, 64 visits. A new run must track these *per generation*
on any hardware; wall time is a separate throughput question.

| gen | fixed top-1 | SF level 0 | SF level 1 | SF level 4 | SF 1,000 nodes | prev-20m | note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.07 | 0.07 | 0.02 | 0.01 | 0.02 | – | random network |
| 10 | 0.17 | 0.09 | 0.03 | 0.01 | 0.00 | 0.89 | visits 200→300 |
| 19 | 0.22 | 0.16 | 0.06 | 0.01 | 0.01 | 0.69 | ≈1 h on 8×3060 |
| 31 | 0.24 | 0.35 | 0.18 | 0.04 | 0.01 | 0.84 | visits 400 |
| 41 | 0.27 | 0.54 | 0.27 | 0.10 | 0.02 | 0.61 | |
| 50 | 0.30 | 0.63 | 0.47 | 0.15 | 0.02 | 0.61 | visits 500; ≈3 h |
| 61 | 0.32 | 0.84 | 0.66 | 0.25 | 0.02 | 0.45 | ≈4 h |
| 79 | 0.35 | 0.93 | 0.80 | 0.43 | 0.05 | 0.48 | |
| 100 | 0.37 | 0.94 | 0.90 | 0.62 | 0.09 | 0.51 | LR 0.005→0.0035 |
| 124 | 0.37 | 0.99 | 0.95 | 0.67 | 0.10 | 0.54 | |

Pass/fail rule for a test run: at generation 50, fixed top-1 ≥ 0.27 and SF level 0 ≥ 0.5; at generation 20, prev-20m
≥ 0.65. The fixed-dataset number is only comparable if the v1 dataset and the dense-softmax metric are used; with the v2
dataset use the match scores only. The fixed-node rungs 30/100/300 have no four-day reference values; they will be
established by Run 1 and become the reference from then on.

## 2. Phase A — development and small-scale tests (1–2 GPU node)

Order matters only where stated. WP1–WP3 and WP8 are code; WP4–WP6 are measurements that need the code; WP7 is
the integration smoke test that gates Phase B.

### WP1 — Output heads and initialisation

Scope: `py/src/training/network.py`, `py/src/training/checkpoint/persistence.py` (export), tests.

1. New shared `PolicyPlaneHead(in_channels, hidden=64, planes=76)`: 1×1 conv `in→64` + BN + ReLU → 3×3 conv
   `64→64` + BN + ReLU → 1×1 conv `64→76` (bias). ≈51k parameters at any trunk width ≥ 64. No pooled-bias branch.
   Final conv weight init N(0, 0.01), bias 0.
2. Auxiliary 4,864-way heads (`next_policy`, `legal_moves`) use the same module with `hidden=32` (≈16k parameters
   each: 1×1 `in→32`, 3×3 `32→32`, 1×1 `32→76`). Scalar auxiliary heads and the value head: keep their structure, add
   the same small-init rule on the last layer.
3. Attention trunk: add a final `LayerNorm(embedding_size)` on the tokens before `AttentionOutput`. Replace the
   blanket Kaiming-ReLU init: `nn.Linear` in attention blocks → N(0, 0.02); attention output projection and
   feed-forward output projection additionally scaled by 1/√(2·num_layers); embeddings unchanged. Convolutional trunk
   keeps Kaiming for convs followed by ReLU.
4. Trainer: optional linear LR warm-up (`trainer.warmup_optimizer_steps`, default 0 so old configs resolve
   unchanged; production configs set 1,000–2,000).
5. Startup guard: after model creation at generation 0, run one real replay batch (or the Stockfish fixed dataset) and
   assert legal-masked policy-logit std ≤ 1.5 and masked entropy ≥ 0.6 × ln(legal count); log both to TensorBoard as
   `init/policy_logit_std`, `init/policy_entropy_ratio`.
6. Keep the old dense head kind selectable (`policy_head: {kind: dense}`) for the replica run.

Acceptance: `py/tools/init_probe.py` reports logit std ≤ 1.0 for all production
models; JIT export matches training forward to 1e-5 for CNN and attention (existing export tests extended to the new
head); all `py/test` pass; the C++ inference smoke (`TestMovePolicyProcessing`) passes against an exported new-head
model.

### WP2 — Ingestion rework

Scope: `py/src/training/coordinator.py`, `py/src/replay/{manager,parallel_materialization,materialization,store}.py`,
`credit_ledger.py`, tests.

Today (for reference): between quanta, the main process globs and lexicographically sorts the inbox, cuts it into
batches of 128 paths, calls `executor.map` synchronously on an 8-process pool, receives per-sample `ReplaySample`
dataclasses back through pickling, writes them row by row into the mmap in a Python loop, then unlinks the files. Only
one game is ever assigned to one worker (the pool does that) and only the main process writes the store, so it is
*correct*; it is slow because of the object pickling and the per-row Python writes, and it only runs in the window
between quanta with a 10-second cap, so freshness is not guaranteed.

Design (agreed), file-staged variant — chosen over an in-RAM staging buffer because it has no IPC payload and is
crash-durable:

- Three roles, three directories under the run's `completed-games/`: `inbox/` (self-play workers write game JSON,
  atomically via rename — unchanged), `staging/` (materialiser workers write finished row blocks), and the live
  replay store (main process only).
- **Materialiser pool** (the existing 8 processes, same initializer) runs continuously, including during training
  quanta. A dispatcher thread in the coordinator process lists `inbox/` once per second, sorts by mtime, and submits
  each path exactly once to the pool (an in-memory `claimed` set prevents re-submission). The GIL is not a concern
  here: the trainer ranks are separate spawned processes (`trainer/group.py`), so the coordinator process itself is
  mostly blocked in pipe reads, sleeps and `nvidia-smi`-style telemetry, all of which release the GIL, and the
  dispatcher's own work is a directory listing and a few `submit` calls per second. The heavy work is in the pool
  processes. If profiling ever shows the coordinator saturating its GIL, the dispatcher moves to its own small
  process with a queue to the coordinator; the design does not change.
- A worker parses the game, replays it, computes targets, and writes **one file per game**,
  `staging/<game-id>.rows.npy` (an `np.ndarray` of `layout.row_dtype`, ≈200 rows ≈ 250 kB) plus a small
  `<game-id>.meta.json` (plies, termination reason, resignation observations, policy-truncation statistics), both via
  `src/util/atomic_file.py` (temp name + `os.replace`). **The worker then unlinks the inbox original itself**, so a
  game is never present in both `inbox/` and `staging/` at rest. It returns only the game id and row count to the
  dispatcher, which adds the credits to the ledger. No sample data crosses process boundaries.
- **Nothing is appended to the live store during a quantum** — the trainer samples the mmap ring in place. At the
  quantum boundary the coordinator lists `staging/` (mtime order), concatenates the blocks, does one
  `store.extend_rows(block)` (≈128k rows ≈ 150 MB per generation, sub-second), flushes, consumes the meta files for
  telemetry and resignation calibration, then unlinks the staged files. Freshness guarantee: every game materialised
  before the boundary is in the next quantum's data.
- Crash safety: a game exists in exactly one of three states — `inbox/` (not yet materialised), `staging/`
  (materialised, not yet appended), appended (no file). A worker crash between writing the staging files and unlinking
  the inbox original leaves both; on restart (and on every listing) a game id already present in `staging/` is skipped
  and its inbox file removed. An orphaned temp file is deleted. A game can never be appended twice because append and
  unlink are one boundary step that completes before the next listing; a crash between them is caught on restart by
  the store row count versus the ledger (the existing manifest/ledger check) and the staged files are re-checked
  against the store's last appended game ids.
- Worker pausing has exactly two rules: (a) during a training quantum, pause `node_ids_to_pause_during_training`;
  (b) when available credits exceed `self_play_backpressure_quanta` quanta, pause the same set until credits fall below
  it (checked by the main loop once per second; this is a ledger read, no IPC). There is no inbox-count rule and no
  pausing while waiting for credits. `INGESTION_PAUSE_BACKLOG_GAMES` and the 10-second slice are removed.
- The generation loop regains its reference shape: wait-for-credits → boundary append → train quantum → publish.

Acceptance: ingestion wall time per generation ≤ 3 % of the quantum on the 1–2 GPU node (measure
`replay/ingest_seconds`); `credit/wait_seconds` is dominated by self-play, not by materialisation; inbox depth stays
bounded under a synthetic flood (test with pre-generated game files); the existing replay-manager tests pass with the
new API; a kill -9 during a quantum followed by restart loses no game and duplicates none (test).

### WP3 — Small fixes

- Mirror augmentation: swap planes 12↔13 and 14↔15 (own/opponent king-side ↔ queen-side castling rights) in
  `transform_encoded_state` for the file-mirror transform, and add a test that the mirrored tensor equals the C++
  encoding of the mirrored position for positions *with* castling rights (the harness in `flip-harness/` already
  covers the no-castling case).
- `parallel_searches` becomes a schedule (`staged`), with production values 1 (or 2) below 600 full-search visits and
  4 from 600 on. Add a `virtual_loss_weight` parameter (default 1.0 = current behaviour) in `SearchTree.hpp` so the
  in-flight penalty can be a fraction of a full loss.
- Early-game policy: config option `early_termination` with a generation-staged `maximum_game_plies` that cuts games
  short in the first generations and adjudicates with the existing material heuristic / Syzygy, and a censoring flag
  for the remaining-game-length target on cut games. The forced-fast-search schedule remains available.
- Evaluator: `stockfish_fixed_nodes` definitions for 30 / 100 / 300 / 1,000 nodes, plus a single logistic-fit Elo
  across the rungs (`evaluation/ladder_elo`) using the agreed anchors; levels 0 and 4 kept in the test-run configs
  only.
- `reports/comparison.md` intervals: the rung-level bootstrap from the analysis is adopted for future ladders.

### WP4 — Supervised testbed on the frozen replay

Data: the four-day freeze replay store (1.5 M rows) as training set; the three-day full-state store and the Stockfish
fixed dataset as held-out. Tool: `py/tools/benchmark_training_overfit.py` generalised to a full-store loader (uniform
sampling, mirror augmentation on), eight epochs (= the online presentation count), evaluation every 1,000 steps.

Matrix (each ≤ 1 h on one 3090 for the ≤ 1.1 M models):

| model | head | init/norm | LR |
|---|---|---|---|
| CNN 12×112 | old dense | old | 5e-3 (reference) |
| CNN 12×112 | new plane head | new | 5e-3 |
| CNN 12×144 | new plane head | new | 5e-3 |
| attention 8×128 | new plane head | new (final LN, 0.02 init) | 1e-3, 2e-3, 5e-3, each with 1k warm-up |
| attention 6×96 | new plane head | new | best LR from above |

Acceptance: the new-head CNN matches the old-head CNN within 2 % on held-out policy CE at equal steps; the attention
model has a stable gradient norm (< 2 after the first 200 steps) at the chosen LR and reaches held-out CE within 5 % of
the CNN of equal parameter count by the end of epoch 8. Anything that fails here does not go online.

### WP5 — Throughput measurements (one GPU, fixed reference model)

- Batch fill versus `parallel_searches` ∈ {1, 2, 4} at 400 and 800 full-search visits, 512 games per process, 2
  workers × 2 outstanding batches: record achieved batch size, searches/s, positions/s.
- `virtual_loss_weight` ∈ {0.5, 1.0} at `parallel_searches` 4 (low priority; λ < 1 only reduces the in-flight bias, it is not expected to matter much): positions/s and the KL between the resulting root
  visit distributions and `parallel_searches: 1` on a fixed set of 500 positions (target-distortion measurement).
- Trainer: samples/s for CNN 12×144 and attention 8×128 / 15×192 with the new heads, compiled and eager.

Acceptance: numbers recorded under `documentation/benchmarks/` with the config SHA; no threshold, these feed the
production config.

### WP6 — Test-run configurations

- `vast-chess-8gpu-run1.yaml` ("old recipe, new head"): r3's values for network (CNN 12×112, global pooling),
  optimizer, replay (capacity schedule, ratio 8 for comparability with the yardstick — ratio 10 is a Run-2 change),
  visit/fast/ply/greedy schedules, the two old auxiliary targets at the old weights, `parallel_searches` per WP3
  schedule, `compilation: disabled`, no Syzygy, no adaptive budget; plus the new plane head, warm-up 1,000 steps,
  fixed ingestion, mirror fix, v1 evaluation dataset and the old evaluation definitions *plus* the new fixed-node
  rungs. Topology for the node at hand (3 processes per GPU with 2 paused during training is fine).
- `vast-chess-8gpu-run2.yaml` ("everything on"): Run 1 plus six auxiliary targets at the `optimal` weights, adaptive
  search budget, Syzygy adjudication, early termination, replay ratio 10, `compilation: default`, v2 evaluation
  dataset (keep v1 definitions too for one more run). Progressive sizing off (inert in 4 h anyway). Attention trunk
  only if WP4 passed for it; otherwise CNN 12×144.
- Both resolve through the existing approval/validation flow; the resolved JSON of each is diffed against the
  resolved r3 configuration from the freeze and the diff is committed next to the config.

### WP8 — Run-control interface and evidence preservation

Scope: `deployment/`, `py/run_approved_experiment.py`, documentation in `documentation/operations/`.

One script, `deployment/run_control.sh`, is the only way humans and agents start, stop, inspect and archive a
production or test run:

- `run_control.sh start <config.yaml>` — validates approval and clean checkout as today, installs the supervisor
  program, starts it, prints run name, TensorBoard directory and state directory.
- `run_control.sh stop <run-name>` — touches the run's `manual_stop_file` (the coordinator already finishes the current
  quantum and writes a checkpoint on it), waits for the process to exit, then runs `preserve`.
- `run_control.sh status <run-name>` — supervisor state, last generation, last evaluation line, inbox/staging depth,
  GPU utilisation.
- `run_control.sh preserve <run-name>` — copies the TensorBoard directory, `run-state/` manifests, resolved
  configuration, credit ledger, run log and the evaluation results into
  `.codex-diagnostics/<run-name>-<UTC timestamp>/` with a SHA256SUMS file; idempotent, safe to re-run.

Preservation must not depend on the run finishing: the runner process installs a `trap` on `EXIT`, `SIGTERM` and
`SIGINT` that calls `preserve`, the supervisor definition gets `stopwaitsecs` large enough for the checkpoint plus the
copy (≥ 300 s), and `stop` calls `preserve` explicitly as well, so a cancelled run is archived by at least one path.
`run_control.sh fetch <run-name> <local-dir>` (run from the workstation) rsyncs the archive from the compute node
into the local `.codex-diagnostics/`, verifying SHA256SUMS; archives live on the node until fetched.
`documentation/operations/run-control.md` documents the five commands, the expected outputs, and the rule that a test
run without a fetched `.codex-diagnostics` archive did not happen.

### WP7 — Pipeline smoke test (1–2 GPU node, gates Phase B)

Run `run1.yaml` with the node's topology for 50 generations. Compare per generation against section 1. Pass: the
generation-20 and generation-50 rules hold; ingestion ≤ 3 % of quantum; no worker stalls; `init/*` guards pass;
evaluator produces the four fixed-node rungs and `ladder_elo` every cadence.

## 3. Phase B — 8-GPU test ladder (2–4 h each)

Run 1 — old recipe with the new head. Pass: tracks section 1 per generation (and, on 3090-class hardware, should reach
generation 50 in well under 3 h). If it fails: the defect is in something WP1–WP3 touched or in the native build on
that node; bisect with the `d9888436` build on the same node before anything else. Do not proceed to Run 2.

Run 2 — everything on. Pass: matches or beats Run 1 per generation at generation 50, and fixed-node `ladder_elo` is
not lower at equal generation. If it fails: Run 2a with the data-side changes only (adaptive budget, early
termination, Syzygy, ratio 10) and Run 2b with the model-side changes only (aux targets, attention, compilation); at
most two more runs.

Every run is started and stopped through `run_control.sh` (WP8), so its TensorBoard directory, run state and
resolved config land in `.codex-diagnostics/` whether it finishes or is cancelled (the failed runs of 18–19 August
were lost; that must not happen again).

## 4. Phase C — the long run

Only after Run 2 (or 2a/2b) has passed. Baseline configuration is the passing Run-2 variant with: a 15×192-class
trunk (CNN or attention per WP4), visits reaching 1,000 by about generation 250 and 1,600 by generation 500, LR decay
pulled forward relative to r4 (the 0.004→0.003 drop produced the last clear gain at ≈50 h), replay ratio 10,
progressive sizing on if the small-node test showed promotion working, fixed-node rungs extended with 3,000 and 10,000
nodes once the 1,000-node rung passes 0.5. Final evaluation: the full fixed-node ladder at s1000 / s10000 / t1s / t5s
with 100–200 games on the rungs around the estimated strength, so the headline number is bracketed rather than
extrapolated.

## 5. Open items to decide later

- Whether to keep the `search_correction` and `irreversible_progress` targets after Run 2 (no literature support; cost
  is small; decide on the Run-2 ablation if one is needed).
- The promotion criterion for progressive sizing (currently total loss including auxiliary terms on the small model's
  data distribution); consider primary-loss only and a minimum generation count on the candidate.
- Whether mirror augmentation should stay at p = 0.5 once the castling planes are fixed, or be reduced for positions
  with castling rights.
