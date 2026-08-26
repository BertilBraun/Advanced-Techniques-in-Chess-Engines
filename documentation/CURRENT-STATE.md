# Current state

As of **2026-08-26**. Update at every phase acceptance and node change; if this date is more than two weeks old,
treat everything below as unverified.

## Programme

Chess recovery, [plan](plan/chess-recovery-plan-20260820.md) of 2026-08-20. Phase A is complete. Multi-day
production runs on the 8×RTX 4070 SUPER node are the current activity: v2 through v9, each a named
configuration under `py/configs/validation/`, judged on **wall-clock** against the four-day baseline. Go 7x7/9x9
screening is paused.

**Live run: `vast-chess-4day-production-v9`** — the strongest engine produced so far. At 4.7 h it stands at
ladder Elo 1502 and level-0 95/0/5, against v8's 1236 and 0.800 at the same point; at 15600 s it scored 97/1/2,
which is the four-day reference's twelve-hour level-0 result reached in 4.3 hours.

## Work-package status

| WP | Scope | Status |
|---|---|---|
| WP1 | Output heads and initialisation | landed (`3644d4cf`; shared plane policy head `9ab92a52`, gen-0 guard `b134af79`, castling-plane mirror fix `b2cb00ce`) |
| WP2 | Ingestion rework (file-staged materialization) | landed (`0c5c9e73` / `1e8e24f9`) |
| WP3 | Small fixes, staged early termination, fixed-node rungs | landed (`7a67cc82`; rungs + ladder fit `426b15cf`) |
| WP4 | Supervised testbed on the frozen replay | done (`7a7bc52c`) |
| WP5 | Throughput measurements | done (`be1f2e00`) |
| WP6 | Test-run configurations | resolved (`ec97fc54`; run1 staging `0d548d6f`) |
| WP7 | Pipeline smoke test | **in flight** (overnight smoke, gates Phase B) |
| WP8 | Run control and evidence preservation | landed (`3e38d9a1`) |

## The last trusted result

The four-day run (r3/r4, source `d39d5c85..d9888436`, tagged
[`four-day-baseline`](https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines/releases/tag/four-day-baseline)):
≈2,800 ladder Elo at 10k visits; generation 445 scored 66.0% vs Stockfish 13 at 6,500 nodes (CI 58.5–73.5%).
Frozen evidence: `.codex-diagnostics/chess-baseline-four-day-freeze-20260817` and
`documentation/evidence/chess-four-day-freeze-20260817/` and the flip harness (see WP status). The pre-rework era is tagged `pre-rework`.

## Pass/fail reference

The [per-generation yardstick](plan/chess-recovery-plan-20260820.md#1-per-generation-yardstick-from-the-four-day-run)
is the reference for any new run. Test-run rule: at generation 50, fixed top-1 ≥ 0.27 and SF level 0 ≥ 0.5; at
generation 20, prev-20m ≥ 0.65.

## What was broken, and where it stands

The four post-rework defects (WP1–WP3, WP6) are fixed and were validated by the Phase A smoke. Since then:

- **Replay materialization wedged four runs.** The claim-queue design did O(inbox) work under a lock shared with
  the coordinator; v6 died at generation 35 with a 203,845-file inbox. Replaced by per-worker dispatch — games are
  renamed straight out of the inbox into a directory each materialization worker owns, with no shared mutable
  state and no global ordering invariant. See
  [replay-materialization-rework.md](architecture/replay-materialization-rework.md). Stable across v7–v9.
- **v7 and v8 could not convert won games** — 13–22 evaluation games per 100 abandoned at the ply cap while ahead
  by a median of 20 pawns. Resolved in v9, which abandons zero. Five hypotheses were eliminated with evidence
  before the cause was found; the investigation, the eliminations and the method lessons are in
  [chess-conversion-investigation-20260826.md](analysis/chess-conversion-investigation-20260826.md).
- **Syzygy adjudication removed entirely** (`02293f38`). A ply-capped game now takes its value from a forced full
  search at the cut position, so no assumed terminal value enters training from any source.
- **A trainer CUDA-stream leak** (+50.2 MiB/generation) and a graph-mempool leak were fixed; trainer rank memory
  has been byte-identical across generations since.

## How to run something

`deployment/run_control.sh` (start / stop / status / preserve / fetch) is the only supported way —
see [run-control.md](operations/run-control.md). A run that has no fetched archive under `.codex-diagnostics/`
did not happen. Node facts live in [operations/current-node.md](operations/current-node.md).

## Architecture diagrams

[C++ overview](architecture/diagrams/cpp-overview.png) ·
[inference pipeline](architecture/diagrams/cpp-inference-pipeline.png) ·
[chess input representation](architecture/diagrams/chess-input-representation.png) ·
[network architecture](architecture/diagrams/neural-network-architecture.png)
(era: pre-rework — structure still broadly accurate, details superseded).

## Deliberately not happening

No Go screening. No checkpoint averaging or model gating. Syzygy tablebases are no longer provisioned or used.

## Known open items

- `test_self_play_worker.py`, `test_trainer_group.py`, `test_interactive_engine.py` and
  `test_game_contracts.py` skip without the native extension, so ~13 failures are visible only on a node. **Run
  the suite on a node before any launch that touches self-play** — two validators that would have rejected every
  ply-capped game were caught only that way.
- `documentation/operations/current-node.md` describes a superseded single-GPU node; `run_control.sh` reads it.
- A failed supervisor spawn runs preserve-on-exit once per retry, so a start that fails for lack of disk creates
  archives until it succeeds or the disk fills.
