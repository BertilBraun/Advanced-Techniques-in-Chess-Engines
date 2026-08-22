# Current state

As of **2026-08-22**. Update at every phase acceptance and node change; if this date is more than two weeks old,
treat everything below as unverified.

## Programme

Chess recovery, [plan](research/chess-recovery-plan-20260820.md) of 2026-08-20. Phase A (development and
small-scale tests on a 1–2 GPU node) is nearly complete; the WP7 overnight pipeline smoke — the gate for
Phase B — is running on the 1×4070 test node using the run2 recipe
(`py/configs/validation/vast-chess-1x4070-overnight.yaml`, attention 8x128). Go 7x7/9x9 screening is paused.

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

The [per-generation yardstick](research/chess-recovery-plan-20260820.md#1-per-generation-yardstick-from-the-four-day-run)
is the reference for any new run. Test-run rule: at generation 50, fixed top-1 ≥ 0.27 and SF level 0 ≥ 0.5; at
generation 20, prev-20m ≥ 0.65.

## What was broken, and where it stands

All four defects behind the post-rework stall have fixes landed and are being validated by the WP7 smoke:
bare 1×1 policy head + Kaiming-ReLU init (WP1), bounded ingestion starving self-play (WP2),
unstaged `parallel_searches` at low visits (WP3/WP6), and the castling-plane mirror oversight (WP1).

## How to run something

`deployment/run_control.sh` (start / stop / status / preserve / fetch) is the only supported way —
see [run-control.md](operations/run-control.md). A run that has no fetched archive under `.codex-diagnostics/`
did not happen. Node facts live in [operations/current-node.md](operations/current-node.md).

## Deliberately not happening

No multi-day run until Run 1 and Run 2 of Phase B pass. No Go screening. No checkpoint averaging or model gating.
A repository-wide cleanup (code + documentation restructuring) is in progress on the `cleanup` branch.
