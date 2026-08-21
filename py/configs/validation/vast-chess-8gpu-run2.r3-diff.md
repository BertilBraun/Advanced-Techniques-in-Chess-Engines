# vast-chess-8gpu-run2 — resolved diff against the r3 snapshot

Run 2 extends `vast-chess-8gpu-run1.yaml`; everything in
[`vast-chess-8gpu-run1.r3-diff.md`](vast-chess-8gpu-run1.r3-diff.md) applies unchanged unless listed here,
including the resolution status, the pending WP1/WP3 keys and the topology/hourly-price placeholders.

## Resolution status (2026-08-21, phase-a @ 7c146d79, locked venv on the Phase A test node)

All cross-stream keys resolve after the phase-a merge, including `chess.self_play.early_termination`.
`load_experiment_configuration` fails only on the two topology placeholders inherited from Run 1. With the
stand-in topology documented in the Run-1 note the configuration resolves fully; provisional
`experiment_configuration_sha256` `6e6161cf011bf796f7f96ff93a1f6ba3ff770bd37a8036c9b81f393451e29f3b`.
Resolved JSON archived under `.codex-diagnostics/wp6-resolution-20260821/`. Canonical SHA follows the real
node topology.

## Additional differences from r3 (Run 2 deltas over Run 1)

| field | r3 | run2 | reason |
|---|---|---|---|
| trunk | CNN 12×112 | CNN 12×144 (`chess-cnn-12x144`) | planned; attention 8×128 only if WP4 passes — open decision, CNN now |
| `compilation` | disabled | default | planned |
| `replay_ratio` | 8 | 10 | planned (presentation credits stay integral: 1,024,000 / 10 = 102,400) |
| full-search budget | fixed schedule | adaptive: minimum 200, maximum = Run 1's schedule, learned gate from generation 50 | planned; secondary parameters carried from the last committed adaptive configuration (61d31c45 / e48b0376^) — open decision |
| Syzygy adjudication | none | `maximum_ply_syzygy_paths: [/workspace/syzygy/wdl345]` | planned |
| early termination | none | staged 80@0 / 120@20 / 150@40, censored remaining-length target (WP3 key) | planned; ply values are a proposal, not in the plan |
| auxiliary targets | `next_policy` 0.1, `remaining_game_length` 0.1 (scale 400) | six targets at the `optimal` weights: next_policy 0.05, remaining_game_length 0.025 (scale 200), future_search_value 0.025, irreversible_progress 0.0125, legal_moves 0.0125, search_correction 0.05 | planned; `search_correction` is also required by the adaptive learned gate |
| evaluation dataset / openings | v1 | v2 (`…direct-policy-evaluation-v2.bin`, v2 openings); all v1-era match definitions and the new rungs kept | planned; v2 fixed-dataset metrics are not numerically comparable with v1 — use match scores across runs |
| run identity | — | `vast-chess-8gpu-run2`, `…/validation/vast-chess-8gpu-run2`, own stop file | new run identity |
