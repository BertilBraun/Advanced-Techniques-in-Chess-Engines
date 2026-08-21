# vast-chess-8gpu-run2 — resolved diff against the r3 snapshot

Run 2 extends `vast-chess-8gpu-run1.yaml`; everything in
[`vast-chess-8gpu-run1.r3-diff.md`](vast-chess-8gpu-run1.r3-diff.md) applies unchanged unless listed here,
including the resolution status, the pending WP1/WP3 keys and the topology/hourly-price placeholders.

## Resolution status (2026-08-21, branch wp8-run-control)

Same failure set as Run 1 plus exactly one more pending key:

| key | error today | owner |
|---|---|---|
| `chess.self_play.early_termination` (staged `maximum_game_plies` + `censor_remaining_game_length_target`) | extra input | WP3 |

With that key and the Run-1 set reverted to today's schema, the merged configuration resolves cleanly through the
pydantic models (verified). Final resolution and diff regeneration after the `phase-a` merge.

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
