# vast-chess-8gpu-run1 — resolved diff against the r3 snapshot

Reference: the r3 `TrainingArgs` repr in the frozen TensorBoard export
(`documentation/images/four-day-analysis/training_args.txt`, `training_args::TrainingArgs/text_summary`, first
entry, run `vast-chess-8gpu-1d-r3`) plus, for the self-play/objective/evaluation sections the snapshot does not
contain, the generations-0–149 slice of `documentation/experiments/vast-chess-8-gpu-config.yaml`.

## Resolution status (2026-08-21, phase-a @ 7c146d79, locked venv on the Phase A test node)

All cross-stream keys resolve after the phase-a merge (`warmup_optimizer_steps`, the deepened
`chess_76_plane_direct_v2` head, staged `parallel_searches`, `virtual_loss_weight`, per-definition `nodes` on
`stockfish_fixed_nodes`). `load_experiment_configuration` fails on exactly the two deliberate topology
placeholders (`training.topology.self_play.device_ids`, `node_ids_to_pause_during_training`) — nothing else.

With the documented stand-in topology (3 processes/GPU on 8 GPUs, every third worker unpaused) the
configuration resolves fully; provisional `experiment_configuration_sha256`
`f52e87ae8ee34e22bf60a3453147d4dddb4f49c214ee8b1e84af83bc244563d3` (run2:
`6e6161cf011bf796f7f96ff93a1f6ba3ff770bd37a8036c9b81f393451e29f3b`). Resolved JSON archived under
`.codex-diagnostics/wp6-resolution-20260821/`. The canonical SHA is recomputed when the real node topology,
`offer_id` and `hourly_price` replace the placeholders.

## Identical to r3

Trunk CNN 12×112, global pooling every second block, value head 2 channels / FC 48; AdamW, batch 2048/256,
bfloat16, `compilation: disabled`, grad clip 0.5, LR 0.005 → 0.0035@100 → 0.002@300; replay capacity linear
300k → 2M over generations 0–100, maximum 2M, 60 policy entries; replay ratio 8, 500 steps/quantum, backpressure 5,
checkpoint interval 6, inference retention 22/20; trainer topology (DDP 0–7, 4 threads) and evaluation device
cycle 4–7; visit schedule 200/300@10/400@30/500@50/600@90/700@180/800@250/1000@550; fast searches
50/75/100/125/150; full-search probability linear 1.0 → 0.25 over 0–70; Dirichlet 0.25/0.3, exploration 1.5,
FPU reduced-parent 0.2, forced playouts 1.5, retained-root 0.6; restart-state start (min remaining plies 15);
greedy 60 → 80@110; max plies 150/160@50/180@80/200@110/250@140; force-fast after ply 200; temperatures 1.3/0.1;
resignation (calibrated, ceiling 0.03, from generation 70); objective weights 1.0/1.0, root-value blend
0 → 0.15 over 50–110, value discount 1.0; auxiliary targets `next_policy` 0.1 and `remaining_game_length` 0.1
(scale 400); evaluation cadence/timeouts/engine, v1 dataset `chess-stockfish-evaluation-v1.bin`, v1 openings, and
the old definitions (fixed-dataset, previous-20/40/60m, Stockfish levels 0–4, fixed-nodes-1000); random seed
20260811; open-file/RAM/disk/telemetry limits.

## Differences from r3

| field | r3 | run1 | reason |
|---|---|---|---|
| policy head | dense 1880-way (`num_policy_channels: 4`), init std 1.0 | new hidden-layer 76-plane head (`chess_76_plane_direct_v2`, WP1) | planned: the fixed head |
| LR warm-up | none | `warmup_optimizer_steps: 1000` | planned (WP1) |
| `parallel_searches` | 1 | staged: 1 below 600 full-search visits, 4 from generation 90 (WP3) | planned |
| `virtual_loss_weight` | key absent (behaviour = 1.0) | 1.0 explicit | WP3 key, value unchanged |
| ingestion | synchronous between-quanta ingestion | `materialization_processes: 8` (file-staged rework, WP2) | planned; code-side |
| evaluation rungs | fixed-nodes-1000 only | + `stockfish_fixed_nodes` 30/100/300 with per-definition `match_nodes` | planned (WP3); 1000-rung node count unchanged |
| evaluation search `sdpa_backend` | absent (automatic) | `memory_efficient` explicit | current-stack convention; same kernel class on 3090 |
| hardware | 8×3060, 64 CPUs, offer instance-47400225… | 8×3090 intended, `offer_id: unconfirmed` | placeholder until rented |
| self-play topology | 16 workers (2/GPU), 8 paused | **PLACEHOLDER** | node unknown |
| `hourly_price` | 0.4608888888888886 | 0.0 **PLACEHOLDER** | set with the offer; approval must match |
| `maximum_wall_time_seconds` | null | 14400 | 2–4 h test run cap |
| `manual_stop_file` | key absent in r3-era schema | `/workspace/run-control/stop/vast-chess-8gpu-run1.stop` | required by run_control.sh |
| `progressive_model_sizing` | block absent in r3-era schema | single model + inert promotion block | schema requirement; sizing off |
| `save_path` / run name | `…/production/vast-chess-8gpu-1d-r3` | `…/validation/vast-chess-8gpu-run1` | new run identity |
