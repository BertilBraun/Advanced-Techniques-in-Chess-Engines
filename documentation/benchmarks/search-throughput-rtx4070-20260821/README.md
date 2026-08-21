# Self-play search throughput — 1× RTX 4070, 2026-08-21 (WP5, trimmed)

Node: Vast.ai `cae651504bf0` (Phase A test node; see
`documentation/operations/node-provisioning-phase-a-test-20260821.md`). Source `phase-a` @ `0fe515c0`;
run configuration: `vast-chess-8gpu-run1.yaml` with the stand-in 3-per-GPU topology,
`experiment_configuration_sha256 f52e87ae8ee34e22bf60a3453147d4dddb4f49c214ee8b1e84af83bc244563d3`.
Tool: `py/tools/benchmark_self_play_search.py`, single process, 512 parallel games, 2 inference workers,
inference batch cap 64, 2 outstanding batches/worker, freshly initialised run1 network (CNN 12×112 GP,
new plane head), 90 s per point after 2 warm-up batches. Raw JSON in
`.codex-diagnostics/wp5-throughput-20260821/`.

Search mix comes from the run1 schedules at the chosen generation — both points use 25 % full searches
(the representative production mix): generation 70 → 500 full-search visits, fast 125; generation 250 →
800 full-search visits, fast 150.

| generation (visits) | parallel_searches | searches/s | avg inference batch | games finished |
|---|---:|---:|---:|---:|
| 70 (500) | 1 | 69,309 | 19.8 | 1,778 |
| 70 (500) | 2 | 82,661 | 28.8 | 2,208 |
| 70 (500) | 4 | 93,471 | 44.3 | 2,533 |
| 250 (800) | 1 | 84,323 | 17.6 | 1,261 |
| 250 (800) | 2 | 99,951 | 25.6 | 1,575 |
| 250 (800) | 4 | 116,411 | 41.3 | 1,898 |

Readings:

- parallel_searches 1→4 buys +35 % searches/s at 500 visits and +38 % at 800 visits in a single process;
  the mechanism is batch fill (avg batch 18–20 → 41–44 against the cap of 64).
- Single-process numbers here (93–116 k/s at ps 4) are consistent with the node-comparison best of 104.6 k/s
  on a 4070 with the multi-process ppg{2,3} grid — multi-process fill still adds on top of ps 1–2, less on ps 4.
- Deviation from the plan's matrix: points at 400/800 visits with ps ∈ {1,2,4} were specified; the closest
  generations with the representative 25 % full-search mix are 70 (500 visits) and 250 (800 visits). The
  `virtual_loss_weight` ∈ {0.5, 1.0} KL check was skipped (plan marks it low priority).
- Hardware note: Phase B is 8× this GPU class, so these numbers scale directly (×8 upper bound, minus
  coordinator/ingestion overhead).
