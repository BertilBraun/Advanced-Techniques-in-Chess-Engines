# Progressive-sizing search throughput — attention rungs, 1× RTX 4070 SUPER, 2026-08-23

Why the four-day run starts on a ~1.5M-parameter rung instead of the 3M target model: the small rung
converts the same GPU into nearly twice the self-play searches per second, and the advantage grows in
the production multi-process topology.

## Setup

- Node: Vast.ai instance 48395477 (8× RTX 4070 SUPER, the four-day production node), GPU 0 only,
  no other GPU load during measurement.
- Source: `master` @ `352ea1af` (benchmark ran before the rung-1 resize commit `778abb6b`).
- Tool: `py/tools/benchmark_self_play_search.py`, generation-0 conditions (200 full-search visits,
  fast 50, 100 % full-search probability), 512 parallel games per process, `parallel_searches 2`,
  2 inference workers, inference batch cap 64, 2 outstanding batches per worker, 90 s per point after
  2 warm-up batches. Raw JSON: `.codex-diagnostics/bench-4day-sizing-20260823/`.
- Models (freshly initialised, TorchScript inference export):
  - "small" = attention 8×128, 4 heads, ff 256, plane head — 1.123M parameters,
    run config `vast-chess-4day-attention.yaml` (pre-resize),
    `experiment_configuration_sha256 b62d72894376bff44070257e873fa6cb5631102e955388cd406ffb03a65ccc97`.
  - "big" = attention 12×176, 11 heads, ff 352, plane head — 3.066M parameters,
    run config `vast-chess-8gpu-exp2-attention.yaml`,
    `experiment_configuration_sha256 1558723a7e005efd598212dc5b43c746318c3193e68a4592733acde0cf7bc0c2`.

## Results

| Topology | 8×128 (1.12M) searches/s | 12×176 (3.07M) searches/s | small / big |
|---|---:|---:|---:|
| 1 process | 33,094 | 24,330 | 1.36× |
| 4 processes, same GPU (production shape) | 101,411 | 54,565 | 1.86× |

Average inference batch was 63.8–63.9 (pinned at the 64 cap) in every cell, so the ratios reflect pure
per-batch model latency, not batching differences. Completed games in 90 s: 52 vs 30 (4-process cells).

## Readings

- Multi-process scaling is 3.06× (33.1k → 101.4k) for the small model but only 2.24× (24.3k → 54.6k)
  for the big one: four concurrent 12×176 processes saturate the GPU's compute, while 8×128 still has
  headroom that process-level fill exploits. `nvidia-smi` utilisation reads 99–100 % in both cases and
  is not a useful signal here.
- Per wall-clock hour of early training, the small first rung therefore yields ~1.86× the self-play
  games of starting directly on the 3M model — this is the entire rationale for the first
  progressive-sizing rung in `vast-chess-4day-attention.yaml`.
- Rung 1 was subsequently resized to 11×128 (8 heads, 1.521M parameters, commit `778abb6b`) for a clean
  ×2 parameter ladder (1.52M → 3.07M → 5.73M). The 11×128 point was not re-measured; interpolating the
  two measured points by depth (11/8 layers on the 8×128 latency share) predicts roughly 80–85k
  searches/s per GPU in the 4-process topology, still ≈1.5× the 12×176 rung.
- Numbers are specific to the RTX 4070 SUPER and the batch-64 inference cap; do not compare against
  other hardware without saying so.
