# Vast node comparison — 2×RTX 3060 / RTX 4070 Super / RTX 4070 / RTX 3090 — 2026-08-21

Four rented Vast.ai instances benchmarked with `deployment/benchmark_node.sh` (untracked at the time; copied to each
node) at source revision `994a8cfa` (master; `dirty=true` in the archives only because of the copied script).
Attention/inference configuration SHA `8e5d551edb1c2d3a111d6032e964ebc9eaf8b75e743021e43173da5acc3c2eec`
(`vast-chess-8gpu-optimal.yaml`); CNN reference `vast-chess-8gpu-cnn-reference.yaml`. Torch 2.12.1+cu126, cuDNN 91002
on all nodes. Prices are whole-instance `dph_total` from `vastai show instances`. Raw archives:
`.codex-diagnostics/node-comparison-20260821/<label>/`. Nodes are already released; nothing remains on them.

**Cross-hardware comparison by design** — hosts differ in CPU, quota, PCIe and RAM; that is what was measured.

## Nodes

| node | GPUs | CPU (host) | effective CPUs (cgroup) | RAM (limit) | PCIe effective (pinned H2D) | price/h |
|---|---|---|---:|---|---|---:|
| rtx3060x2 | 2× RTX 3060 12 GiB | EPYC 7K62, nproc 96 | 18.43 | 125.6 GiB (53.9) | ~6.3 GiB/s (≈gen3 x8) | $0.1483 |
| rtx4070s | 1× RTX 4070 Super 12 GiB | EPYC 7K62, nproc 96 | 15.36 | 220 GiB (86.2) | ~12.4 GiB/s (≈gen4 x8) | $0.1428 |
| rtx4070 | 1× RTX 4070 12 GiB | Ryzen 7 7700X, nproc 16 | 15.36 | 30.5 GiB (29.3) | ~24.9 GiB/s (gen4 x16) | $0.1350 |
| rtx3090 | 1× RTX 3090 24 GiB | i9-9900K, nproc 16 | 7.68 | 62.7 GiB (42.5) | ~5.9 GiB/s (gen3 x8) | $0.1689 |

No thermal or power-brake throttling observed on any node during the runs (post-run `nvidia-smi -q -d PERFORMANCE`);
all GPUs idle at preflight. "gen1" PCIe readings in `hardware.txt` are idle-state (P8) samples; the pinned-bandwidth
probes above are the effective link measurements.

## Self-play search throughput (best grid point; 512 games/process, 60 s, generation 0)

| node | model | best grid point | searches/s (node) | per GPU | per $ (M searches/$·h) |
|---|---|---|---:|---:|---:|
| rtx4070s | att | ppg3 ps1 | 110,868 | 110,868 | 2,795.0 |
| rtx4070 | att | ppg3 ps4 | 104,645 | 104,645 | 2,790.5 |
| rtx3090 | att | ppg3 ps4 | 91,223 | 91,223 | 1,944.4 |
| rtx3060x2 | att | ppg3 ps1 | 81,938 | 40,969 | 1,989.1 |
| rtx4070s | cnn | ppg3 ps4 | 67,940 | 67,940 | 1,712.8 |
| rtx4070 | cnn | ppg2 ps4 | 56,043 | 56,043 | 1,494.5 |
| rtx3090 | cnn | ppg3 ps4 | 54,421 | 54,421 | 1,159.9 |
| rtx3060x2 | cnn | ppg2 ps4 | 48,942 | 24,471 | 1,188.1 |

Full 8-point grids are in each archive's `SUMMARY.md`. Grid sensitivity was small everywhere (≤5% between best and
worst point except rtx4070s att, where ppg3 beat ppg2 by ~18%).

## Inference, batch 64, scripted (positions/s; latency = 64/throughput)

| node (per GPU) | att-1m | att-2m | att-4m5 | att-2m latency |
|---|---:|---:|---:|---:|
| rtx4070 | 82,538 | 50,780 | 29,433 | 1.26 ms |
| rtx3090 | 53,004 | 44,871 | 29,195 | 1.43 ms |
| rtx4070s | 41,186 | 34,682 | 25,703 | 1.85 ms |
| rtx3060 (each of 2) | ~36,000 | ~20,300 | ~13,460 | 3.15 ms |

The rtx4070s small-model inference numbers are launch-bound artifacts of the low-clock shared EPYC host (scripted
single-stream benchmark), not GPU limits — its self-play throughput, which batches across workers, was the best
measured.

## Trainer throughput

**Not measured on any node.** `benchmark_node.sh` step 5 is defective: `benchmark_training_overfit.py` requires an
existing run directory and a pre-populated replay store (`overfit-<m>/replay.bin`, ≥512 rows) that nothing on a
fresh node can produce. Failure logs are in every archive (`trainer-*.log`). Fix the script before its next use.
Second script defect (worked around identically on all four nodes): step 4 copies `self-play-summary.json` but runs
write `summary.json`; per-grid summaries were remapped from run manifests and `SUMMARY.md` regenerated with the
script's own step-6 code — measurements untouched.

## Rankings

- **Self-play per dollar (binding constraint):** rtx4070s ≈ rtx4070 (2,795 vs 2,790 M/$·h att; 1,713 vs 1,494 cnn —
  the 4070S clearly ahead on CNN) ≫ rtx3060x2 (1,989 / 1,188) ≈ rtx3090 (1,944 / 1,160). Ada mid-range is ~40–45%
  more cost-efficient than Ampere for this workload.
- **Self-play per wall hour:** rtx4070s (110.9k att / 67.9k cnn) > rtx4070 (104.6k / 56.0k) > rtx3090 (91.2k /
  54.4k) > rtx3060x2 (81.9k / 48.9k).
- **Notable:** the RTX 3090's 24 GiB VRAM and bandwidth bought nothing for self-play at batch ~64; it was also the
  most expensive and most CPU-quota'd (7.68 cores) node. The 2×3060 node was not CPU-starved (18.4 effective cores;
  ppg2→ppg3 flat ⇒ GPU-bound) — its per-GPU numbers are simply 3060-class.

## Recommendation (decision was the user's; nodes already released)

- **Phase A (needs DDP ⇒ ≥2 GPUs):** rtx3060x2 was the only multi-GPU node and was not CPU-starved; at $0.1483/h it
  was the correct DDP development choice despite mediocre per-GPU throughput.
- **Phase B 8-GPU node GPU class:** Ada mid/high consumer class (4070 Super / 4070 Ti Super / 4090), not Ampere
  3090-class. Per-dollar self-play favours Ada by ~40%+; check PCIe ≥ gen4 x8 per GPU and a CPU quota of ≥2 real
  cores per worker process. Caveat: trainer throughput was never measured, so if the trainer becomes the bottleneck
  on an 8-GPU node the 24 GiB VRAM class could still matter — measure after the script defect is fixed.
