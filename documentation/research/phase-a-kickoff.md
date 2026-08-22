# Phase A kickoff — hardware choice and session prompt

## Hardware: what to rent and why

Cost per unit of self-play throughput is what matters, because the loop is self-play-bound after the first hours
and the trainer is a minority of the GPU time. From the project's own measurements: the 8×3060 node cost
$0.46/h ($0.058 per GPU-hour) and delivered ≈20k network evaluations/s per GPU for the 3.1 M CNN; the 8×3090 node
cost $1.12/h ($0.14 per GPU-hour) and delivered roughly 3× the evaluations/s per GPU on small networks. Per dollar
the two are close, with the 3090 slightly ahead; per wall hour the 3090 is far ahead, needs fewer processes for the
same output, and has 24 GiB, which the larger trunks and the supervised testbed want. A 4090 is typically another
1.6–2× over a 3090 for small-batch inference at about 2× the price, so again roughly break-even per dollar and
better per hour, but 4090 hosts often ship with fewer CPU cores per GPU, which this workload cannot afford. Avoid
offers that are GPU-rich and CPU-poor: the C++ self-play needs about 8 effective cores per GPU (2–3 processes ×
512 games), and materialisation needs another 8 cores on the node.

Development node (Phase A, WP4–WP7): prefer **2× RTX 3090** if available at ≤ $0.45/h, otherwise **1× RTX 3090**.
Two GPUs are worth it because WP7 must exercise the DDP trainer group, the per-GPU self-play topology and the
training-time worker pause, which a single GPU cannot. Two 3060s are the fallback (cheapest, but 12 GiB limits the
supervised sweep batch sizes and the 15×192 trunk).

Test and long-run node (Phase B/C): **8× RTX 3090**, ≥ 64 effective cores, ≥ 200 GiB RAM, ≥ 150 GB disk, ≥ 95 %
reliability, ≥ 500 Mbps down. 8× 4090 only if the price per GPU-hour is under about 2× the 3090 price and the host
has ≥ 96 effective cores.

Minimum requirements for any offer: `verified`, `reliability > 0.95`, `cpu_cores_effective ≥ 8 × num_gpus + 8`,
`cpu_ram ≥ 24 GiB × num_gpus`, `disk_space ≥ 100`, `inet_down ≥ 200`, `pcie_bw ≥ 10` (batch-64 inference is
PCIe-sensitive), driver supporting CUDA 12.6 user-space (`cuda_vers ≥ 12.6`), image `vastai/pytorch:cuda-13.0.3-auto`.

Queries (on-demand; add `-i` to see interruptible prices, which are often half but can be pre-empted mid-run —
acceptable for WP4/WP5 sweeps, not for WP7 or Phase B):

```bash
# development node
vastai search offers 'num_gpus=2 gpu_name in [RTX_3090] reliability>0.95 cpu_cores_effective>=24 cpu_ram>=64 disk_space>=100 inet_down>=200 pcie_bw>=10 cuda_vers>=12.6' -o 'dph_total' --limit 20
vastai search offers 'num_gpus=1 gpu_name in [RTX_3090,RTX_4090] reliability>0.95 cpu_cores_effective>=16 cpu_ram>=48 disk_space>=100 inet_down>=200 pcie_bw>=10 cuda_vers>=12.6' -o 'dph_total' --limit 20

# 8-GPU node
vastai search offers 'num_gpus=8 gpu_name in [RTX_3090,RTX_4090] reliability>0.95 cpu_cores_effective>=64 cpu_ram>=200 disk_space>=150 inet_down>=500 pcie_bw>=10 cuda_vers>=12.6' -o 'dph_total' --limit 20

# sanity view of value per dollar across GPU types
vastai search offers 'num_gpus=8 reliability>0.95 cpu_cores_effective>=64 cpu_ram>=200 disk_space>=150' -o 'dlperf_per_dphtotal-' --limit 20
```

Check `cpu_cores_effective` rather than `cpu_cores` (the cgroup quota is what you get), and look at
`direct_port_count` if you want the TensorBoard forward without an SSH tunnel.

## Prompt for the new session

Paste as the first message:

> Read `CLAUDE.md`, then `documentation/research/chess-recovery-plan-20260820.md` (the plan) and, for background,
> `documentation/research/chess-post-four-day-regression-analysis-20260820.md`. We are starting Phase A of the plan.
>
> Act as orchestrator. Create one task per work package and run WP1 (output heads and initialisation), WP2
> (file-staged ingestion rework) and WP8 (run-control interface) as three parallel sub-agent work streams on
> branches `wp1-heads`, `wp2-ingestion`, `wp8-run-control`. Fold the WP3 fixes into the stream that owns the touched
> files (mirror castling planes → WP1 stream; `parallel_searches` schedule and `virtual_loss_weight` → a small
> native change in its own branch `wp3-search`; early termination and the fixed-node evaluator → WP2 stream or a
> fourth stream, your call). Each stream works to the acceptance criterion in the plan, with tests, ruff and a
> compile check, and reports back with the handoff format from `CLAUDE.md`. You review each stream's diff before
> merging into an integration branch `phase-a`; nothing goes to `master` until I accept.
>
> Development node: `<1×/2× RTX 3090, instance id, ssh command>`; image `vastai/pytorch:cuda-13.0.3-auto`;
> provision with `deployment/setup_remote.sh` on branch `phase-a` once WP1/WP2/WP8 have merged. Use it for WP4
> (supervised testbed on the frozen replay — the four-day freeze and three-day stores are at
> `.codex-diagnostics/chess-baseline-four-day-freeze-20260817/`, upload the replay and the Stockfish evaluation
> dataset), WP5 (throughput) and WP7 (50-generation smoke test). Ask before starting anything that runs longer
> than 30 minutes on the GPU, and before every node-level action that costs money.
>
> Report in the handoff format after each stream lands and after each measurement, with the numbers against the
> plan's acceptance criteria.

Fill in the node line. If you want the streams to start before the node exists, leave the node line out; WP1, WP2,
WP8 and the WP3 items need no GPU.
