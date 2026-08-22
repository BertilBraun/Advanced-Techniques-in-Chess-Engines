# Phase A kickoff — hardware choice (2026-08-21)

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

