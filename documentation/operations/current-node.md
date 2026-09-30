# Current node

Rented 2026-09-30 for the AdamW attention run
([configuration](../../py/configs/production/vast-chess-8gpu-final-attention-adamw.yaml)).

- Destination: `root@175.155.64.175:19898`.
- Local key: `C:/Users/berti/.ssh/vast-ssh`; never copy it to the node or repository.
- Vast instance: `53479423`. `/workspace` is container storage, not a volume: nothing survives a destroy.
- GPUs: 8 x NVIDIA GeForce RTX 3090, 24,576 MiB each, 350 W, one VBIOS (94.02.59.00.05) across all cards.
- Driver: `570.144`; driver maximum CUDA: `12.8`.
- CPU: Intel Xeon Gold 6330, 112 logical CPUs visible; cgroup v2 quota **107.5 CPUs**.
- RAM: **483 GiB** cgroup limit; the host reports 503 GiB.
- Disk: 200 GiB overlay.
- Downloads measured before provisioning: about 1.7 MB/s per stream from PyPI, pypi.nvidia.com, the NVIDIA apt
  mirror and GitHub, 4.5 MB/s from download.pytorch.org; the apt and uv caches were filled in parallel first.
- The Vast image sets `UV_NO_CACHE=1`; set `UV_CACHE_DIR` and unset it when a reusable uv cache is wanted.

The previous node (instance `53401154`, 8x RTX 4070 SUPER, 120 GiB RAM) ran the SGD attention run to generation
116; its evidence is under `.codex-diagnostics/vast-chess-8gpu-final-attention-20260930/`.

Use `deployment/remote_command.sh` for every remote command and `deployment/run_control.sh`
for run start/stop/status/preserve/fetch. `run_control.sh` runs on the node; only `fetch`
runs from the workstation.
