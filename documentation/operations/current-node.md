# Current node

**No node is rented** (as of 2026-10-03). The last one, described below, ran the AdamW attention run and was
destroyed after it stopped at generation 950 on 2026-10-02; its evidence is under
`.codex-diagnostics/vast-chess-8gpu-final-attention-adamw-20261002/`
([results](../benchmarks/attention-adamw-final-run-rtx4080s-20261002/README.md)). Nothing on it survived.

Rented 2026-09-30 for the AdamW attention run
([configuration](../../py/configs/production/vast-chess-8gpu-final-attention-adamw.yaml)).

- Destination (gone): `root@45.77.214.165:24803`.
- Local key: `C:/Users/berti/.ssh/vast-ssh`; never copy it to the node or repository.
- Vast instance: `53481269`. `/workspace` is container storage, not a volume: nothing survives a destroy.
- GPUs: 8 x NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB each, **power limit 200 W**, mixed board vendors (VBIOS
  95.03.44.00.F3 and .D4), so TensorRT warns about engines used across "different models of devices".
- Driver: `580.95.05`.
- CPU: AMD EPYC 7V12, cgroup v2 quota **122.9 CPUs**.
- RAM: **241 GiB** cgroup limit; the host reports 251 GiB.
- Disk: 200 GiB overlay.
- Downloads: 9-11 MB/s per stream from PyPI, pypi.nvidia.com, download.pytorch.org and the NVIDIA apt mirror,
  4 MB/s from GitHub. `setup_remote.sh` provisioned it in 1,222 s.
- The Vast image sets `UV_NO_CACHE=1`; set `UV_CACHE_DIR` and unset it when a reusable uv cache is wanted.

A fallback 8x RTX 3090 node (instance `53479423`, `root@175.155.64.175:19898`, 483 GiB RAM, 107.5 CPUs, 1.7 MB/s
per stream) was provisioned in parallel. The previous 8x RTX 4070 SUPER node (instance `53401154`, 120 GiB RAM)
ran the SGD attention run to generation 116; its evidence is under
`.codex-diagnostics/vast-chess-8gpu-final-attention-20260930/`.

Use `deployment/remote_command.sh` for every remote command and `deployment/run_control.sh`
for run start/stop/status/preserve/fetch. `run_control.sh` runs on the node; only `fetch`
runs from the workstation.
