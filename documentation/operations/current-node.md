# Current node

Provisioned 2026-09-29 for the attention run
([plan](../plan/chess-attention-final-run-plan-20260929.md)).

- Destination: `root@137.175.22.196:17886`.
- Local key: `C:/Users/berti/.ssh/vast-ssh`; never copy it to the node or repository.
- Vast instance: `53401154`. `/workspace` is container storage, not a volume: nothing survives a destroy.
- GPUs: 8 x NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB each, devices 0-7, **power limit 160 W** (card maximum
  220 W; not adjustable from the unprivileged container).
- Driver: `580.159.03`; driver maximum CUDA: `13.0`.
- Locked runtime: PyTorch `2.12.1+cu126`, CUDA `12.6`, cuDNN `91002` (9.10.2); Python TensorRT `10.14.1.48.post1`;
  native `libnvinfer10 10.14.1.48-1+cuda12.9` (pinned in `setup_remote.sh`; apt offers 11.3 for CUDA 13.4 here).
- CPU: Intel Xeon E5-2673 v4, 80 logical CPUs visible; cgroup v2 quota **76.8 CPUs**.
- RAM: **120 GiB** cgroup limit; the host reports 125 GiB.
- Disk: 200 GiB overlay, 177 GiB free after provisioning.
- Downloads measured at provisioning: 13-16 MB/s per stream from PyPI, pypi.nvidia.com, download.pytorch.org and
  the NVIDIA apt mirror, 38 MB/s over four parallel PyPI streams, 6 MB/s from GitHub. Provisioning took 934 s.
- Control checkout: `/workspace/alphazero-engine` (shallow clone of `master`).
- Locked environment: `/workspace/alphazero-engine-venv`, shared by every checkout.
- Evaluation engines: `/workspace/alphazero-engine/engines`: Stockfish 18 and 13, KataGo 1.17.1
  `cuda12.8-cudnn9.8.0`; the smoke checks passed.
- The Vast image sets `UV_NO_CACHE=1`; set `UV_CACHE_DIR` and unset it when a reusable uv cache is wanted.

Use `deployment/remote_command.sh` for every remote command and `deployment/run_control.sh`
for run start/stop/status/preserve/fetch. `run_control.sh` runs on the node; only `fetch`
runs from the workstation. `/etc/vast-agents-guide.md` was read for this provisioning.
