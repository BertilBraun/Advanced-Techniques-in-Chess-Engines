# Current node

Verified 2026-09-06 for the integrated v31 chess run authorized by the user.

- Destination: `root@38.49.42.120:53893`.
- Local key: `C:/Users/berti/.ssh/vast-ssh`; never copy it to the node or repository.
- Vast instance: `48571853`.
- GPUs: 8 x NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB each, devices 0-7.
- Driver: `595.71.05`; driver maximum CUDA: `13.2`.
- Locked runtime: PyTorch `2.12.1+cu126`, CUDA `12.6`, cuDNN `91002` (9.10.2).
- CPU: 80 logical CPUs visible; cgroup quota 76.8 CPU equivalents.
- RAM: 251 GiB total, 238 GiB available at preflight.
- Disk: 150 GiB overlay, 35 GiB available before cleanup; ephemeral container storage.
- Control checkout: `/workspace/alphazero-engine`.
- Locked environment: `/workspace/alphazero-engine-venv`.
- Evaluation engines: `/workspace/alphazero-engine/engines`; KataGo installation records
  `cuda12.8-cudnn9.8.0`, version 1.17.1.

At preflight all production supervisor jobs were stopped/exited. A separate pytest process
finished during inspection; no GPU workloads remained. The v31 run has not yet launched.

Use `deployment/remote_command.sh` for every remote command and `deployment/run_control.sh`
for run start/stop/status/preserve/fetch. `/etc/vast-agents-guide.md` was read completely.
The user authorized implementation, node builds and validation, disk cleanup, and production
launch after validation, with no scheduled wall-time limit and manual stopping.
