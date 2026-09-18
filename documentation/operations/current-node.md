# Current node

Verified 2026-09-18 for the V79 FP16 control run authorized by the user.

- Destination: `root@38.49.42.120:53893`.
- Local key: `C:/Users/berti/.ssh/vast-ssh`; never copy it to the node or repository.
- Vast instance: `48571853`.
- GPUs: 8 x NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB each, devices 0-7.
- Driver: `595.71.05`; driver maximum CUDA: `13.2`.
- Locked runtime: PyTorch `2.12.1+cu126`, CUDA `12.6`, cuDNN `91002` (9.10.2).
- CPU: 80 logical CPUs visible; cgroup quota 76.8 CPU equivalents.
- RAM: 251 GiB total, 190 GiB available at preflight.
- Disk: 150 GiB overlay, 47 GiB available; ephemeral container storage.
- Control checkout: `/workspace/alphazero-engine`.
- Locked environment: `/workspace/alphazero-engine-venv`, shared by every checkout.
- Evaluation engines: `/workspace/alphazero-engine/engines`; KataGo installation records
  `cuda12.8-cudnn9.8.0`, version 1.17.1.

The V79 run executes from `/workspace/alphazero-engine-calibration-diagnostic` at revision
`8cb256182a3330a33583db174cdbe9474131d7ee`, whose `py/AlphaZeroCpp.so` is a symlink into
`/workspace/alphazero-engine-v75-v35-small-prefold-int8`; V79 changes configuration only, so that
Release build still applies.

Disk is the binding constraint on this node. One progressive run's `replay.bin` reaches about 25 GiB
at the 6M-sample capacity stage, and the capacity schedule continues to 9M at generation 400. Before
starting a run, free space and preserve first: `run_control.sh preserve` archives the resolved
configuration, logs, TensorBoard and run state, but not model weights, so deleting a run directory
discards its checkpoints permanently. Space for V79 was reclaimed by deleting the V77 `replay.bin`,
`completed-games` and `resignation` after preserving that run, which the user authorized.

Use `deployment/remote_command.sh` for every remote command and `deployment/run_control.sh`
for run start/stop/status/preserve/fetch. Note that `run_control.sh` runs on the node; only `fetch`
runs from the workstation. `/etc/vast-agents-guide.md` was read for the earlier provisioning.

Outstanding: no archive of the V77/V78 or V79 evidence has been copied off the node yet, and a
`test_trainer_group.py` pytest process (pid 1328777) has been hung for 20 days without holding a GPU.
