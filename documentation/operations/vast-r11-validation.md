# R11 Vast integrated validation

This runbook records the rented node, reproducible bootstrap, scaled validation configurations, approvals, and
evidence for the R11 integrated-validation pass. The node is ephemeral and is not a production benchmark baseline.

## Connection

From the Windows development host:

```powershell
ssh -i C:\Users\berti\.ssh\codex_vast_ed25519 -p 56488 root@171.101.230.38 -L 8080:localhost:8080
```

The private key remains at `C:\Users\berti\.ssh\codex_vast_ed25519`; never copy or commit it. Read
`/etc/vast-agents-guide.md` before acting on the instance. `/workspace` is not a persistent Vast volume on this
offer, so copy evidence off-node before recycle or destruction.

## Rented hardware and runtime

Inventory was recorded on 2026-08-09 with `vastai show instances --raw`, `vast-capabilities metrics,packages`,
`nvidia-smi`, `lscpu`, `free -h`, `df -hT`, `nvcc --version`, and the installed cuDNN headers.

| Item | Observed value |
| --- | --- |
| Vast instance / host | `47261298` / `95290` |
| Price | `$0.303/hour` total (`$0.261333/hour` GPUs + `$0.041667/hour` disk) |
| Image | `vastai/pytorch:cuda-13.0.3-auto` |
| GPU | 4× NVIDIA GeForce RTX 3060, 12 GiB each, compute capability 8.6 |
| Driver / host toolkit | NVIDIA 580.65.06 / CUDA 13.0.88 |
| Host cuDNN | 9.14.0 |
| Locked training runtime | Python 3.12.3, PyTorch 2.12.1+cu126, wheel CUDA 12.6 / cuDNN 9.10.2 |
| CPU | 2× AMD EPYC 7B12, 256 logical CPUs; Vast reports 85.33 effective CPUs |
| RAM | 251 GiB total |
| Disk | 150 GiB ephemeral NVMe-backed container storage |

The listing was expected to be an RTX 4070 offer, but the live allocation is four RTX 3060s. Configuration and
evidence use the observed devices, not the advertised assumption.

## Bootstrap

KataGo 1.17.1 has no exact CUDA 13.0/cuDNN 9.14 release asset. The selected official CUDA 12.8/cuDNN 9.8.0
Linux asset is the nearest lower compatible build. The CUDA 13.0 driver runs CUDA 12.x applications; the bootstrap
exposes the pinned PyTorch wheel's CUDA 12 libraries to KataGo. The official archive SHA-256 is
`458d226c2c8533600251bba3b2ee612d3aee0c796f592a2b53839a6a05b0826e`.

On a fresh instance, fetch `deployment/setup_remote.sh` from the exact approved source revision, then run:

```bash
export ENGINE_REPOSITORY_REF=master
export ENGINE_KATAGO_BACKEND=cuda12.8-cudnn9.8.0
export ENGINE_KATAGO_ARCHIVE_URL=https://github.com/lightvector/KataGo/releases/download/v1.17.1/katago-v1.17.1-cuda12.8-cudnn9.8.0-linux-x64.zip
export ENGINE_KATAGO_ARCHIVE_SHA256=458d226c2c8533600251bba3b2ee612d3aee0c796f592a2b53839a6a05b0826e
bash /tmp/setup_remote.sh COMMAND ARGUMENTS
```

The bootstrap clones to `/workspace/alphazero-engine`, creates `/workspace/alphazero-engine-venv`, installs the
locked cu126 runtime, builds the Release native extension, extracts the KataGo AppImage for FUSE-free execution,
installs and smokes both engines, exports their paths and `ENGINE_SOURCE_REVISION`, and finally executes the runner.
For test-suite validation only, install the test runner into the generated environment:

```bash
/workspace/alphazero-engine-venv/bin/uv pip install \
  --python /workspace/alphazero-engine-venv/bin/python \
  pytest==8.4.2
```

Pytest is not added to the locked production training dependencies.

## Scaled validation topology

The three successful smokes are intended to run concurrently with one visible physical GPU each:

| Physical GPU | Configuration | Purpose |
| --- | --- | --- |
| 0 | `py/configs/validation/vast-r11-chess.yaml` | five-generation chess smoke |
| 1 | `py/configs/validation/vast-r11-go-7x7.yaml` | five-generation Go 7x7 smoke |
| 2 | `py/configs/validation/vast-r11-go-9x9.yaml` | five-generation Go 9x9 smoke |
| 3 | `py/configs/validation/vast-r11-timeout-go-7x7.yaml` | separate 30-second wall-time stop |

Each process receives one GPU through `CUDA_VISIBLE_DEVICES`, so its approved experiment correctly records one
visible RTX 3060 and uses logical device zero for trainer, one self-play worker with four parallel games, and the
evaluation device cycle. Successful runs use one optimizer step per generation and stop after generation five.
Evaluation uses one paired opening (two games), two searches/engine visits per move, and the fixed dataset. The
30-second configuration requests 500 generations so the wall-time limit, rather than the optimizer-step limit,
must stop it.

The fixed datasets still retain the architecture's required 480–520 positions. Preparation uses two-node/two-visit
engine labels and writes immutable ignored artifacts under `py/reference/`; it is intentionally much larger than
the two-game match smoke. Prepare or run the successful 7x7 configuration before the timeout configuration so the
latter reuses the same immutable 7x7 dataset and opening manifests.

## R10 infrastructure gate

This Vast image is an unprivileged Docker container with read-only cgroup v1 mounts. R10 requires a delegated,
writable cgroup v2 memory scope for each slot so the RAM limit covers the complete experiment process tree. The
queue's scheduler/configuration tests and expected preflight failure can be run here, but the real queue launcher
must not be weakened or bypass its cgroup validation. A future R10 integration node must expose delegated cgroup v2
(or be a suitable VM). The three R11 smokes therefore use explicit direct processes with disjoint
`CUDA_VISIBLE_DEVICES` assignments.

## Approvals and launch

Approval files stay outside Git. For every configuration, load the resolved experiment, compute
`src.experiment.run.experiment_sha256`, and write an `ApprovalRecord` containing the exact source revision, run
name, provider/instance identity, `$0.303` hourly price, configured cost cap, and configured wall-time minutes.
Set the runtime image expected by run preparation and launch from the repository root:

```bash
export TRAINING_RUNTIME_IMAGE=vastai/pytorch:cuda-13.0.3-auto
CUDA_VISIBLE_DEVICES=PHYSICAL_GPU \
  /workspace/alphazero-engine-venv/bin/python py/train.py \
  --run-config CONFIGURATION \
  --expected-source-revision FULL_SOURCE_REVISION \
  --approval-file APPROVAL_JSON
```

Run logs, run directories, generated evaluation inputs, evaluation result JSON, TensorBoard event files, GPU memory
samples, and the exact commands are copied into the dated R11 evidence directory after validation.

## Results

Pending execution on the rented node.
