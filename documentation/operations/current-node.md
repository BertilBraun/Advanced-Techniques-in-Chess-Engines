# Current node

Two nodes are described here while both are rented. When one is released, delete its section and move the
facts to a dated provisioning note.

## Facts (RTX 4070 SUPER, rented 2026-08-27)

| | |
| --- | --- |
| Provider / container | Vast.ai container `48910449` |
| SSH | `root@154.64.230.50:50623` (key is local-only; never in the repository or on the node) |
| Port forward | provider port 8080 → node `localhost:8080` |
| GPU | 1× RTX 4070 SUPER 12 GiB, driver 580.159.03, compute capability 8.9 |
| CPU / RAM / disk | 80 visible CPUs (2× Xeon E5-2673 v4), 9.6 effective CPU quota · 188 GiB · 32 GiB overlay |
| `/workspace` volume-backed | **no** — nothing survives recycle/destroy; fetch evidence before release |
| Environment | PyTorch 2.12.1+cu126 · CUDA runtime 12.6 · cuDNN 9.10.2 · system toolkit 12.8.93 |
| Engines | Stockfish 18 and 13; KataGo 1.17.1 `cuda12.8-cudnn9.8.0`; 7×7 and 9×9 CUDA smokes passed |

### Purpose and limits

Dedicated single-GPU convolutional self-play inference throughput experiments. It is not a production-run
node. This is the production card, so **throughput measured here does transfer** — unlike the RTX 3060
below.

### Sharing it with a second work stream

`/workspace/alphazero-engine` is the inference-throughput checkout. The attention work used its own
directories beside it: `/workspace/av-throughput` (source and its own native build), `/workspace/av-cells`
(checkpoints), `/workspace/av-results`. Two traps cost time there and will cost it again:

1. The clone is **shallow (depth 9)**, so a `git bundle` against any older base does not apply. Ship the
   tree with `git archive` and record the source revision in a file beside it.
2. An `AlphaZeroCpp.so` built from another branch will not necessarily match your Python: the
   `InferenceConfiguration` binding differed between branches. Build the extension from your own `cpp`
   tree, and note that the post-build step runs `ruff` on the generated stub, so `ruff.toml` has to be
   present at the tree root or the build fails on `N999`.

## Facts (RTX 3060, attention-viability experiments, rented 2026-08-21)

| | |
| --- | --- |
| Provider | Vast.ai |
| SSH | `root@50.120.65.61:41841` (key is local-only; never in the repository or on the node) |
| GPU | 1× RTX 3060 12 GiB, driver 595.84, CUDA 13.2 reported by the driver |
| CPU / RAM / disk | 56 effective CPUs · 62 GiB · 150 GiB overlay, 13 GiB free on 2026-08-27 |
| `/workspace` volume-backed | **no** — nothing survives recycle/destroy; fetch evidence before release |
| Environment | Python 3.12.3 venv at `/workspace/alphazero-engine-venv`, torch 2.12.1+cu126 |
| Engines | not required by that work; the distillation path needs neither Stockfish nor KataGo |
| Checkouts | `/workspace/alphazero-engine` (control, `master`), `/workspace/attention-viability`, `/workspace/av-tools` |

### Purpose and limits

Supervised distillation experiments only. Not a production-run node and not a clean benchmark node.

**Throughput measured here does not transfer.** This repository has measured that the
attention-versus-convolution throughput verdict inverts between an RTX 3060 and the RTX 4070 SUPER
production hardware (`documentation/benchmarks/chess-attention-sdpa-backends-*`,
`chess-architecture-contended-rtx3060-*`). Never compare a number from this node against a 4070 SUPER
number.

**It is shared.** On 2026-08-27 the GPU was idle at 14:02 UTC and a `measure_policy_target_fidelity` job
from the separate `search-evaluations` work started on it at about 14:33 UTC, holding 290 MiB and driving
the GPU to 58% utilisation on its own. Confirm with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` before trusting any timing, and
never `pkill -f` a pattern that also matches your own polling shell.

## The production node is off limits

`38.49.42.120:53893` runs the live four-day production run. Do not connect to it from experiment work.

## Before you connect

Connect with `deployment/remote_command.sh <HOST[:PORT]> <command …>` (it owns the key and the
connection options; the destination is the SSH row above), not a hand-written `ssh` line.
Read `/etc/vast-agents-guide.md` on the node before changing it. Runs go through
`deployment/run_control.sh` only. A long job must be detached with
`setsid … < /dev/null > log 2>&1 &`, because the SSH channel stays open until every descriptor is
redirected. Provisioning history and deviations are recorded in the dated benchmark notes.
