# Current node

Exactly one node is described here. When it is released, replace the body with
`No node rented as of <date>` and move the facts to a dated provisioning note.

## Facts (rented 2026-08-27)

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

## Purpose and limits

Dedicated single-GPU convolutional self-play inference throughput experiments. It is not a production-run node.

## Before you connect

Connect with `deployment/remote_command.sh <HOST[:PORT]> <command …>` (it owns the key and the
connection options; the destination is the SSH row above), not a hand-written `ssh` line.
Read `/etc/vast-agents-guide.md` on the node before changing it. Runs go through
`deployment/run_control.sh` only. Provisioning history and deviations will be recorded in the dated
inference-throughput benchmark note.
