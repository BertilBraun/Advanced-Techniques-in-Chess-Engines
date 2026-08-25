# Current node

Exactly one node is described here. When it is released, replace the body with
`No node rented as of <date>` and move the facts to a dated provisioning note.

## Facts (rented 2026-08-21)

| | |
| --- | --- |
| Provider / container | Vast.ai `cae651504bf0` |
| SSH | `root@80.59.54.98:10574` (key is local-only; never in the repository or on the node) |
| GPU | 1× RTX 4070 12 GiB, driver 595.71.05, compute capability 8.9 |
| CPU / RAM / disk | Ryzen 7 7700X, 16 effective CPUs · 30 GiB · 150 GiB overlay |
| `/workspace` volume-backed | **no** — nothing survives recycle/destroy; fetch evidence before release |
| Environment | Python 3.12 venv at `/workspace/alphazero-engine-venv`, torch 2.12.1+cu126, cuDNN 91002 |
| Engines | Stockfish UCI + KataGo CUDA (smoked at setup) |

## Purpose and limits

Phase A validation and the WP7 overnight smoke only. Not a benchmark or production-run node.

## Before you connect

Connect with `deployment/remote_command.sh <HOST[:PORT]> <command …>` (it owns the key and the
connection options; the destination is the SSH row above), not a hand-written `ssh` line.
Read `/etc/vast-agents-guide.md` on the node before changing it. Runs go through
`deployment/run_control.sh` only. Provisioning history and deviations:
[node-phase-a-20260821.md](../evidence/node-phase-a-20260821.md).
