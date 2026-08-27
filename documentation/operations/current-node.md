# Current node

Exactly one node is described here. When it is released, replace the body with
`No node rented as of <date>` and move the facts to a dated provisioning note.

## Facts (attention-viability experiment node, confirmed 2026-08-27)

| | |
| --- | --- |
| Provider | Vast.ai |
| SSH | `root@50.120.65.61:41841` (key is local-only; never in the repository or on the node) |
| GPU | 1× RTX 3060 12 GiB, driver 595.84, CUDA 13.2 reported by the driver |
| CPU / RAM / disk | 56 effective CPUs · 62 GiB · 150 GiB overlay, 13 GiB free on 2026-08-27 |
| `/workspace` volume-backed | **no** — nothing survives recycle/destroy; fetch evidence before release |
| Environment | Python 3.12.3 venv at `/workspace/alphazero-engine-venv`, torch 2.12.1+cu126 |
| Engines | not required by this work; the distillation path needs neither Stockfish nor KataGo |
| Checkouts | `/workspace/alphazero-engine` (control, `master`), `/workspace/attention-viability` (branch worktree) |

## Purpose and limits

Supervised distillation experiments on the `attention-viability` branch only. Not a production-run node
and not a clean benchmark node — see the contention note below.

**Throughput measured here does not transfer.** This repository has already measured that the
attention-versus-convolution throughput verdict inverts between an RTX 3060 and the RTX 4070 SUPER
production hardware (`documentation/benchmarks/chess-attention-sdpa-backends-*`,
`chess-architecture-contended-rtx3060-*`). Never compare a number from this node against a 4070 SUPER
number.

**The node is shared.** On 2026-08-27 the GPU was idle at 14:02 UTC, and a `measure_policy_target_fidelity`
job from the separate `search-evaluations` work started on it at about 14:33 UTC, holding 290 MiB and
driving the GPU to 58% utilisation on its own. Confirm with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` before trusting any timing, and
never `pkill -f` a pattern that also matches your own polling shell.

## The production node is off limits

`38.49.42.120:53893` runs the live four-day run `vast-chess-4day-production-v9`. Do not connect to it
from this branch's work.

## Before you connect

Connect with `deployment/remote_command.sh 50.120.65.61:41841 <command …>` (it owns the key and the
connection options), not a hand-written `ssh` line. A long job must be detached with
`setsid … < /dev/null > log 2>&1 &`, because the SSH channel stays open until every descriptor is
redirected.
