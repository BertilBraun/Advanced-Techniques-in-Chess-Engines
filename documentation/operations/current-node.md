# Current node

Exactly one node is described here. When it is released, replace the body with
`No node rented as of <date>` and move the facts to a dated provisioning note.

No node rented as of 2026-08-28.

The two single-GPU nodes used on 2026-08-27 were released that day:

- `154.64.230.50:50623`, 1× RTX 4070 SUPER — convolutional inference throughput, and the
  attention-viability throughput and native-build work.
- `50.120.65.61:41841`, 1× RTX 3060 — the attention-viability supervised distillation cells.

Their facts and the traps they cost are recorded in
[chess-attention-viability-rtx3060-20260827](../benchmarks/chess-attention-viability-rtx3060-20260827/README.md)
and [cnn-inference-throughput-rtx4070s-20260827](../benchmarks/cnn-inference-throughput-rtx4070s-20260827/README.md).

## The production node

`38.49.42.120:53893` runs the live four-day production run and is not an experiment node.

## Before you connect

Connect with `deployment/remote_command.sh <HOST[:PORT]> <command …>` (it owns the key and the
connection options), not a hand-written `ssh` line. Read `/etc/vast-agents-guide.md` on the node before
changing it. Runs go through `deployment/run_control.sh` only.

Three things that cost time on the 2026-08-27 nodes and will cost it again:

1. **Fetch every result as it completes, not at the end of the session.** A node released before its
   output is fetched takes the evidence with it; the batch-320 and batch-64 rung measurements in the
   attention-viability note survive only as transcriptions for exactly this reason.
2. **Detach long jobs** with `setsid … < /dev/null > log 2>&1 &`. The SSH channel stays open until every
   descriptor is redirected, so a plain `nohup … &` still blocks the caller.
3. **Never `pkill -f` a pattern that also matches your own polling shell**, and check for foreign
   workloads with `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` before
   trusting any timing. A node advertised as dedicated had another work stream's job on it.
