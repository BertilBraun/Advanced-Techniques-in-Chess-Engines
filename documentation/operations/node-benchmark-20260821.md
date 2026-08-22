# Node comparison — how to run it

The four rented nodes cannot be reached from a Cowork cloud session (no route to Vast hosts, no SSH client, and the
`vast-ssh` key is local to your machine), so the benchmark has to be driven from a Claude Code session on your
workstation (or by hand). Everything needed is committed:

- `deployment/benchmark_node.sh` — runs on a node: hardware facts, PCIe/host bandwidth, per-GPU inference
  (batch 1/16/64/256), self-play search throughput for CNN 12×144 and attention 8×128 across
  {2, 3} processes per GPU × `parallel_searches` {1, 4}, 60 s each, and fixed-batch trainer throughput; writes
  `SUMMARY.md` with searches/s per node, per GPU and per dollar.
- The four node lines you gave, with the price per hour to be filled in.

## Prompt for the local Claude Code session

> Read `CLAUDE.md`. Four Vast nodes are rented for a hardware comparison; I decide afterwards which to keep. Use the
> `vast-ssh` key for all of them (`ssh -i ~/.ssh/vast-ssh …`; never copy the key anywhere). Nodes and instance prices:
>
> - `rtx3060x2`: `ssh -p 20391 root@122.228.216.178`, $<price>/h, 2× RTX 3060
> - `rtx4070s`: `ssh -p 21909 root@45.77.214.165`, $<price>/h, 1× RTX 4070 Super
> - `rtx4070`: `ssh -p 10162 root@80.59.54.98`, $<price>/h, 1× RTX 4070
> - `rtx3090`: `ssh -p 17110 root@37.191.136.34`, $<price>/h, 1× RTX 3090
>
> Start one sub-agent per node, in parallel, each with exactly this job:
>
> 1. SSH in, read `/etc/vast-agents-guide.md`, record `nvidia-smi`, driver, effective CPUs (`nproc` and
>    `/sys/fs/cgroup/cpu.max`), RAM, disk, and whether the GPU is idle. Abort and report if the GPU is not idle or
>    the driver cannot serve CUDA 12.6.
> 2. Provision with `deployment/setup_remote.sh` at revision `master` (`ENGINE_REPOSITORY_REF=master`,
>    `ENGINE_REPOSITORY_DIRECTORY=/workspace/alphazero-engine`, `ENGINE_VIRTUAL_ENVIRONMENT=/workspace/alphazero-engine-venv`,
>    runner command `/bin/true`). The script builds the Release extension and smokes the engines; do not skip the
>    smoke. If the node lacks `jq`, install it (`apt-get install -y jq`).
> 3. Run, from `/workspace/alphazero-engine` inside the venv:
>    `NODE_LABEL=<label> HOURLY_PRICE=<price> bash deployment/benchmark_node.sh`
>    It takes roughly 25–35 minutes (2 models × 4 grid points × 60 s self-play, plus inference and trainer
>    passes). If a step fails, read its log in the output directory, fix only what is environmental (missing
>    package, path), re-run that step by hand with the same arguments, and note it; do not change benchmark
>    parameters between nodes.
> 4. `scp` the whole output directory back to `.codex-diagnostics/node-comparison-20260821/<label>/` and return
>    `SUMMARY.md` plus `hardware.txt` verbatim, with a one-paragraph note on anything unusual (shared host load,
>    PCIe x4 links, CPU quota below `nproc`, thermal throttling in `nvidia-smi -q -d PERFORMANCE`).
>
> When all four are back, build one table: per node — GPUs, effective CPUs, RAM, PCIe link, price/h; for each model
> — best self-play searches/s (which grid point), searches/s per GPU, searches per dollar; inference batch-64
> latency and throughput; trainer samples/s. Then give me: (a) throughput per dollar ranking for self-play (the
> binding constraint after the first hours of a run), (b) throughput per wall hour ranking, (c) which node you'd
> keep for Phase A development (needs DDP → at least two GPUs, or the cheapest single GPU if the 2×3060 is
> CPU-starved) and which GPU class to look for in the 8-GPU Phase B node. Do not release any node; I decide.

## What to look for in the results

The decisive number is self-play searches per second per dollar at the best grid point; per-GPU searches/s tells
you the class, per-dollar tells you what to rent. Expect the attention model to be more GPU-bound and the CNN more
PCIe/CPU-bound at batch 64, so a node with fast GPUs and a slow PCIe link (x4, gen 3) will look good on the trainer
and bad on self-play — that is the failure mode to catch. Also check `cpu.max` versus `nproc`: a 2×3060 host
advertising 16 threads but quota'd to 8 will starve three processes per GPU.

Fill in the four `<price>` fields from `vastai show instances`; the per-dollar column needs them.
