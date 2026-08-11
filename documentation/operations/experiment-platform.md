# Experiment platform

This is the current operational entry point for configuring, provisioning, launching, observing, and collecting
AlphaZero experiments. The architecture rework ledgers are historical design records; consult them only for an
architectural change or a concrete design ambiguity.

## Architecture at a glance

Python owns typed configuration, run preparation, the coordinator, replay credits, two-rank DDP training,
self-play, elapsed-time evaluation, TensorBoard, and result export. Native C++ owns game rules, representations,
and batched search. `py/train.py` runs one approved experiment. `py/queue_experiments.py` owns resource slots and
starts stable children through `py/run_approved_experiment.py`; the queue is supervision, not another training
mode.

Experiment YAML has one canonical resolved form. A small authored override may use `extends`; paths resolve relative
to the child YAML, mappings merge recursively, lists replace, and changing a base invalidates every dependent
approval. The current production baseline is `py/configs/baselines/vast-go-9x9-2gpu-4h.yaml`, its removal
ablations are under `py/configs/screening/go-9x9-strong`, and the canonical queue is
`py/configs/queues/vast-go-9x9-screening.yaml`. See [`py/README.md`](../../py/README.md) for the schema and entry-point
details.

The 9x9 baseline is an integrated wall-clock recipe rather than a minimal algorithmic control: 8x96 global-pooling
network, AdamW `0.01` to `0.001` through generation 150, full searches 64 to 256 and fast searches 16 to 64 through
generation 50, full-search probability 1.0 to 0.25 through generation 25, 0.6 tree retention, balanced restart
states, reduced-parent FPU 0.2, forced playouts, delayed root-value blending to 0.15, and next-policy plus
remaining-length auxiliary heads. Challengers remove one treatment at a time; fixed-budget, no-mixed-search,
replay-ratio-8, and constant-LR-0.007 runs complete the authored twelve-run campaign.

## Training and evaluation lifecycle

Run preparation verifies a clean exact source revision, approval, hardware/runtime contract, output paths, and
immutable evaluation inputs. The coordinator starts self-play, ingests completed games into replay, grants trainer
credits at the configured replay ratio, publishes inference checkpoints after DDP quanta, schedules evaluation,
and records a durable outcome on clean shutdown. Six of eight self-play workers pause during each training quantum
in the current two-GPU baseline.

Evaluation belongs to the coordinator and is scheduled by elapsed 20-minute boundaries. Jobs are asynchronous,
device-cycled, limited to ten concurrent jobs, and individually time out after 20 minutes. The 9x9 ladder contains
the book-derived fixed dataset; previous 20-, 40-, and 60-minute checkpoints; the same-time baseline; and KataGo at
96 visits.
Checkpoint and same-time matches use 200 paired openings (400 games); KataGo uses 50 pairs (100 games). The checked-in
v2 dataset and openings are immutable and must be reused, never regenerated. Exact engine assets, checksums, and
artifact preparation checks are in [Evaluation engines](evaluation-engines.md).

For a screening campaign, run a fresh baseline first. Its
`py/training_data/screening/go9-strong/00-baseline/evaluations/reference-checkpoints.json` and referenced
checkpoints are the only runtime same-time baseline. Historical archives are analysis evidence, not runtime input.
Give the baseline roughly two to five minutes of lead before adding challengers, then confirm at their first
boundary that `same-time-baseline` jobs were scheduled rather than skipped.

## Hardware gate, fresh-node bootstrap, and approval

Before renting, compare the offer's GPU model/count/VRAM, CPU allocation, RAM, disk capacity and throughput,
network, reliability, maximum duration, and hourly price with the intended experiment topology and budget. After
connecting but before bootstrap, read `/etc/vast-agents-guide.md`; record the actual `nvidia-smi` inventory and
topology, CPU affinity/NUMA and quota, RAM, disk, network, driver, and image; and run a bounded pinned CPU check when
host contention is in doubt. Stop before `deployment/setup_remote.sh` if measured hardware, available resources,
or cost do not support the authored slots. Marketplace claims are selection hints, not validation evidence.

`deployment/setup_remote.sh` is the authoritative fresh bootstrap. Give it the exact revision and runner command;
it creates the locked virtual environment, builds the Release native extension, installs engines, exposes the
PyTorch NVIDIA libraries during engine smoke, and exports `ENGINE_SOURCE_REVISION`. Omit KataGo archive variables to
use the checked-in pinned CUDA default. A different official CUDA/cuDNN release requires all three
`ENGINE_KATAGO_BACKEND`, `ENGINE_KATAGO_ARCHIVE_URL`, and `ENGINE_KATAGO_ARCHIVE_SHA256` values. CPU, OpenCL, and
TensorRT KataGo builds are forbidden. Success requires `engines/INSTALLATION.txt` to name CUDA, `katago version` to
report the CUDA backend, and both board-size smokes to pass. Exact assets and hashes remain in
[Evaluation engines](evaluation-engines.md).

For later supervised commands, preserve the bootstrap runtime requirements: raise the soft open-file limit above
the configured minimum, set `TRAINING_RUNTIME_IMAGE`, and include every locked-venv `site-packages/nvidia/*/lib`
directory in `LD_LIBRARY_PATH`. Otherwise KataGo children can fail to load CUDA libraries even though bootstrap
passed.

Every production run needs a new approval JSON outside Git. Create it only after the final revision and resolved
configuration are fixed. It must bind the approver, exact Git revision, canonical configuration hash, offer, hourly
price, and wall-time limit. Never reuse an approval from another revision, configuration, node, or campaign.

## Queue ownership and operation

The queue owns slot allocation, per-experiment detached Git worktrees, child process groups, sampled process-tree
RAM limits, logs, and the atomic summary.
It reloads the desired queue and pending experiment YAML before every scheduling pass. Adding, removing, reordering,
or editing pending entries does not require a restart. Each pending entry names an exact source commit. When assigned
a slot, the queue creates one detached worktree, builds that revision, and launches the child from it; it never
prebuilds pending work. Running and terminal experiment IDs are immutable in configuration and revision; a failed
entry releases its resources and is not silently retried.

Training artifacts use the queue's central runtime directory and TensorBoard uses its central log root, both outside
the disposable worktree. After a successful exit and complete process-tree shutdown, the queue preserves the
authored configuration chain and workspace provenance under `.queue-evidence`, then deletes that worktree. It keeps
failed or incompletely preserved worktrees for diagnosis. The control checkout can therefore be pulled and used for
future development while active experiments remain pinned to their own revisions.

The supervisor snapshots its queue-owned Linux launch helpers into the persistent runtime root when it initializes.
Future launches use that immutable supervisor snapshot, not helper files in the pullable control checkout. Desired
queue YAML and pending experiment YAML remain the only intentionally live-read control-checkout inputs.

### Add new code while experiments are running

Active experiments no longer pin the control checkout. To make a new implementation available to later queue
entries:

1. Develop and validate the code and its complete experiment-configuration inheritance chain in the control
   checkout, then commit them.
2. Put that commit's full 40-character SHA in the pending entry's `source_revision` and create a fresh approval for
   that exact revision and resolved configuration.
3. Atomically replace the live queue YAML. The supervisor validates the committed configuration chain immediately,
   but creates and builds the detached worktree only when a slot is assigned.
4. Confirm the queue summary records the intended `source_revision`, `source_worktree`, and central
   `runtime_directory` when the experiment starts.

Pulling or otherwise advancing the control checkout does not change any running experiment, and it does not
implicitly advance a pending entry: change the pending `source_revision` and its approval explicitly. Pending
entries may be added, removed, reordered, or revised through live reload; never change the identity of a running or
terminal entry. Setup and compilation happen after slot assignment and therefore occupy that slot briefly.

On success, the supervisor waits for tracked descendants to exit, preserves configuration and workspace provenance
under the central runtime root, and removes the disposable worktree. It deliberately retains worktrees for failed,
terminated, setup-failed, or preservation-failed experiments; inspect and export the evidence before removing them
manually. Result export continues to work after successful cleanup by using the preserved configuration snapshot.

Use the node's supported supervisor instead of a detached shell. The supervised command is:

```bash
/workspace/alphazero-engine-venv/bin/python py/queue_experiments.py run \
  --queue-config /workspace/run-control/r15/go9-screening-live.yaml
```

For a baseline-first launch, materialize the committed queue outside Git with absolute experiment paths and exact
source revisions. Initially
include only the baseline; after its evaluation manager and reference manifest exist and the lead interval has
elapsed, atomically replace the live file with the complete materialized queue. Create fresh revision-bound approvals
for every entry. The control checkout may advance afterward; each child remains isolated at its recorded revision.

Start and inspect the supervisor with the node's `supervisorctl`:

```bash
supervisorctl start experiment-queue
supervisorctl status experiment-queue
/workspace/alphazero-engine-venv/bin/python py/queue_experiments.py status \
  --summary /workspace/run-control/r15/go9-screening-summary.json
```

Update only pending work by atomically replacing the live queue file. Stop with
`supervisorctl stop experiment-queue` only when active runs should also terminate: queue shutdown sends termination
to every active child. Restarting against a summary that still says `running` marks those entries failed rather
than adopting stale processes, so first verify all recorded process groups are gone. Deleting the summary creates a
new campaign and makes configured IDs eligible again; do that only deliberately with clean run, log, and TensorBoard
paths.

## Monitoring and TensorBoard

Inspect the queue summary, per-slot stdout/stderr, run outcome, credit ledger, evaluation manager state/results,
resource telemetry, `nvidia-smi`, host RAM, and disk. A healthy four-slot launch shows four independent two-GPU
children progressing in games, credits, optimizer steps, generations, checkpoints, and TensorBoard. Watch replay
ratio and available-quantum fraction together, and distinguish credit waits from training time. Check GPU memory,
temperature, power, and utilization across self-play, DDP, and evaluation contention.

The node TensorBoard service listens on port 16006 and watches `/workspace`. Forward it separately, for example:

```powershell
ssh -i C:\path\to\vast_key -p SSH_PORT root@HOST `
  -L 6006:localhost:16006
```

Open `http://localhost:6006`. Evaluation custom scalars group match W/D/L and scores, dataset accuracy/cross-entropy,
and one combined duration chart. Compare experiments on elapsed time, with optimizer steps and generations at each
boundary as supporting context.

## Result export and validation checklist

Export terminal entries with `py/export_experiment_results.py`. A full archive contains authored/resolved configs,
identity/outcome, telemetry, queue and child logs, evaluation state/results, TensorBoard, every elapsed evaluation
checkpoint, and only the latest optimizer; replay and completed games stay remote. Exporting does not delete source
directories. Commands, selection rules, and manifest semantics are in
[Experiment result export](experiment-result-export.md). Download the ZIP to the project evidence directory, verify
its SHA-256 and ZIP integrity, and retain both local and remote copies until the campaign is accepted.

Before launch, verify:

- exact clean revision, fresh approvals, measured slot resources, locked dependencies, Release native tests, and
  CUDA KataGo/Stockfish integration;
- v2 dataset/opening hashes reuse immediately, focused/full Python tests, and a bounded real DDP smoke covering
  credits, prefetch, pause/resume, checkpointing, scheduled evaluation, TensorBoard layout, and clean shutdown;
- empty production run/TensorBoard/log/summary paths, baseline initialization before challenger live reload, and
  four concurrent progressing slots through the challengers' first evaluation boundary.

After completion, verify terminal outcomes, every due evaluation result or explicit failure, archive manifest and
ZIP integrity, elapsed-time comparisons against the fresh same-time baseline and previous checkpoints, fixed-dataset
metrics, KataGo-96, replay/credit balance, throughput, and resource contention. Do not promote a feature
automatically; report the evidence for a user decision. Detailed prior-node validation evidence remains in
[R11 Vast integrated validation](vast-r11-validation.md), and the historical baseline rationale is in
[Go 7x7 two-GPU training baseline](../benchmarks/go-7x7-two-gpu-training-baseline.md).
