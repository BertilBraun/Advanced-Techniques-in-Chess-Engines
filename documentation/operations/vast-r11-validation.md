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
| CPU | Host exposes 2× AMD EPYC 7B12 and 256 logical CPUs; this rental owns 85.33 effective CPUs |
| RAM | Host exposes 251 GiB; this rental owns 86 GiB |
| Disk | 150 GiB ephemeral NVMe-backed container storage |

The listing was expected to be an RTX 4070 offer, but the live allocation is four RTX 3060s. Configuration and
evidence use the observed devices, not the advertised assumption.

The host inventory is not the experiment's resource entitlement. Other tenants may use the logical CPUs and RAM
outside the 85.33-CPU/86-GiB rental allocation. Do not size a run from `lscpu`, `free`, host-wide CPU percentages, or
host-wide free-memory readings alone. Every run must stay within an explicit slot carved from the rental allocation.

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

The bootstrap exports the CUDA libraries bundled with the locked PyTorch wheels. Ad-hoc engine commands outside
the bootstrap runner must reproduce that environment or KataGo fails to load `libcublas.so.12`:

```bash
nvidia_root=/workspace/alphazero-engine-venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$(find "${nvidia_root}" -mindepth 2 -maxdepth 2 -type d -name lib -print | paste -sd:):${LD_LIBRARY_PATH:-}"
```

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

## Evaluation artifact persistence

Dataset and opening preparation is create-if-missing, not per-startup work. Once an immutable manifest and its
payload exist under `py/reference/`, later runs validate and reuse them. Because this offer's `/workspace` is
ephemeral, restore the artifacts before launching a replacement node. The validated copies are retained outside
Git in `.codex-diagnostics/r11-reference-artifacts/` on the development host.

| Game | Dataset | Positions | Source games | Dataset SHA-256 |
| --- | --- | ---: | ---: | --- |
| Chess | `vast-r11-chess-evaluation-v1.bin` | 520 | 11 | `3b6321d3c47ece5b106dce4ca7130bea9a10414df269154ffc3cac5300df232f` |
| Go 7x7 | `vast-r11-go-7x7-evaluation-v2.bin` | 491 | 28 | `225df3705f027b5fb544839a3410e97f496e3ca336f5cc17ac3a910cb7d5ed36` |
| Go 9x9 | `vast-r11-go-9x9-evaluation-v2.bin` | 484 | 18 | `08fb1e295a7b8aa29063ca9b0d2cafdb72084bafbefd1701de48b9c1879b9d0f` |

Do not commit these generated reference artifacts by default. Store them as versioned run inputs in persistent
artifact storage when a production baseline is frozen; Git remains an option only if the project deliberately
wants binary reference data in repository history.

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

### End-to-end smokes

Chess, Go 7x7, and Go 9x9 each completed five generations concurrently with exit code zero. The runs exercised
preparation/reuse of evaluation inputs, native self-play/search, Python inference and training, checkpoint
publication, asynchronous evaluation scheduling, result JSON, and TensorBoard output. The scaled definitions used
two match games and two searches/engine visits per move.

The original five-generation Go runs predated the capped-position scoring fix. Their training lifecycle passed,
but their policy-random and search-random artifacts failed with `Go adjudication requires a scored terminal
position`, and their KataGo jobs were cancelled when the intentionally short training run ended. Do not count those
original match artifacts as successful evaluation evidence. Commit `addb2e9e` added deterministic area scoring for
capped Go matches; the post-fix timeout run, isolated jobs, and the manager exercise below provide the authoritative
match evidence.

| Evaluation path | Chess | Go 7x7 | Go 9x9 | Authoritative result |
| --- | --- | --- | --- | --- |
| Fixed engine-labelled dataset | Passed, 520 positions | Passed, 491 positions | Passed, 484 positions | All three payloads loaded and produced policy accuracy/cross-entropy results. |
| Policy-only vs random | Passed | Passed post-fix | Passed post-fix | Two balanced games completed for every game/board variant. |
| Search vs random | Passed | Passed post-fix | Passed post-fix | Two balanced native-search games completed for every game/board variant. |
| Candidate vs checkpoint | Passed, generation 1 vs 0 | Passed post-fix | Passed post-fix, generation 1 vs 0 | The shared checkpoint-opponent runtime completed both colors/player orders. |
| External engine | Stockfish level 0 passed | KataGo passed in 15.97 s | KataGo passed in 16.40–16.49 s | UCI and KataGo analysis boundaries both produced match results. |

Stockfish levels 1–3 were not separately played in the tiny smoke; they use the same tested UCI client and differ
only in the validated `UCI_LimitStrength`/skill configuration. Likewise, fixed generations 10, 20, and later could
not be scheduled by a five-generation run. They use the same tested checkpoint-opponent match path; scheduling
tests verify that only existing older generations are launched. These are configuration-variant gaps, not untested
runtime boundaries.

A dedicated two-boundary Go 9x9 `EvaluationManager` run launched the complete suite in parallel. Boundary 2
completed fixed-dataset, policy-random, search-random, and KataGo. Boundary 4 repeated those jobs for generation 1
and also completed generation 1 against boundary generation 0. All nine artifacts were successful, the manager
collected and reported them later, and no evaluation or KataGo process remained. This proves KataGo inside the real
manager lifecycle rather than only as a direct isolated job.

The 30-second wall-time smoke stopped for `maximum wall time reached`, returned exit code zero, and left no training,
evaluation, or KataGo processes behind. The final process-group implementation recorded 44.94 seconds from
coordinator start to persisted outcome (51 seconds including command/preflight startup). This intentionally harsh
configuration schedules five evaluation jobs every two seconds on one GPU, so its approximately 15-second bounded
shutdown overhead is not representative of the 20-minute production cadence. Evaluation children and their
external-engine subprocesses now share a job process group; shutdown gives all running jobs one common grace period
and then force-stops surviving groups.

### Evaluation timing estimate

The tiny chess smoke jobs completed in approximately 1.0–2.0 seconds for fixed-dataset, policy-random,
search-random, and Stockfish level 0, and approximately 3.1 seconds for a previous-checkpoint match. These are
functional timings, not production estimates: they use only two games and two searches per move. A production
suite uses roughly 500 fixed positions, 50 paired openings (100 games), 64 candidate searches per move, four
Stockfish levels, recent-boundary checkpoints, and retained ten-generation checkpoints.

With jobs launched in parallel and cycled across four evaluation GPUs, reserve **5–10 minutes per evaluation
boundary** as the initial operational estimate. KataGo adds a fixed approximately 16-second startup to each KataGo
job. This estimate is deliberately provisional; measure a complete production-sized boundary once the final
topology is frozen. The configured 20-minute job timeout should make an unexpectedly slow or contended job visible
without permitting silent overlap indefinitely.

### Evaluation checkpoint retention

Evaluation does not depend on retaining every full training checkpoint. A published checkpoint has separate model,
optimizer, and trimmed inference artifacts. The chess template keeps resumable model and optimizer artifacts for
generation 0, the active generation, and every third generation. Separately, it keeps trimmed inference artifacts
for generation 0, the latest 11 generations, and every tenth generation. The evaluation manager additionally marks
all configured fixed-checkpoint generations, recent elapsed-boundary opponents, and checkpoints referenced by
pending jobs as required before the coordinator applies retention.

Consequently, the configured fixed generations 10, 20, ..., 100 remain evaluable for the life of the run even after
their full optimizer state is pruned. The three previous elapsed-boundary opponents are also protected independently
of their generation numbers. No retention change is required for the current evaluation ladder; restoring training
from every fixed evaluation generation would be a separate and substantially larger storage policy.

### Current self-play/search throughput

Training and inference batch sizes are independent. `global_batch_size` is the number of replay rows consumed by one
DDP optimizer step, while `local_batch_size` is the per-rank portion. Configuration validation requires
`global_batch_size == local_batch_size * len(ddp_device_ids)`. The generic chess template is still a one-rank local
template and therefore currently says 2,048 for both values. A four-GPU production configuration must explicitly
set a global batch of 2,048, a local batch of 512, four DDP device IDs, CUDA/NCCL, and the node topology. That
production configuration has not yet been frozen; the self-play benchmarks below do not exercise DDP training.

The game-generic benchmark drives the real production `SelfPlayWorker`: scheduled search parameters, normalized
native multi-game search, TorchScript inference, move selection, tree updates, terminal handling, and completed-game
publication. It excludes replay ingestion, optimizer work, and evaluation contention, so it is a self-play capacity
benchmark rather than a complete training-throughput benchmark. Benchmark models use the production network shape
with zeroed weights; they measure identical compute without claiming playing strength.

The accepted historical topology was not roughly 1,000 games per process. It was four processes per GPU and 192
games per process: 16 processes and 3,072 active games across four GPUs. R11 held total active games at 3,072 and
varied only their process distribution. Every current matrix run used CPU pinning, two direct-inference workers,
batch size 64, two outstanding batches per worker, the generation-0 300/75 full/fast search schedule, and the same
12×112 model.

| Processes/GPU | Games/process | Window/makespan | Searches/s | Mean batch | Aggregate process CPU | Mean host CPU | Mean GPU utilization |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 96 | one short batch, 9.28 s | 40,613.98 | 19.71 | 5,080% | not sampled | not authoritative |
| 4 | 192 | 30 s / 32.34 s | 59,715.93 | 26.89 | 3,179% | 14.17% | 90.33–96.50% |
| 2 | 384 | 30 s / 34.57 s | **67,560.50** | 36.70 | 1,618% | 9.00% | 91.16–92.97% |
| 1 | 768 | 30 s / 34.74 s | 67,152.53 | 55.83 | 810% | 5.58% | 71.55–77.09% |

Two processes per GPU are 13.14% faster than four and 66.35% faster than the earlier eight-process diagnostic. One
process nearly matches raw throughput through larger inference batches, but leaves substantially less GPU
utilization margin. Two processes are therefore the recommended baseline on this four-RTX-3060 node.

The confirming mature-generation run used two processes/GPU, 384 games/process, the generation-20 600/150 search
schedule, and a requested 60-second window:

| Searches | Makespan | Searches/s | Searched plies/s | Mean batch | Aggregate process CPU | Mean host CPU | Mean GPU utilization | Peak GPU memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 4,668,190 | 67.23 s | **69,439.93** | 274.17 | 36.74 | 1,644% | 9.23% | 95.79–97.84% | 531 MiB/device |

The requested high-concurrency follow-up kept 380 games in every worker and changed only the number of workers per
GPU. These runs intentionally raised total active games instead of holding the prior 3,072-game total constant:

| Processes/GPU | Games/process | Total games | Searches/s | Mean batch | Aggregate process CPU | Mean host CPU | Summed worker peak RSS | Mean GPU utilization | Peak GPU memory |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2 | 384 | 3,072 | **69,439.93** | 36.74 | 1,644% | 9.23% | not recorded | 95.79-97.84% | 531 MiB/device |
| 3 | 380 | 4,560 | 68,608.30 | 36.25 | 2,449% | 10.95% | 18,403.59 MiB | 93.10-96.02% | 794 MiB/device |
| 4 | 380 | 6,080 | 68,976.85 | 36.19 | 3,250% | 14.55% | 24,328.13 MiB | 94.63-96.45% | 1,055 MiB/device |

Three and four processes use approximately 7.2% and 9.5% of the node's 251 GiB RAM respectively when summed from
each worker's peak RSS; each worker peaks near 1.5 GiB. CPU and RAM therefore remain comfortable at four processes,
and all topologies saturate the GPUs. The additional games do not improve mean batch size: both new runs remain near
36.2, effectively the same as 2x384. Their throughput is 1.20% and 0.67% below 2x384 respectively, well within the
range where repeat-run variance could change their order but not evidence of a material gain.

The unexpectedly low average batch has one concrete cause. Every process has two inference workers and permits two
outstanding batches per worker, giving four concurrent inference slots. Only 25% of games receive the full 600-search
budget; the other 75% finish after 150 searches. With 380 games, the long full-search tail therefore contains only
about 95 roots, or about 24 independent positions per slot. The recorded histograms agree: approximately 31% of
calls are full batches of 64 and approximately 63% are below 32. With sequential tree search, the approximate number
of games needed to fill the tail is `batch size * inference slots / full-search probability`.

The follow-up increased independent games without enabling parallel leaves:

| Processes/GPU | Games/process | Batch limit | Mean batch | Full calls | Searches/s | Approx. batch latency | Host CPU | Summed worker peak RSS | Mean/peak power |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 384 | 64 | 36.74 | not recorded | 69,439.93 | 11 s | 9.23% | not recorded | not recorded |
| 2 | 768 | 64 | 55.89 | 57.28% | 68,224.86 | 23 s | 7.46% | 14.48 GiB | 50.85/62.72 W |
| 2 | 1,024 | 64 | **63.53** | **92.45%** | **76,719.77** | 27 s | 8.63% | 16.37 GiB | 54.29/63.63 W |
| 4 | 1,024 | 64 | 63.32 | 91.33% | 76,353.73 | 54 s | 18.35% | 31.69 GiB | 54.69/66.56 W |
| 2 | 2,048 | 128 | 126.97 | 92.86% | 71,453.58 | 57 s | 8.11% | 22.95 GiB | 55.43/68.22 W |

The 2x1,024 topology is 10.49% faster than 2x384 and fills essentially every useful 64-position inference batch
without changing MCTS semantics. Doubling the process count provides no throughput gain and doubles CPU, RAM, and
batch latency. Doubling both games and the batch limit fills batches of 128 but is slower and makes lifecycle
response too coarse. The isolated self-play recommendation is therefore 2 processes/GPU, 1,024 games/process,
inference batch size 64, two inference workers, two outstanding batches per worker, and one parallel search.

Despite the filled batches, power remains around 54 W rather than the 120 W training level. A batch of 64 for the
small 8x8 inference network is still much less compute-dense than a local training batch of 512 with forward and
backward passes. The full batches improve useful throughput, but NVIDIA's utilization percentage continues to
overstate arithmetic saturation.

The subsequent Go 7x7 sweep reached 446,725.00 searches/s with four processes/GPU, 1,024 independent games/process,
batch 512, and sequential tree search. The complete baseline-to-optimized comparison, causal analysis, frozen chess
and Go settings, and raw evidence are recorded in the
[self-play throughput baseline](../benchmarks/self-play-throughput-rtx3060.md).

### Pre-rework comparison

The historical comparison uses commit `bb334eb8c5c48eaa686c4d9a9a43c20b3cac10e4`, the last Python-owned runtime
before the formal architecture-rework sequence, built in a detached worktree on the same node. Its benchmark
reporter required two diagnostic-only corrections outside the historical checkout: pass the required run config,
and restore eight result fields that were accidentally indented under the publisher class. Neither correction
changes the historical runtime or native search.

The authoritative same-node comparison now uses the historical training topology and a complete 30-second window:

| Topology/workload | Current | Pre-rework | Current delta |
| --- | ---: | ---: | ---: |
| 4 GPUs, 4 processes/GPU, 192 games/process, 300/75 searches | 59,715.93 searches/s | 60,458.72 searches/s | -1.23% |
| Mean inference batch | 26.89 | 26.87 | +0.08% |
| Aggregate process CPU | 3,179% | 3,166% | +0.41% |
| Mean host CPU utilization | 14.17% | 14.27% | -0.10 percentage points |
| Mean per-GPU utilization range | 90.33–96.50% | 90.00–94.57% | comparable |

The 1.23% difference is practical parity. The earlier 16.61% loss was real for the eight-process/GPU topology, but
that topology is poor for both runtimes and should not be used to characterize the rewrite. At the useful 4×192
topology, search rate, batch size, CPU use, and GPU saturation are all equivalent.

A requested 60-second historical 4×192 run again reproduced the old natural-terminal defect: four workers failed
near the end with `ChessSearchObservation` visits referencing illegal actions. The other 12 ran for 61–63 seconds,
but their survivor-only throughput is not used for comparison. Current completes the full 60-second topology. The
30-second historical window above is the longest complete, directly comparable result on this code state.

The benchmark result does not yet include simultaneous replay, training, and evaluation contention. Before freezing
an R12 production configuration, run one integrated baseline with the proposed `[0, 0, 1, 1, 2, 2, 3, 3]`
self-play device assignment and 1,024 games per worker, then verify that trainer/evaluation scheduling and the
approximately 27-second worker batch latency do not delay lifecycle operations unacceptably. Do not copy the
node-specific topology into the generic local template without an explicit production run configuration and
approval.

### Validation and remaining gate

Runtime validation completed at `a62219a422ef2e191fcc2ead65c652e63b703901`; the resource-aware benchmark
harness, batching overrides, and this corrected report are on `bdd6059e03e96232d6e3cfeffd4aaf52ddbc13a6` and its
documentation follow-up:

- `python -m pytest --import-mode=importlib test -q`: 200 passed, 6 skipped, 59 warnings in 21.45 seconds.
- `ctest --test-dir cpp/build --output-on-failure`: 1/1 native test target passed in 1.34 seconds after configuring
  the validation build with `BUILD_TESTING=ON`.
- Focused external-engine and evaluation-manager tests: 10 passed in 4.14 seconds.
- `bash -n py/tools/run_self_play_search_benchmark.sh` passed, followed by a real pinned CUDA harness smoke and the
  topology measurements above.

The only infrastructure validation still blocked on this offer is R10's real resource-aware queue launch because
the container exposes read-only cgroup v1 rather than delegated cgroup v2. The functional R11 path, engines,
datasets, evaluation lifecycle, timeout, and current/pre-rework throughput comparison are complete. The scaled
topology question is resolved for isolated self-play on this node. A production-sized evaluation timing run and an
integrated self-play/replay/training/evaluation contention measurement remain R12 benchmark follow-ups, not
functional blockers.

The complete ignored evidence archive is `.codex-diagnostics/r11-final-evidence.tar.gz` on the development host
(557,092 bytes, SHA-256 `21231f09572d8439177d87baf8f02121e2b7936d50b20949a542d242fc0341e9`). It contains
the smoke/timeout logs, result artifacts, benchmark JSON and logs, persistent reference inputs, diagnostic-only
historical benchmark wrappers, and final pytest/CTest output. Preserve it before destroying the instance.
The requested 3x380 and 4x380 follow-up artifacts are in the supplemental ignored archive
`.codex-diagnostics/r11-topology-3x380-4x380-evidence.tar.gz` (28,529 bytes, SHA-256
`e9aacd2e9699394403e5c7d24f76b2077ab88a3c299d63ed10c8bdc9cc3414e1`).
The independent-game and inference-batch sweep is in
`.codex-diagnostics/r11-batching-sweep-evidence.tar.gz` (44,997 bytes, SHA-256
`f0a18498341169a30794cad1acbcf21f3b769cb459035fdca98e9d8d6f7fed65`).
The Go 7x7 throughput sweep is in `.codex-diagnostics/r11-go7-throughput-evidence.tar.gz` (160,260 bytes, SHA-256
`f49e3eda997a2a041d747e64596ec21e03e55f18a9e1da54bdecf48c89b43259`).
