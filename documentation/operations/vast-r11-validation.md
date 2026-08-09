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
two match games and two searches/engine visits per move. Chess completed fixed-dataset, policy-vs-random,
search-vs-random, Stockfish, and previous-checkpoint jobs at early boundaries. Go completed the corresponding
native jobs; a separately isolated Go 7x7 KataGo match completed two capped games in 15.97 seconds. Almost all of
that isolated duration was KataGo's roughly 15-second CUDA graph warm-up.

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

### Current self-play/search throughput

The game-generic benchmark drives the production `SelfPlayWorker` and normalized native search binding with one
process per GPU, 64 parallel games per process, and a 60-second measurement window. Benchmark models use the real
production architecture with zeroed weights, so the figures measure the same inference/search shape without
claiming playing strength.

| Game/model | Searches | Makespan | Searches/second | Mean inference batch |
| --- | ---: | ---: | ---: | ---: |
| Chess, 12×112 | 1,224,837 | 61.26 s | 19,993.92 | 27.39 |
| Go 7x7 smoke network | — | 60 s window | 95,754.34 | 61.09 |
| Go 9x9 smoke network | — | 60 s window | 76,216.56 | 63.20 |

The chess evidence is
`r11-benchmarks/current-independent/self-play-search-chess-gpus4-processes1-games64-60s-20260809T152032Z`.
Go's much smaller boards/networks explain their higher rates; these are useful within-game baselines, not a direct
strength-normalized comparison.

### Pre-rework comparison

The historical comparison uses commit `bb334eb8c5c48eaa686c4d9a9a43c20b3cac10e4`, the last Python-owned runtime
before the formal architecture-rework sequence, built in a detached worktree on the same node. Its benchmark
reporter required two diagnostic-only corrections outside the historical checkout: pass the required run config,
and restore eight result fields that were accidentally indented under the publisher class. Neither correction
changes the historical runtime or native search.

Three matched short runs show parity at one process per GPU:

| Topology | Current mean | Pre-rework mean | Current delta |
| --- | ---: | ---: | ---: |
| 4 GPUs, 1 process/GPU, 64 games/process | 16,732.88 searches/s | 16,586.90 searches/s | +0.88% |
| 4 GPUs, 8 processes/GPU, 96 games/process | 41,007.77 searches/s | 49,174.23 searches/s | -16.61% |

The one-process result indicates no material rewrite regression in the search/inference path itself. The repeatable
16.61% loss appears only under 32-process contention and is therefore a process/topology scaling regression to
profile before freezing the production worker count. Do not tune from the single five-second figure alone; rerun a
longer scaled comparison after profiling. A 60-second historical run cannot complete because the old runtime fails
after natural chess game completion with `ChessSearchObservation` visits that reference illegal actions. That
historical defect is itself evidence against using the old runtime as a long-duration baseline.

### Validation and remaining gate

At source revision `a62219a422ef2e191fcc2ead65c652e63b703901`:

- `python -m pytest --import-mode=importlib test -q`: 200 passed, 6 skipped, 59 warnings in 21.45 seconds.
- `ctest --test-dir cpp/build --output-on-failure`: 1/1 native test target passed in 1.34 seconds after configuring
  the validation build with `BUILD_TESTING=ON`.
- Focused external-engine and evaluation-manager tests: 10 passed in 4.14 seconds.

The only infrastructure validation still blocked on this offer is R10's real resource-aware queue launch because
the container exposes read-only cgroup v1 rather than delegated cgroup v2. The functional R11 path, engines,
datasets, evaluation lifecycle, timeout, and current/pre-rework throughput comparison are complete. The scaled
multi-process regression and a production-sized evaluation timing run remain benchmark/topology follow-ups, not
functional blockers.

The complete ignored evidence archive is `.codex-diagnostics/r11-final-evidence.tar.gz` on the development host
(418,114 bytes, SHA-256 `2e92fc4aaef67e6707770ae91d5b6288aa48e0295de46a5c63891590f00d0682`). It contains
the smoke/timeout logs, result artifacts, benchmark JSON and logs, persistent reference inputs, diagnostic-only
historical benchmark wrappers, and final pytest/CTest output. Preserve it before destroying the instance.
