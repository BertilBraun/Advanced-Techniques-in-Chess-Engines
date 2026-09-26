# Lc0 teacher diagnostic — plan and node runbook

**Branch `worktree-lc0-teacher`. Not for merge.** The chess input becomes Lc0's classical 112-plane
layout, so this project's own networks cannot run on this branch at all. Its output is evidence, not
code to keep.

## Question

The final self-play lineage plateaued at ~2360 benchmark Elo at 64 searches. The learning-rate floor,
label quality, replay diversity and capacity have each been measured and eliminated. This asks the
remaining structural question in two parts:

1. **Phase A — is the harness able to exploit a strong evaluator at a low search budget?** Run this
   project's MCTS, unchanged, on Lc0's evaluator and measure it on the frozen Stockfish 13 ladder.
2. **Phase B/C — can this project's own network absorb that evaluator?** Distil Lc0's raw policy and
   WDL into checkpoint 1026 and re-measure.

Phase A is the gate. If Lc0's evaluator does not play far above 2360 at 64 searches in our own
search, the teacher/search combination cannot test the ceiling and Phase B is not worth running.

### What is already known, and therefore not being retested

The same network and search reach **3251 benchmark Elo at 100,000 searches** (0.530 against
Stockfish 13 at 200,000 nodes) and scale cleanly across four decades. A globally broken harness is
already ruled out. Phase A tests low-budget behaviour specifically: cPUCT, FPU and value scaling were
tuned against this project's policy sharpness, and Lc0's policy is sharper, so a null result there is
more likely to be a search-parameter mismatch than a knowledge result. Sweep those before concluding
anything.

## Teacher selection

Parameters for an Lc0 ResNet-SE network are about `18 x blocks x filters^2`:

| Architecture | ≈ parameters | Use |
|---|---:|---|
| 10x128 | 3.0M | smaller than the 6.26M student; not useful |
| 15x192 | 10.0M | Phase B/C teacher |
| 20x256 | 23.6M | Phase B/C teacher |
| 24x320 | 44M | Phase A only |
| 15x1024 | 283M | Phase A only |

**Use different networks for the two phases.** Phase A wants the strongest teacher available, since
the question is whether the search can exploit strong evaluation at all. Phase C wants a teacher near
the student's 6.26M: this project's own compression benchmark measured a 13.4x size gap costing about
430 Elo at 10,000 searches, so distilling a 283M network into 6.26M would make a null result
uninterpretable. Running Phase A at two sizes also gives the strength-versus-size curve that decides
the Phase C teacher, cheaply.

Networks are listed at `lczero.org/play/networks/bestnets/` and `training.lczero.org/networks/`.
Avoid the BT-series transformer networks: `inspect_lc0_network.py` will reject anything that is not
the classical 112-plane input, which is what the encoder on this branch implements.

## Hardware

One machine, one GPU, for every stage. Phase A at 64 searches is minutes of GPU; at 1,000 searches a
100-game match is roughly an hour even with a large teacher. Data generation costs one evaluation per
stored position. Storage is negligible: records are 875 payload bytes plus a sparse top-64 policy.

- 1x RTX 4090 (24 GiB), or a 4070 SUPER 12 GiB
- 32 vCPU (16 workable) — search-free generation is CPU-bound on move generation
- 64 GB RAM, 100 GB disk

**`nproc` and `free` report the host, not the container.** Vast.ai limits the container through
cgroup v1, so the effective allocation is only visible in the quota files. Two candidate nodes probed
on 2026-09-26 both advertised far more than they grant:

| Node | Advertised | cgroup CPU quota | cgroup memory | GPU |
|---|---|---:|---:|---|
| 85.238.208.233:40711 | 56 vCPU, 251 GB | **13.44 CPUs** | **120.8 GiB** | RTX 3060 Ti 8 GiB, CC 8.6 |
| 83.233.222.244:26204 | 28 vCPU, 62 GB | **13.44 CPUs** | **42.5 GiB** | RTX 3070 8 GiB, CC 8.6 |

Measured, not just read: a CPU-bound Python workload scaled near-linearly to 14 processes (26.6 and
29.0 units/s) and then fell at 28 (21.9 and 12.7 units/s) while the cgroup throttle counters climbed
by 13.2 s and 29.5 s. Anything sized from `nproc` oversubscribes the quota by 2-4x and runs slower
than sized correctly. Both GPUs were dedicated, idle, and downloaded a PyTorch wheel at 58-60 MB/s
sustained.

Derive parallelism from the quota everywhere on the node:

    EFFECTIVE_CPUS=$(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) ))

`deployment/benchmark_node.sh` reads only the cgroup v2 path and reports `unknown` on these nodes.

## Runbook

Everything below runs on the node. Nothing native has been compiled locally, so step 2 is the first
real check and small compile errors there are expected.

### 1. Provision

    deployment/setup_remote.sh <HOST[:PORT]> worktree-lc0-teacher

Install the Lc0 extra and an Lc0 binary with a CUDA backend, never the Eigen fallback:

    uv sync --extra lc0

Record `nvidia-smi`, driver, GPU model, CPU count, RAM and the Lc0 binary version in the provisioning
note.

### 2. Build and run the native tests

    cmake -S cpp -B ~/advanced-chess-compile-check       -DCMAKE_BUILD_TYPE=CompileCheck -DBUILD_TESTING=OFF       -DBUILD_BENCHMARKS=OFF -DENABLE_NATIVE_ARCHITECTURE=OFF
    cmake --build ~/advanced-chess-compile-check --target AlphaZeroCpp --parallel "${EFFECTIVE_CPUS}"

Then a Release build with `NativeTests` enabled, since anything measured or deployed needs Release.

The `Board` placement window and the 112-plane encoder have never been compiled. Fix what the
compiler finds before going further.

### 3. Regenerate the codec fixtures

    cd py && python tools/regenerate_packed_plane_fixtures.py

The committed fixtures were recorded against the 52-plane encoder and their test skips itself until
payload lengths match. This re-arms it.

    python -m pytest --import-mode=importlib ./test -q

### 4. Export and inspect the network

    lc0 leela2onnx --input=<net>.pb.gz --output=<net>.onnx
    python tools/inspect_lc0_network.py --onnx <net>.onnx

Refuses anything that is not (batch, 112, 8, 8) with a 1858-wide policy and a 3-wide WDL head.

### 5. Build the policy permutation

    python tools/build_lc0_policy_map.py \
        --lc0-bitboard-source <lc0-src>/src/chess/bitboard.cc \
        --output artifacts/lc0-policy-map.json

Derived empirically by walking positions, not from a hardcoded table. It fails loudly if any Lc0
index ever maps two ways. Raise `--position-count` until every one of the 1858 indices is covered;
`build_lc0_teacher_model.py` refuses a map with holes.

### 6. Wrap the teacher

    python tools/build_lc0_teacher_model.py \
        --onnx <net>.onnx \
        --policy-map artifacts/lc0-policy-map.json \
        --output artifacts/lc0-teacher.pt

Bakes in the permutation, the rule-50 scaling and the WDL conversion. Swapping to a larger network
later is a repeat of steps 4-6, not a code change.

### 7. Gate: fidelity against real Lc0

    python tools/verify_lc0_teacher_fidelity.py \
        --lc0-binary <lc0> --lc0-network <net>.pb.gz \
        --teacher-model artifacts/lc0-teacher.pt \
        --openings reference/chess-stockfish-8moves-v3-openings-50.tsv \
        --positions 20

**Do not proceed past a failure here.** This asserts the plane order, the history planes, the policy
permutation and the WDL conversion at once, by comparing against Lc0's own root priors on identical
move sequences. The plane order in the encoder is written from documentation rather than from a
verified reference; this is what verifies it. A silent mismatch produces a teacher that looks merely
weak, which is indistinguishable from the result the experiment is trying to measure.

If it fails, the encoder plane order in `cpp/src/games/chess/encoding/ChessEncoding.cpp` is the first
thing to correct, then rebuild and rerun.

### 8. Phase A — matches

Run the frozen evaluation protocol at 64 and 1,000 searches against the Stockfish 13 ladder, with the
unchanged search configuration. Then repeat at a few cPUCT values, because the default was tuned for
a different policy sharpness.

Also measure the cost of the placement window: the tree keeps one `Board` per node and `makeMove` now
allocates a 7-entry snapshot. Compare searches per second against the archived baseline so a slowdown
is not misread as Lc0 being slow.

**Gate: does the teacher play far above 2360 at 64 searches?** If not, stop and report; Phase B tests
nothing at that point.

### 9. Phase B — label positions

    python tools/distill_build_dataset.py \
        --lc0-teacher artifacts/lc0-teacher.pt \
        --output artifacts/lc0-dataset-pilot.bin \
        --positions 200000 --parallel-games 512 \
        --random-opening-plies 8 \
        --sampling-temperature 1.3 --final-temperature 0.1 --greedy-after-ply 80 \
        --random-seed 20260926

Search-free play, one evaluation per position, storing the teacher's plain policy and WDL. No
auxiliary targets are captured, no terminal outcome and no discounting. Measure throughput and
storage on this pilot before scaling to millions; the pilot is a pipeline check, not the dataset.

The builder is one process, so on its own it uses about one of the ~13 granted cores. For the full
dataset run several builders with distinct `--random-seed` values against the one GPU and merge:

    python tools/distill_merge_datasets.py --input part-0.bin --input part-1.bin ... --output lc0-dataset.bin

Size the builder count from `EFFECTIVE_CPUS` minus one for the GPU feeder, and measure where
throughput stops rising rather than assuming it scales; the merged dataset's held-out tail comes
from the last `--input`, so pass a builder whose seed is not reused elsewhere last.

Because we play the games ourselves, every position carries real move history, so the teacher is
never asked to infer a position's past.

### 10. Phase C — distil and match

    python tools/distill_train_student.py ...    # student initialised from checkpoint 1026
    python tools/distill_match.py ...

Two arms worth separating: policy-only distillation first, then policy and value. Checkpoint 1026's
value head was trained on discounted terminal outcomes, and Lc0's WDL is a different quantity, so
training both at once can degrade search behaviour while policy improves.

## Decision rules

- **Teacher plays far above 2360 in our search, student matches it on game-reached positions and
  breaks the plateau** — the network, training path and search can support higher strength; the
  self-play loop is the thing to investigate.
- **Teacher works, student cannot match it even on held-out positions** — representation, capacity,
  optimization or objective. Check the known floor first: this project encodes 8 recent move
  from/to squares rather than 8 board stacks, so there are positions where the teacher's output is
  not a function of the student's input. Measure that floor by querying the teacher on positions
  reachable by several histories; a student at that floor has succeeded.
- **Student fits held-out positions but not student-generated ones** — coverage and distribution
  shift. Label a limited set of those positions before scaling.
- **Student matches the teacher but does not approach its strength in the same search** — audit
  deployed inference and search integration, with concrete divergent positions.
- **Teacher itself does not improve play in our search** — this teacher/search combination cannot
  test the ceiling. Do not read student failure as an architectural limit.

## Known gaps

- The encoder plane order is unverified until step 7 passes.
- Only the classical 112-plane input format is implemented.
- The placement window's throughput cost is unmeasured.
- Phase C's loss weights are not yet specified; fix them before running, and log policy and value
  losses separately.
