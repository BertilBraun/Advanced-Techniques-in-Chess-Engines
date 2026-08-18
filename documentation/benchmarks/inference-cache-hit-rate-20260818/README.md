# Neural-inference repetition upper bound

## Decision

A bounded cross-batch inference cache deserves a separate implementation experiment. At the production-like high budget, 1,334,960 of 3,547,095 encoded inputs repeated (37.6353%). Only 480 positions repeated within their own batch (0.0135% of all positions), so the opportunity is overwhelmingly reuse across batches rather than same-batch coalescing.

This result does not justify shipping the unbounded tracker or assuming a 37.6% speedup. The measurement is an upper bound without capacity, eviction, lookup-transfer, or reuse-distance costs. A follow-up should measure capacity-limited hit rate and end-to-end throughput before production use.

## Implementation

The measurement-only instrumentation runs at `InferencePipeline::submit`, after each position has been encoded as the exact contiguous `int8` network input and before normal model execution. Every row is observed sequentially and every position is still evaluated. Search nodes, batching, selection, visits, outputs, and statistics reuse are unchanged.

The tracker uses a 128-bit hash over the encoded bytes plus all inference dimensions. All workers for one `SearchExecutor`/model generation share the same unbounded set. A committed model refresh clears the tracker, preventing cross-revision hits. Telemetry reports total positions, unique hashes, repeated hashes, repeat rate, set size, and same-batch versus prior-batch repeats through the native and Python APIs.

This is a hash-only upper-bound estimate. A 128-bit collision is negligibly likely but is not detected. An actual cache should either retain enough key material to verify equality or explicitly accept and document its collision model.

## Compute environment

- Revision: `abf5690ff92efa57c3e50d4a3dd0863460c042c0`, created from master `897983301495efda8709644d3dffc364e12e01a6` with the documented experiment-worktree workflow.
- Node: two NVIDIA GeForce RTX 4070 SUPER 12-GiB GPUs; driver 595.71.05; driver API maximum CUDA 13.2; toolkit CUDA 12.8.
- Runtime: Python 3.12.3, PyTorch 2.12.1+cu126, CUDA 12.6, cuDNN 9.10.2.
- Assigned device: GPU 1. `nvidia-smi` was recorded immediately before every arm and showed 1 MiB, 0% utilization, and no GPU processes. GPU 0 was not used.
- Model: current compatible 29-plane TorchScript model, SHA-256 `52d735c1e4f6d4c1a651e40240c5453afef8c46b4979fe495c98e53c00a13554`. Its first convolution shape was verified as `[112, 29, 3, 3]`.
- Positions: 50 unique Stockfish 8moves-v3 chess openings, SHA-256 `ec0fe73449d7ec6ff58db6b060d4c2535afef1d2cfdac098fb44b0cff22735b2`.
- Search: normal tree-search path, root noise enabled, inference batch limit 256, no graph search.

The model file happened to remain on the ephemeral node under an older benchmark-results directory. Only that model artifact was used; no MCGS code, branch, search behavior, or evidence was used as the base.

## Sustained results

Each arm ran for approximately 120 seconds after two warm-up steps.

| Arm | Search budget / parallel searches | Inference positions | Unique during measurement | Potentially avoidable | Repeat rate | Same-batch | Prior-batch | Searches/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Low/moderate | 64 / 4 | 3,176,368 | 46,329 | 3,130,039 | 98.5414% | 0 | 3,130,039 | 26,482.7 |
| High | 600 / 64 | 3,547,095 | 2,212,135 | 1,334,960 | 37.6353% | 480 | 1,334,480 | 29,834.8 |

The low arm repeatedly cycled a finite 50-opening suite: it completed 1,027 measurement steps and encountered 3,202 terminal roots. Its 98.5% result is therefore chiefly a suite-replay upper bound and should not be generalized to a diverse online stream. The high arm progressed much farther per root, completed 124 steps, and encountered 93 terminal roots; it is the more decision-relevant result.

The set grew from 6,330 to 52,659 entries in the low arm and from 58,469 to 2,270,604 entries in the high arm (warm-up entries are excluded from the reported unique-during-measurement value). Peak process RSS was 1,040.7 MiB and 1,239.7 MiB respectively. The unbounded growth is deliberate for this experiment and is unsuitable for production.

## Instrumentation overhead

An A/B throughput check used the accepted production benchmark configuration with 256 parallel games, 64 parallel searches, one inference worker, batch size 256, and GPU 1. The 60-second pair was run in feature-then-control order to counter the 30-second control-then-feature order.

| Duration | Master control searches/s | Instrumented searches/s | Observed penalty |
| --- | ---: | ---: | ---: |
| 30 seconds | 28,902.6 | 27,797.6 | 3.823% |
| 60 seconds | 28,964.6 | 27,930.1 | 3.572% |
| Time-weighted aggregate | 28,943.5 | 27,885.8 | 3.654% |

This is a clean revision-level control, but only two short pairs on one node. It measures the always-on hash/set instrumentation, not the cost or benefit of a real cache. The feature arm's peak RSS was about 62 MiB above control in the 60-second pair, consistent with an unbounded growing set, though peak RSS is noisy.

## Validation

Compilation and tests were run only on the compute node.

- Release CUDA build for compute capability 8.9 completed at the exact committed revision.
- `CUDA_VISIBLE_DEVICES=1 ctest --test-dir /workspace/codex-inference-cache-abf5690f/cpp/test-build --output-on-failure`: 1/1 `NativeTests` passed in 1.56 seconds.
- `PYTHONPATH=/workspace/alphazero-engine-venv/lib/python3.12/site-packages:/workspace/codex-inference-cache-abf5690f/py /usr/bin/python3 -m pytest --import-mode=importlib ./test -q`: 394 passed, 4 skipped, 5 failed in 21.19 seconds. All five failures are unrelated experiment-queue process tests whose temporary `repository_directory` fixtures are not Git repositories; each fails at `git rev-parse --show-toplevel`. The inference and search tests, including unchanged outputs/results, passed.
- Local non-compiling checks only: `uv run ruff format` and `uv run ruff check --fix` passed for the changed Python files.

Native deterministic coverage includes first observation, a prior-batch repeat, duplicate rows within one batch, tracker reset on committed model refresh, unchanged inference outputs, and unchanged batched-search results.

## Evidence

- `low-64.json` and `high-600.json`: raw sustained telemetry.
- `*-environment.txt`: GPU ownership, process snapshots, revisions, and artifact hashes captured immediately before runs. Ephemeral node access tokens were redacted before commit.
- `control-master-*.json` and `instrumented-feature-*.json`: raw throughput A/B output.
- `validation-native.txt` and `validation-python.txt`: full node validation logs.
- `SHA256SUMS`: checksums of the committed evidence after redaction.

## Limitations

- The unbounded set answers only whether an ideal cache could avoid an evaluation. It does not model finite capacity, eviction, reuse distance, tensor retention, device transfers, synchronization, or lookup overhead.
- Identical encoded input is deliberately narrower than search-node identity. History-distinct nodes remain separate and no rule state, node state, or search statistics are reused.
- Repetition in a fixed 50-opening self-play suite can be much higher than repetition in a broad production distribution.
- Root noise changes move sampling but does not guarantee divergent trajectories, especially at the low budget.
- The 128-bit tracker does not verify collisions byte-for-byte.
- The observed instrumentation penalty and memory growth make this tracker an experiment artifact, not a production cache design.
