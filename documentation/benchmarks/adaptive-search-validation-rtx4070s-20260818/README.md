# Adaptive Chess search validation on RTX 4070 SUPER

This is implementation evidence for the adaptive-search design, not threshold calibration for the second four-day
run. The continuous-root audit and the native adaptive canary used revision
`ea0edc0b9454f9d1f111f185968bd35f15e4a643` and a seed-zero, randomly initialized copy of the first production model
stage. A representative trained model with the three-output search-correction ABI was not available on the node, so
the configured `0.50` learned-gate threshold remains provisional.

## Environment

- Two NVIDIA GeForce RTX 4070 SUPER GPUs with 12,282 MiB each; GPU 0 was used.
- NVIDIA driver 595.71.05, whose maximum reported CUDA API is 13.2.
- Locked Python 3.12.3 environment with PyTorch 2.12.1+cu126, CUDA 12.6, and cuDNN 9.10.2.
- Release C++ build used CUDA toolkit 12.8.93.
- KataGo 1.17.1 reported its CUDA backend, and the provisioning 7x7 and 9x9 analysis smokes passed.

## Protocol and result

`tools.calibrate_adaptive_search` searched 16 fixed opening positions continuously from 400 through 3,200 root
visits, retaining one root per position and recording policy and root-value snapshots every 100 visits. It then
replayed every configured maximum and correction threshold offline and ran a fresh native generation-450 adaptive
search canary.

The 51,200-reference-visit audit completed in 3.935 seconds at 13,011 simulations per second with an average inference
batch of 63.94. The native adaptive canary averaged 768.75 visits, 10,814 simulations per second, and a 20.49-position
inference batch; all 16 positions stopped through the deterministic rule. These numbers demonstrate continuous-root
checkpoint collection, offline rule replay, tail termination, and useful batching. They do not measure playing
strength or justify any production threshold because the network was random.

The complete per-position snapshots and candidate metrics are in
[`random-generation450-16.json`](random-generation450-16.json).

## Checkpoint search

A read-only search covered `/workspace/alphazero-engine` and all other model artifacts under `/workspace`. The only
plausible trained artifact was
`/workspace/alphazero-engine/documentation/benchmarks/chess-results/best_model.jit.pt`; its metadata identifies an old
12-hour run and its TorchScript forward method returns two tensors. The discovered
`current-r3-seed-20260818.jit.pt` and both architecture benchmark models are also benchmark artifacts with the old
two-output ABI. None contains the learned search-correction output required for representative gate calibration.

## Validation

- Release C++ extension and `NativeTests` built successfully.
- Native tests passed after the checkpoint-policy binding was added; one initial run exposed an existing
  Dirichlet-noise-sensitive exact-visit assertion, which was made deterministic by disabling noise in that test.
- Targeted calibration, self-play-worker, and game-contract tests passed: 33 tests.
- The full Python suite passed 383 tests and skipped 4. Five Linux queue-process tests failed because they pass a
  non-Git `tmp_path` to repository validation. Both the failing test file and validation implementation are unchanged
  from base revision `30c4fbf8`; the failures are unrelated to this feature.
- `ruff format` and `ruff check --fix` passed for the new calibration files. The earlier complete feature-file pass
  left no warnings introduced by this branch.

Production threshold calibration still requires this same audit on sampled positions from a representative trained
checkpoint. At minimum, compare thresholds around 0.50 using move agreement, policy total variation, root-value
error, mean visits, maximum-cap frequency, and a small equal-average-compute paired match before approving the run.
