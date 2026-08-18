# Adaptive Chess search validation on RTX 4070 SUPER

This is implementation evidence for the adaptive-search design, not threshold calibration for the second four-day
run. The continuous-root audit and the native adaptive canary were regenerated at revision
`a3d73359a0c68f14ceae6f147e32b6d7cb9bd06e` with a seed-zero, randomly initialized copy of the first production model
stage. A representative trained model with the three-output search-correction ABI was not available on the node. The
production default is therefore a provisional `0.40`, selected from the target semantics and the asymmetric risk of
premature stopping rather than from this random-model audit.

## Environment

- Two NVIDIA GeForce RTX 4070 SUPER GPUs with 12,282 MiB each; GPU 0 was used.
- NVIDIA driver 595.71.05, whose maximum reported CUDA API is 13.2.
- Locked Python 3.12.3 environment with PyTorch 2.12.1+cu126, CUDA 12.6, and cuDNN 9.10.2.
- Release C++ build used CUDA toolkit 12.8.93.
- KataGo 1.17.1 reported its CUDA backend, and the provisioning 7x7 and 9x9 analysis smokes passed.

## Protocol and result

`tools.calibrate_adaptive_search` searched 16 fixed opening positions continuously from 400 through 3,200 root
visits, retaining one root per position and recording policy and root-value snapshots every 100 visits. Calibration
requests explicitly enabled detailed policy snapshots; ordinary self-play keeps scalar checkpoint telemetry only. The
tool then replayed every configured maximum and correction threshold offline and ran a fresh native generation-450
adaptive search canary.

The 51,200-reference-visit audit completed in 3.772 seconds at 13,573 simulations per second with an average inference
batch of 63.94. The native adaptive canary averaged 718.75 visits, 9,441 simulations per second, and a 27.03-position
inference batch; all 16 positions stopped through the deterministic rule. These numbers demonstrate continuous-root
checkpoint collection, offline rule replay, tail termination, and useful batching. They do not measure playing
strength or justify any production threshold because the network was random.

Schema version 2 compares the raw most-visited action used for move selection, while retaining the post-pruning policy
leader for stopping metrics. It also records the final policy correction, value correction, combined search-correction
target, prediction error by stopping-visit bucket, and unstable-position recall after the learned gate. For example,
the 1,000-visit/0.50 candidate used 606.25 mean visits and had 56.25% raw move agreement with the 3,200-visit reference;
its constant random-model prediction is mechanics evidence only, not a threshold recommendation.
Offline replay now enforces the same 400-visit minimum as native search. The minimum stopping bucket across every
candidate in this report is 400 visits; no candidate stopped below the production minimum.

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
- Focused adaptive-search, configuration, self-play-worker, replay, telemetry, and game-contract tests passed: 101 tests.
- The full Python suite excluding the separately evidenced queue fixture passed 389 tests and skipped 4. The unfiltered
  suite passed 391 tests, skipped 4, and had five Linux queue-process failures because those tests pass a
  non-Git `tmp_path` to repository validation. Both the failing test file and validation implementation are unchanged
  from base revision `30c4fbf8`; the failures are unrelated to this feature.
- The Release native suite passed five consecutive runs. The naive Python MCTS benchmark completed an eight-simulation
  GPU smoke through the three-output inference ABI.
- A focused native test independently reconstructs policy correction from the clean pre-noise legal prior and the
  post-pruning searched policy under enabled Dirichlet noise, and reconstructs value correction from root Q and the
  network WDL scalar. A focused offline test proves pre-minimum checkpoints cannot stop candidate replay.
- `ruff format` was unchanged. `ruff check --fix` left no warnings introduced by this branch; the checked canonical
  stub name and an older completed-game positional-only hook retain their pre-existing warnings.

Production threshold calibration still requires this same audit on sampled positions from a representative trained
checkpoint. At minimum, compare thresholds around 0.40 using move agreement, policy total variation, root-value
error, mean visits, maximum-cap frequency, and a small equal-average-compute paired match before approving the run.
