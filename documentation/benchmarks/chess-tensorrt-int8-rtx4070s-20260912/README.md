# Chess TensorRT FP16/INT8 feasibility — RTX 4070 SUPER, 2026-09-12

Status: harness ready; measurement deferred until the live v34 run stops. This file is not benchmark evidence yet.

| | |
| --- | --- |
| `experiment_configuration_sha256` | Recorded by the harness from the resolved terminal-run configuration |
| Source revision | Recorded by the harness; benchmark commit listed after validation |
| Node | Vast node at `38.49.42.120:53893`, 8× NVIDIA GeForce RTX 4070 SUPER (SM 8.9), driver 595.71.05 |
| Date | 2026-09-12 preparation; measurement date recorded when run |

## Method

[`benchmark_tensorrt_inference.py`](../../../py/tools/benchmark_tensorrt_inference.py) accepts a selected checkpoint
manifest, an immutable benchmark dataset, and either an immutable calibration dataset or replay store. It verifies the checkpoint and immutable dataset hashes,
then exports the shipped trimmed policy/WDL TorchScript model to a fixed `[320, 52, 8, 8]` FP16 ONNX graph. It
builds TensorRT 10.14 engines for FP16 and entropy-calibrated INT8 with FP16 fallback.

All three measured arms start with the same decoded `int8` tensor for the first 320 legal chess positions in the
benchmark dataset. Their captured CUDA graphs include the device-side input cast, model execution, and conversion
of policy/WDL outputs to FP32 staging tensors:

- production reference: frozen TorchScript, BF16, channels-last, cuDNN benchmarking enabled;
- TensorRT FP16;
- TensorRT INT8 calibrated from 32,000 deterministically sampled v34 replay positions (100 full batches).

The fidelity workload is the first 320 positions of `chess-stockfish-evaluation-v33.bin`, matching the dataset in
the resolved v34 experiment configuration. That immutable file contains 516 positions and has SHA-256
`147b525c430d2a677a6e1309ce60bf7bb22a4886726c5affae63a9f3349d96c8`; its manifest has SHA-256
`d5f902b4283e223f5005860868e57be1971bb7216d9c265b408a3614c3f0a6a5`. The manifest's 332-byte packed
state payload matches the current chess contract (40 binary planes × 8 bytes plus 12 scalar bytes), and its
representation digest is `aa77cda28749d276e28fb3081cd9db924224bd002d0207401c78a72507c6ab4d`.

The 516-position evaluation set is too small for representative INT8 calibration, so it is never used as the
calibration source. The launch samples 32,000 distinct logical rows from the terminal v34 replay with seed
`20260912`, without replacement. The report records the stopped replay's available row count, file size, header and
layout hashes, excluded fidelity-overlap count, selected logical-index hash, packed-state hash, and decoded-input hash.
The sampler discards replay rows whose decoded representation matches any fidelity input and deterministically
continues through its seeded no-replacement order until exactly 32,000 rows remain. Calibration refuses partial
batches and never wraps rows.

The default measurement performs 50 warm-up calls and 15 synchronized repetitions of 100 full batches. The report
retains every repetition, median and p95 batch latency, and median positions/s. Fidelity uses legal-action-masked
policy top-1 agreement and `KL(reference || candidate)`, plus WDL probability and expected-value (`P(win)-P(loss)`)
errors. The command exits nonzero after writing its JSON report if either TensorRT arm violates a configured limit.

The timing scope is the captured GPU execution used by the production inference core. It excludes native search,
CPU encoding, host-to-device input transfer, device-to-host output transfer, and result processing. A favorable
result therefore establishes backend feasibility; it is not a production throughput claim until a native backend
and end-to-end search benchmark exist.

FP16 is selected for the floating TensorRT arm. RTX 4070 SUPER (Ada, SM 8.9) has accelerated FP16 and INT8, and the
pinned TensorRT exposes FP16, BF16, and INT8 builder flags. FP16 gives the ONNX graph and the non-quantized fallback
precision for the calibrated INT8 engine one representation. BF16 remains unmeasured by this probe.

## Dependencies prepared on the node

The idle control virtual environment `/workspace/alphazero-engine-venv` now contains:

```text
onnx==1.22.0
tensorrt-cu12==10.14.1.48.post1
tensorrt-cu12-bindings==10.14.1.48.post1
tensorrt-cu12-libs==10.14.1.48.post1
```

They were installed without importing CUDA, exporting ONNX, calibrating, building an engine, or running inference.
The exact optional pins are in [`requirements-tensorrt-benchmark.txt`](../../../py/requirements-tensorrt-benchmark.txt);
they remain outside the production dependency lock so preparing this probe does not change the approved run hash.

## Post-stop launch

After the production run has stopped and its terminal checkpoint has been preserved, set only the terminal
generation selected for this benchmark. Run from the exact benchmark source revision on the node:

```bash
cd /workspace/alphazero-engine/py
terminal_generation=TERMINAL_GENERATION_SELECTED_AFTER_STOP
/workspace/alphazero-engine-venv/bin/python -m tools.benchmark_tensorrt_inference \
  --configuration /workspace/run-control/configs/vast-chess-8gpu-integrated-v34-resume-g1702.yaml \
  --checkpoint-manifest "/workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34/checkpoint_${terminal_generation}.json" \
  --checkpoint-generation "${terminal_generation}" \
  --benchmark-dataset /workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v33.bin \
  --calibration-replay /workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34/replay.bin \
  --calibration-position-count 32000 \
  --calibration-random-seed 20260912 \
  --artifact-directory /workspace/tensorrt-v34-terminal \
  --output /workspace/tensorrt-v34-terminal/report.json \
  --gpu-id 0 \
  --warmup-iterations 50 \
  --repetitions 15 \
  --iterations-per-repetition 100 \
  --minimum-policy-top1-agreement 0.98 \
  --maximum-mean-policy-kl-divergence 0.005 \
  --maximum-wdl-mean-absolute-error 0.01 \
  --maximum-expected-value-mean-absolute-error 0.015 \
  --acknowledge-gpu-load
```

TensorRT engine construction profiles GPU tactics and calibration runs the model. Confirm all training/evaluation
GPU processes have stopped before launching. ONNX export or parser failures identify the failing checkpoint/path;
TensorRT parser diagnostics are included in the exception. No Connect Four speedup is assumed or transferred to
this model.

## Results

Pending the terminal checkpoint and an idle GPU. Copy `report.json`, the ONNX model, both engines, and the calibration
cache off the ephemeral node. Add their hashes and raw timing/fidelity table here before treating this directory as
evidence.
