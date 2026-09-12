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
manifest and separate benchmark/calibration dataset paths. It verifies the checkpoint and immutable dataset hashes,
then exports the shipped trimmed policy/WDL TorchScript model to a fixed `[320, 52, 8, 8]` FP16 ONNX graph. It
builds TensorRT 10.14 engines for FP16 and entropy-calibrated INT8 with FP16 fallback.

All three measured arms start with the same decoded `int8` tensor for the first 320 legal chess positions in the
benchmark dataset. Their captured CUDA graphs include the device-side input cast, model execution, and conversion
of policy/WDL outputs to FP32 staging tensors:

- production reference: frozen TorchScript, BF16, channels-last, cuDNN benchmarking enabled;
- TensorRT FP16;
- TensorRT INT8 calibrated from the separately named immutable legal-position dataset.

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
cd /workspace/alphazero-engine
terminal_generation=TERMINAL_GENERATION_SELECTED_AFTER_STOP
/workspace/alphazero-engine-venv/bin/python py/tools/benchmark_tensorrt_inference.py \
  --configuration /workspace/run-control/configs/vast-chess-8gpu-integrated-v34-resume-g1702.yaml \
  --checkpoint-manifest "/workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34/checkpoint_${terminal_generation}.json" \
  --checkpoint-generation "${terminal_generation}" \
  --benchmark-dataset /workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v32.bin \
  --calibration-dataset /workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v32.bin \
  --calibration-position-count 480 \
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
