# Chess TensorRT FP16/INT8 feasibility — RTX 4070 SUPER, 2026-09-12

Status: complete. TensorRT FP16 passes the fidelity limits and reaches 100,482 positions/s. Calibrated INT8 PTQ is
not viable for this checkpoint under the unchanged fidelity limits, even when explicit Q/DQ is restricted to one
trunk convolution.

| | |
| --- | --- |
| `experiment_configuration_sha256` | `d485553069fa4660ce2dbd97c747da64e4ca67f6921889d0ec6c1664ea320c73` |
| Source revision | `0d3d2c864c04d76a5d6fc8340419ff93d3f14b39` for the final one-convolution probe |
| Node | Vast node at `38.49.42.120:53893`, 8× NVIDIA GeForce RTX 4070 SUPER (SM 8.9), driver 595.71.05 |
| Date | 2026-09-12 |

## Method

[`benchmark_tensorrt_inference.py`](../../../py/tools/benchmark_tensorrt_inference.py) accepts a selected checkpoint
manifest, an immutable benchmark dataset, and either an immutable calibration dataset or replay store. It verifies
the checkpoint and immutable dataset hashes, then exports the shipped trimmed policy/WDL TorchScript model as
separate fixed `[320, 52, 8, 8]` FP16 and FP32 ONNX graphs. NVIDIA ModelOpt 0.46.1 performs max calibration on the
FP32 graph and inserts explicit Q/DQ nodes around the first convolution in the final residual block. It excludes the policy and WDL heads,
including their reductions, indexing, scatter, and outputs. TensorRT 10.14 builds the explicit-Q/DQ graph with FP16
as the high-precision fallback.

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

## Production-path sanity check

The isolated TorchScript result is higher than end-to-end self-play throughput, but its full batch is representative.
The final archived v34 self-play process averaged 319.39 positions per inference call against the configured cap of
320. A separate 126.47-second native control used generation 1785, 512 concurrent games, 800 visits, native
parallelism 4, one inference worker, two outstanding slots, BF16, channels-last, and cuDNN benchmarking. Only the
opening source was changed to random openings so the stopped run's restart-state database was not mutated. It
processed 6,129,023 model positions in 19,170 calls, averaged 319.72 positions/call, and achieved 48,463 model
positions/s on one GPU. Of those calls, 19,124 used the full batch of 320.

The terminal 14x160/800-visit training phase recorded about 506 accepted self-play positions/s across all eight
GPUs. Multiplying by 800 visits, applying the native control's 0.99756 model-position/search ratio, and dividing by
eight estimates 50,477 model positions/s/GPU during training. That estimate is 4.2% above the direct native control,
which is reasonable given the phase-level rate is rounded and covers a longer workload.

| TorchScript measurement | Model positions/s/GPU | Relative to isolated core |
| --- | ---: | ---: |
| Isolated captured batch-320 core | 61,694 | 1.000 |
| Native production self-play control | 48,463 | 0.786 |
| Training-phase estimate | 50,477 | 0.818 |

The native control and training estimate agree; the isolated core is about 22% faster because it deliberately omits
search scheduling, encoding, result processing, transfers, and gaps between model calls. The TensorRT FP16 1.63x
ratio is consequently valid as an inference-core comparison under identical timing scope, but it is not a measured
self-play speedup. If the native non-model time remained fixed and non-overlapped, the two measured latencies imply
roughly a 1.44x self-play ceiling. A production TensorRT backend must be integrated and measured before quoting an
end-to-end gain.

The compact native control output is preserved as
[`production-self-play-reference.json`](results/production-self-play-reference.json). Its configuration hash differs
from the resolved run only because of the documented start-position override.

FP16 is selected for the floating TensorRT arm. RTX 4070 SUPER (Ada, SM 8.9) has accelerated FP16 and INT8, and the
pinned TensorRT exposes FP16, BF16, and INT8 builder flags. FP16 supplies the non-quantized fallback precision for
the calibrated INT8 engine. BF16 remains unmeasured by this probe.

## Dependencies prepared on the node

The idle control virtual environment `/workspace/alphazero-engine-venv` now contains:

```text
onnx==1.21.0
nvidia-modelopt[onnx]==0.46.1
tensorrt-cu12==10.14.1.48.post1
tensorrt-cu12-bindings==10.14.1.48.post1
tensorrt-cu12-libs==10.14.1.48.post1
```

The exact optional pins are in [`requirements-tensorrt-benchmark.txt`](../../../py/requirements-tensorrt-benchmark.txt);
they remain outside the production dependency lock so preparing this probe does not change the approved run hash.

## Reproduction command

The production run had stopped and GPU processes were absent before this command was launched. The selected terminal
checkpoint is generation 1785. Run from source revision `0d3d2c864c04d76a5d6fc8340419ff93d3f14b39`:

```bash
cd /workspace/alphazero-engine/py
/workspace/alphazero-engine-venv/bin/python -m tools.benchmark_tensorrt_inference \
  --configuration /workspace/run-control/configs/vast-chess-8gpu-integrated-v34-resume-g1702.yaml \
  --checkpoint-manifest /workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34/checkpoint_1785.json \
  --checkpoint-generation 1785 \
  --benchmark-dataset /workspace/evaluation-artifacts/chess/chess-stockfish-evaluation-v33.bin \
  --calibration-replay /workspace/alphazero-engine-v34-lr-001/py/training_data/production/vast-chess-8gpu-integrated-v34/replay.bin \
  --calibration-position-count 32000 \
  --calibration-random-seed 20260912 \
  --artifact-directory /workspace/tensorrt-v34-terminal-explicit-qdq-one-conv-v2 \
  --output /workspace/tensorrt-v34-terminal-explicit-qdq-one-conv-v2/report.json \
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

The command is expected to write the complete report and then exit nonzero because the INT8 candidate fails all four
fidelity gates. TensorRT engine construction profiles GPU tactics and calibration runs the model, so all other GPU
work must remain stopped while reproducing it. No Connect Four speedup is assumed or transferred to this model.

## Results

The primary comparison uses the final one-convolution report so all rows share one export, build, and timing run.
Positions/s is the median of 15 repetitions of 100 synchronized full batches after 50 warm-up batches.

| Backend | Median ms/batch | Median positions/s | vs. Torch | Policy top-1 | Mean policy KL | WDL MAE | EV MAE | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TorchScript BF16/channels-last | 5.1869 | 61,694.0 | 1.00x | reference | reference | reference | reference | — |
| TensorRT FP16 | 3.1847 | 100,481.7 | 1.63x | 0.99375 | 0.000192 | 0.000961 | 0.002472 | pass |
| TensorRT explicit INT8, one trunk Conv | 3.1605 | 101,250.4 | 1.64x | 0.26250 | 1.277747 | 0.216095 | 0.535823 | fail |

The FP16 engine passes every limit. The one-convolution INT8 engine is only 0.8% faster than FP16 and fails every
fidelity limit by a wide margin. The bounded selective-Q/DQ variants establish that broader PTQ does not recover a
usable accuracy/speed tradeoff:

| Explicit-Q/DQ scope | Q nodes | Median positions/s | Policy top-1 | Mean policy KL | WDL MAE | EV MAE | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| All 29 trunk convolutions | 58 | 183,050.9 | 0.11250 | 21.528638 | 0.130539 | 0.347158 | fail |
| Both convolutions in final residual block | 4 | 102,768.2 | 0.190625 | 8.654629 | 0.221212 | 0.572782 | fail |
| First convolution in final residual block | 2 | 101,250.4 | 0.26250 | 1.277747 | 0.216095 | 0.535823 | fail |

The first legacy TensorRT calibration attempt supplied FP16 bytes through a calibrator API that interprets the input
as FP32. Its cache recorded an impossible input scale of about `1.737e11`. Supplying an FP32 ONNX input and FP32
calibration/runtime buffers corrected that defect and produced a plausible input maximum of 99 and scale of about
0.997, but implicit INT8 still failed: 180,964.3 positions/s, 0.43125 policy top-1 agreement, 0.951680 mean policy
KL, 0.044995 WDL MAE, and 0.117260 EV MAE. The buffer bug explains the initial catastrophic legacy result; it does
not explain the remaining accuracy loss.

ModelOpt explicit Q/DQ then removed ambiguity about which layers TensorRT quantized. Both heads stayed in high
precision, including policy indexing/scatter and outputs. Quantizing all trunk convolutions was fast but invalid,
and quantizing a single late trunk convolution remained severely inaccurate while eliminating nearly all INT8 speed
benefit. Calibrated PTQ is therefore not viable for this v34 checkpoint under the strict gates. Further INT8 work
requires quantization-aware training so the network can adapt to activation quantization; the benchmark gates should
remain unchanged.

## Preserved evidence

Every JSON file below is copied byte-for-byte from the node and includes all repetition timings, fidelity metrics,
model/input/configuration hashes, engine hashes, dependency versions, and calibration selection hashes.

| Report | Source revision | Report SHA-256 | Q/DQ ONNX SHA-256 |
| --- | --- | --- | --- |
| [`explicit-one-conv.json`](results/explicit-one-conv.json) | `0d3d2c864c04d76a5d6fc8340419ff93d3f14b39` | `582300ee78d5fc646b2eaf47c396e26bca7cb3101d4d56ca1c0b8ab0dac3d41c` | `0e296f409b560062fb4ad99d21d25db179c66233a60f19adad14f89224d6373e` |
| [`explicit-final-block.json`](results/explicit-final-block.json) | `5c59e32173f3610216756fe2524f1cf80cca320a` | `52b8784a48870dfb09b5c1529becfd6e5ce0239c20a1bc3232835c7dbce05782` | `8409a8218977e1b66227e87d3ceab481613edbf85cc16c051adbabc2714ddf63` |
| [`explicit-all-trunk-conv.json`](results/explicit-all-trunk-conv.json) | `1938e8f900df340ba4d3c38ab6fdbd8f574b8aac` | `8876e9a3d6517ded932a24ba2a3c9823fbf75d848cdb81ce10f55ae235a997bd` | `2397ac9986c257b2b3e965e4d006084bef77b83b71421e6d75f9e92992e456aa` |
| [`legacy-fp32-calibration-buffer.json`](results/legacy-fp32-calibration-buffer.json) | `042a7df2453b10a86c4159a9ab26194de896aa45` | `2e9876c703eb3b7761544b667cdef5f31bc84b1d7610cd4e5b50d571a95c5846` | n/a |
| [`legacy-fp16-calibration-buffer.json`](results/legacy-fp16-calibration-buffer.json) | `fc3b69dfcb21d5b2094af6a09f185db09be9e9f9` | `fc015907f370da03dc660c3d8dd93625cd5616c443aa16642af483c930ee39c7` | n/a |

The selected checkpoint manifest SHA-256 is
`618646b9e7b398a55702305722275fb507078579a590f043014472e4ee6b645f`; the model SHA-256 is
`8899d1d4fedda4dc0faf85b54c2d433dde65d5953be0ae078ea69354ec01d4bf`. The fidelity state SHA-256 is
`90186855d1ab471583e0d4ae53294b4d549e8074e915c93b6b4d034cf03087d3`, and its legal-action mask SHA-256 is
`a47162a0ecdd2736f353cc24b6acc19648c9c9e784433b43f5767e02ddf16e9b`.

The replay contained 10,000,000 positions. Sampling excluded two rows whose decoded representations overlapped
the fidelity batch, then selected exactly 32,000 distinct logical indices without replacement. The index SHA-256 is
`ff12aaba5ac04159d9f400208dcb60c368e9f152006e7868e476ae915595f987`, packed-state SHA-256 is
`19ea95d4182dc3d5fe171d43e29efee6431bd4c77ad369aa34d12acc027ca316`, and decoded-state SHA-256 is
`afb79829e75d2a9f8ba860fe69447d803f197fa1a35c2b700ed906dae6ff051e`.
