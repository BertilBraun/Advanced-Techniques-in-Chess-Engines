# TensorRT INT8 salvage investigation

This benchmark tests whether TensorRT INT8 can accelerate v34 self-play without changing the model's
policy or value behavior enough to invalidate training targets. It was run on one of the production
RTX 4070 SUPER GPUs after the generation-1785 training checkpoint was archived.

## Identity

| Item | Identity |
| --- | --- |
| Experiment source revision | `f71fb0474b9bf4ea13c296173da6785a84eecbf3` |
| Experiment config SHA-256 | `d485553069fa4660ce2dbd97c747da64e4ca67f6921889d0ec6c1664ea320c73` |
| Checkpoint manifest SHA-256 | `618646b9e7b398a55702305722275fb507078579a590f043014472e4ee6b645f` |
| Generation-1785 inference model SHA-256 | `8899d1d4fedda4dc0faf85b54c2d433dde65d5953be0ae078ea69354ec01d4bf` |
| Replay header SHA-256 | `8be8c58121140900958be306704d2a272044daf8a0135e3c04e0318256eaa50a` |
| Replay layout SHA-256 | `3c5b6f6f2c68ff5c5525c308b41e3c0228537b75d1faf361e4a27156276849eb` |
| Held-out dataset SHA-256 | `147b525c430d2a677a6e1309ce60bf7bb22a4886726c5affae63a9f3349d96c8` |
| TensorRT | 10.14.1.48.post1 |
| ModelOpt | 0.46.1 |
| PyTorch | 2.12.1+cu126 |
| Batch size | 320 |

The acceptance gates were 98% policy top-1 agreement, mean legal-policy KL at most 0.005, WDL mean
absolute error at most 0.01, and expected-value mean absolute error at most 0.015. Selection and
holdout positions were disjoint. TensorRT throughput is isolated model-core throughput; the observed
production control was 48,463 evaluated positions/s/GPU and includes search and transport overhead.

## Result

No tested INT8 route provided a fidelity-valid material speedup.

| Candidate | Policy top-1 | Policy KL | WDL MAE | EV MAE | Core positions/s | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TensorRT FP16 | 100.0% | 0.000157 | 0.00101 | 0.00250 | 102,092 | Valid reference |
| Full trunk, max PTQ best seed | 13.3% | 22.76 | 0.125 | 0.330 | 183,051 | Invalid |
| Full trunk, entropy PTQ holdout | 25.9% | 2.589 | 0.109 | 0.268 | 182,320 | Invalid |
| Early contiguous blocks 0-3, max PTQ holdout | 95.9% | 0.00264 | 0.00354 | 0.00899 | 114,552 | Near-valid, fails policy gate |
| Early blocks QAT exports | 94.7%-96.9% | 0.00177-0.00179 | 0.00417-0.00460 | 0.0102-0.0113 | 104,779-105,263 | Invalid and slower than PTQ |
| Full-trunk QAT | 35.0% | 1.061 | 0.0878 | 0.209 | 151,516 | Invalid |
| Full-trunk weight-only INT8 | 98.8% | 0.000137 | 0.00099 | 0.00226 | 98,841 | Valid but slower than FP16 |
| Full-trunk SmoothQuant | 65.6% | 0.167 | 0.0503 | 0.113 | 151,532 | Unsupported for the Conv trunk and invalid |
| Full-trunk FP8 | 89.1% | 0.0231 | 0.00831 | 0.0206 | 102,640 | Invalid and no faster than FP16 |

The strongest near-valid INT8 candidate quantizes eight early trunk convolutions. Its 12.2% isolated
core gain projects to only about a 9% end-to-end upper-bound gain using the measured production/core
overhead split, and it still changes the policy's top move on about 4% of unseen positions. A bounded
QAT recovery did not close that gap. The strongest fidelity-valid quantized candidate is weight-only
INT8, which is 3.2% slower than the TensorRT FP16 reference. Neither warrants integration or chess
matches.

## Calibration and sensitivity

Ten max-calibration seeds on a common 196-position selection split produced 10.2%-13.8% policy
top-1 agreement for full-trunk INT8. Entropy calibration improved the best selection result to 29.6%
but remained unusable; its untouched 320-position holdout measured 25.9%. The available ModelOpt
INT8 wrapper maps non-entropy methods to min/max, so percentile calibration is not exposed. Increasing
max-calibration data from 3,200 to 32,000 to 128,000 positions produced 12.2%, 13.3%, and 15.8%
top-1 agreement respectively. The 128,000-position run had a mean KL of 114 because max calibration
became dominated by extreme activation values.

All 29 trunk convolution nodes were tested individually. Blocks 0-4 were the least sensitive, while
quantizing the first convolution in blocks 5-13 caused large policy errors. Cumulative partitions
showed a sharp frontier:

| Partition | Convolutions | Policy top-1 | Policy KL | Core positions/s |
| --- | ---: | ---: | ---: | ---: |
| Ranked safest | 4 | 97.45% | 0.00110 | 104,170 |
| Ranked safest | 8 | 94.90% | 0.00451 | 111,265 |
| Contiguous blocks 0-3 | 8 | 95.92% | 0.00357 | 115,155 |
| Ranked safest | 12 | 79.59% | 0.146 | 118,631 |
| Ranked safest | 20 | 36.2% | 1.356 | 124,866 |
| Ranked safest | 28 | 14.8% | 23.56 | 175,575 |

The detailed TensorRT inspector confirms that the full-trunk engine really uses INT8 tensor-core
convolution tactics with INT8 inputs, weights, and outputs. Global-context islands remain FP32 and
the heads remain FP16. Input and residual activation ranges grow sharply through the network: the
observed activation scale rose from about 0.78 at the input to roughly 40-43 in later residual blocks.
This accumulation, rather than a bad ONNX export, is the principal PTQ failure mode.

## Runtime isolation

TorchScript BF16 and unquantized ONNX FP32 agree on 99.375% of policy top actions with KL 0.000159,
so the export contract is sound. A progressive comparison shows that ONNX Runtime QDQ and TensorRT
are mutually close for a single early convolution, then diverge as QDQ nodes accumulate. At full
trunk, both QDQ runtimes are independently far from the unquantized model. Some runtime rounding
errors cancel when measured against the source, so ONNX Runtime cannot be used as a proxy for
TensorRT fidelity; every candidate was checked in the final TensorRT engine.

ModelOpt's throughput-only autotuner evaluated 470 schemes and selected three Q/DQ pairs for a 1.009x
internal speedup. The selected graph retained only 64.1% policy top-1 agreement and ran at 100,740
positions/s in the common harness. ModelOpt produced that final graph before its
`remove_partial_input_qdq` postprocessing step raised `IndexError: list index out of range`; the
benchmark recovered only that validated `optimized_final.onnx` artifact, and the structured report
records the recovery condition and message. It therefore found neither an accurate nor a faster
partition.

## QAT and value-head ablation

The targeted QAT run used 409,600 replay positions for online BF16-teacher matching, 51,200 positions
for checkpoint selection, and 51,200 untouched positions for its final report. Training only the 1.75M
parameters in the eight quantized convolutions did not improve policy agreement materially. Allowing
the downstream trunk and heads to absorb systematic shifts raised held-out top-1 agreement from 96.33%
to 96.50% after 500 steps and 96.45% in the selected 1,500-step run. KL tail outliers remained and
the TensorRT export did not pass the policy gate.

Full-trunk QAT started at 32.9% held-out policy top-1 agreement in the PyTorch fake-quant model.
Within 100 optimizer steps the policy collapsed to 9.4% and did not recover over 3,000 steps. The
value error improved during training, but the candidate selected on policy remained the initial model;
its TensorRT export measured 35.0% top-1 agreement. Replacing the 2-channel value head with a
32-channel head and training its 103,651 parameters for 3,000 steps reduced held-out expected-value
MAE from 0.444 to 0.140. That remained worse than the 2-channel run's 0.115 at step 3,000, while the
policy stayed unusable. The wider value head therefore did not address the binding trunk failure.

## Interpretation

The approximately 3x isolated INT8 throughput is real, but it applies only after quantizing almost the
entire convolutional trunk, exactly where this trained network is most sensitive. Calibration choice,
ten-seed selection, 40x more calibration data, sensitivity-driven partitions, automatic tuning,
short QAT, SmoothQuant, weight-only INT8, per-channel activation attempts, and FP8 did not produce a
candidate that was both accurate and materially faster.

A future run can revisit INT8 by training with fake quantization from early generations, before large
late-block activation ranges become established, or by redesigning the residual trunk to constrain
activation ranges. Those are new training/architecture experiments rather than a safe conversion of
the v34 model.

The `raw/` directory contains the experiment reports, the exact diagnostic scripts, the progressive
runtime comparisons, and the TensorRT layer inspector output. Large ONNX models and serialized engines
are intentionally excluded; their identities are recorded in the reports. The packaged raw-evidence
archive has SHA-256 `62f0a24dc3809f423c3d67a03a585860694071d246eb2fca2670b4f9672cf1a9`.
