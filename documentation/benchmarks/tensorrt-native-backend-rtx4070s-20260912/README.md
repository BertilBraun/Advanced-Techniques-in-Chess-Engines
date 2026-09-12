# Native TensorRT inference on RTX 4070 SUPER

This benchmark validates the production C++ TensorRT backend on one RTX 4070 SUPER. The branch is
`codex/int8-native`; the final implementation commit recorded here is `15095393` or later.

## v34 FP16 result

The paired native search benchmark used generation 1785, 400 active roots, 64 visits per root,
`parallel_searches: 4`, one inference worker, and a fixed batch cap of 320. Both arms completed ten
measured search batches after two warm-up batches.

| Backend | Average actual batch | Simulations/s | Relative |
| --- | ---: | ---: | ---: |
| TorchScript BF16 | 314.58 | 40,715.7 | 1.000x |
| TensorRT FP16 | 313.93 | 75,888.7 | **1.864x** |

Five legal-position policy probes preserved every top action. The largest legal-action probability
difference was 0.00601 and the largest WDL-component difference was 0.00171. These differences
compare TorchScript BF16 with TensorRT FP16 rather than identical arithmetic.

The refittable FP16 template accepts every updated ONNX weight with no missing weights. Publishing
generation 1784 and 1785 took about 19-22 seconds per generation, including loading the published
TorchScript checkpoint, fixed-batch ONNX export, refit, engine serialization, hashing, and atomic
publication. A real two-worker self-play refresh from generation 1784 to 1785 took 0.220 seconds;
search then continued at 73,449 simulations/s. The generated engine is cached by the checkpoint and
template SHA-256 values, and concurrent publishers serialize through a file lock.

## Early-fold QAT INT8 probe

The available INT8 candidate is an early-fold 14x160 QAT experiment trained for only 6,000 total
steps. It is a deployment experiment, not a trained replacement for v34.

| Backend for the same QAT weights | Simulations/s | Relative to candidate FP16 |
| --- | ---: | ---: |
| TensorRT FP16 | 75,338.5 | 1.000x |
| TensorRT INT8 | 98,396.7 | **1.306x** |

The native FP16-engine versus INT8-engine probe preserved all five top actions. The largest legal
prior difference was 0.05753 and the largest WDL-component difference was 0.00293. A paired
64-search match used five openings with reversed colours. INT8 scored 4 wins, 2 draws, and 4 losses:
50.0% [25.0%, 70.0%], or 0 Elo [-191, +147]. Ten games only reject a catastrophic deployment
failure; the interval cannot establish equivalence.

The INT8 candidate reached 98,397 simulations/s, 2.42x the v34 TorchScript baseline measured above.
That ratio combines TensorRT with a quantization-oriented architecture and therefore does not
predict a 2.42x production-training gain. The controlled incremental result is 1.31x over the same
candidate's TensorRT FP16 engine. A production INT8 run still requires a well-trained QAT model and
a higher-resolution strength gate.

Raw machine-readable results are in [`raw/`](raw/).
