# V76 v35 medium pre-fold backend throughput

Date: 2026-09-18

## Result

The v35 scaled-post 14x160 model benefits materially from TensorRT INT8 at the production actor topology.
Using the same generation-9 QAT checkpoint and matched search settings, INT8 completed 1,273,125 searches/s
across eight RTX 4070 SUPER GPUs versus 915,315 searches/s for TensorRT FP16, a 1.391x ratio.

| Backend | Searches/s total | Searches/s/GPU | Model positions/s total | Mean batch | Mean active GPU utilization | Mean active power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorRT FP16 | 915,315 | 114,414 | 646,564 | 316.24 | 98.74% | 196.68 W |
| TensorRT INT8 | 1,273,125 | 159,141 | 910,889 | 317.02 | 97.18% | 194.61 W |
| INT8 relative | **1.391x** | **1.391x** | **1.409x** | 1.002x | 0.984x | 0.990x |

The backend speedup is therefore large enough to matter during the medium stage. This result supersedes the
inference drawn from the short-run replay-credit comparison: those live runs already used four self-play actor
processes per GPU, but training, evaluation, game completion, and replay-credit timing obscured backend throughput.

## Matched workload

- Source revision: `72c92774254c7c4e369bd88319965fff4b3f3257`.
- GPUs: eight RTX 4070 SUPER devices.
- Topology: four self-play actor processes per GPU, one inference thread per actor, 512 games per actor.
- Search: 400 baseline visits, automatically selected parallel-search count 2.
- Inference: batch cap 320, two outstanding batches per inference worker, channels-last layout.
- Measurement: two warm-up batches followed by a nominal 60-second synchronized arm.
- Checkpoint: v69 generation 9, v35 scaled-post 14x160, pre-fold QAT phase, 4,500 optimizer steps.
- FP16 model SHA-256: `fdf19411af3d1806cb2129a2bcb2609d2a51e44020abdabc923607b96bc0792d`.
- INT8 model SHA-256: `73f782de60ed165dd4242c78d58b9507e3cf0713eb0bef759bae98b6ee2cf62c`.
- FP16 configuration SHA-256: `60de19d34f10fca5416c815dd02fa18f4f7e0c7e28d93487e4e075c46dc8c6bb`.
- INT8 configuration SHA-256: `ebe284118a69549fccba4bd7d77b01f8584ea03137a2db558ee49242d0ebd668`.

Both arms used the production `SelfPlayWorker` and native search path. All 32 workers completed without stderr.
The FP16 arm completed 59,596,800 searches in 65.111 seconds; INT8 completed 80,896,000 in 63.541 seconds.

## Interpretation limits

Short benchmark game and replay-position completion counts are right-censored because all 16,384 games begin
from fresh states. Searches/s and model positions/s are the decision metrics. The benchmark establishes throughput;
it does not replace the existing policy/value fidelity diagnostics or the short learning-stability evidence.
