# Credit runtime Stage 6 diagnostic overhead

These compute-node smoke runs measure the cost of the Stage 6 ply- and
material-sliced value diagnostics. Each run used the production four-rank DDP
path on GPUs 0–3, a global batch size of 1,024, 50 optimizer steps,
and the same 51,200-position generated replay fixture. No self-play or training
run was started.

| Variant | Commit | Optimizer samples/s | Optimizer time | Peak host RSS |
| --- | --- | ---: | ---: | ---: |
| Pre-Stage-6 baseline | `1c8a83b` | 10,140.71 | 5.049 s | 10,371.4 MiB |
| Every-batch slices | `5f0f5d8` | 8,386.09 | 6.105 s | 10,415.0 MiB |
| One-in-ten-batch slices | `23b8561` | 9,651.90 | 5.305 s | 10,414.3 MiB |

Sampling the diagnostic slices once every ten batches recovered most of the
regression: the final smoke is 4.8% below the pre-Stage-6 optimizer throughput,
compared with 17.3% below it when every batch was sliced. Total and
termination-reason value metrics still include every training position. The
ply and material splits deterministically include 5,120 global positions per
50-step quantum.

The JSON files in this directory are the unmodified benchmark outputs. These
short synthetic-fixture runs are implementation smokes, not stable production
throughput estimates; the Stage 7 mixed self-play canary remains authoritative
for launch readiness.
