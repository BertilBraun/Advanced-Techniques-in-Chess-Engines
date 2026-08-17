# Chess architecture contention benchmark

## Scope

This benchmark compares the six frozen CNN and attention definitions on the eight-RTX-3060 production topology
while the read-only `chess-day4-confirmatory-200-parallel-fixed` evaluation shared all GPUs. The evaluation remained
`RUNNING` for every primary run from 2026-08-17 21:10:49 UTC through 21:28:08 UTC. It was never signaled, paused,
restarted, or reconfigured.

Each primary training measurement ran for at least 120 seconds so short evaluation-load fluctuations were averaged
within the sample. CNN and attention candidates were adjacent within each parameter band. Results are comparative
contention measurements, not uncontended capacity claims.

## Protocol

- Source model and harness revision: `e35a9a0bf01519df7e32cade2792695b5cdf7f90`.
- Plan: `py/configs/benchmarks/chess-architecture-contended-120s.yaml`.
- Eight DDP ranks on GPUs 0-7; global batch 2,048 as eight local batches of 256.
- Bfloat16 autocast, AdamW, ten warmup steps, then a 120-second equal-wall-time measurement.
- CPU thread pools were limited to two threads per rank and benchmark processes ran at niceness 5.
- Frozen synthetic replay: 8,192 samples generated with seed `20260817`; SHA-256
  `6ff14bf9f605132c788e325351ea423f9315d96881c0bab8551af4aeb3f897f2`.
- PyTorch 2.12.1+cu126, CUDA 12.6, eight NVIDIA GeForce RTX 3060 12-GiB GPUs.

The synthetic replay fixes tensor shapes, transfers, targets, and sample order for throughput comparison. It does not
represent chess position quality and cannot measure architecture strength or sample efficiency.

## Training results

| Band | CNN samples/s | Attention samples/s | Attention / CNN | Attention slowdown | CNN peak MiB | Attention peak MiB | Memory ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1M | 68,354 | 57,180 | 83.65% | 16.35% | 64.80 | 125.59 | 1.94x |
| 4M | 26,897 | 19,099 | 71.01% | 28.99% | 305.56 | 741.58 | 2.43x |
| 9M | 12,067 | 10,255 | 84.98% | 15.02% | 661.21 | 1,511.73 | 2.29x |

Under this implementation and shared evaluation load, the CNN control trained faster and used materially less peak
allocated memory in every parameter band. The largest relative throughput gap was the 4M pair.

## Inference diagnostics

| Band | CNN batch-64 positions/s | Attention batch-64 positions/s | Attention / CNN |
| --- | ---: | ---: | ---: |
| 1M | 541,660 | 402,059 | 74.23% |
| 4M | 76,405 | 53,604 | 70.16% |
| 9M | 29,524 | 23,013 | 77.95% |

The harness measured 100 inference batches after each long training run. At batch 64 those windows lasted only
0.09-2.22 seconds, so they are useful implementation diagnostics but are not contention-averaged estimates. Raw
results include batches 1, 8, 32, 64, 128, and 256; the short-window results show load-sensitive reversals at small
batches and should not drive an architecture choice.

## Evaluation completion and excluded reverse replicate

A reverse-order replication sequence began at 21:30:53 UTC. The 9M attention replicate completed while the
evaluation remained `RUNNING`, but the evaluation exited naturally at 21:34 UTC during the following 9M CNN run.
The status command's nonzero `EXITED` result stopped the scripted sequence before the 4M and 1M repeats. Both
`replicate2` JSON files are retained as transition evidence but excluded from the primary table because the CNN run
mixed contended and uncontended intervals.

The fully contended attention transition replicate measured 10,980 samples/s; the mixed-state CNN replicate measured
17,093 samples/s. The discontinuity reinforces why results spanning the evaluation completion must not be averaged
with the primary paired runs.

## Raw artifact hashes

| Result | SHA-256 |
| --- | --- |
| `chess-cnn-1m-equal-wall-120s.json` | `ae765171aba849c235e6c82714364ce1e5bb8e5f88a499e7ad8cb03817e21a85` |
| `chess-attention-1m-equal-wall-120s.json` | `0298f367162a5ed06fd2d5baa411a0080044537266e5b61b881b0babf56496ff` |
| `chess-cnn-4m-equal-wall-120s.json` | `7075c56bc6a5e43baf4b8e4c112f795db1eb13009a771f87e03cf00c6e3d0d74` |
| `chess-attention-4m-equal-wall-120s.json` | `6b351f44ef9c102b8222989a779f280735e9c28f4b38db8cf70ded2ed17fab4a` |
| `chess-cnn-9m-equal-wall-120s.json` | `899dcc594aee79b16cd14a39902ef953a7ea611fd8961620c670b2b76bfe95fa` |
| `chess-attention-9m-equal-wall-120s.json` | `69e0d0096b23e37e378c77e0d60684c84de8e5311e559660e61129880e309205` |

