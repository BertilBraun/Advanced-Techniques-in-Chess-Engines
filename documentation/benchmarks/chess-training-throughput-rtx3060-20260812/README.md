# Chess training throughput on eight RTX 3060 GPUs

This benchmark diagnoses the stopped `vast-chess-8gpu-1d-r2` run on Vast instance `47400225`. The production run
used source revision `4889318405f22cd4f2d95038b0c7f5674a9e0760`, a 12x112 chess network, global batch 1,024,
four NCCL ranks on GPUs `[0,1,2,3]`, and 24 self-play workers. During each training quantum it paused the 12 workers
on the trainer GPUs and kept the 12 workers on GPUs `[4,5,6,7]` active. The run was explicitly disposable and was
stopped at generation 70 before benchmarking.

The node has eight 12-GiB RTX 3060 GPUs, 64 logical AMD EPYC 7452 CPUs under a 61.44-CPU quota, approximately
188 GiB RAM, driver 580.126.20, PyTorch 2.12.1+cu126, CUDA 12.6, and cuDNN 9.10.2. The benchmark uses the real
generation-70 model and optimizer, the 729,600-row production mmap replay, deterministic disjoint rank sampling,
pinned-memory prefetch, the production objective, gradient clipping, AdamW, NCCL DDP, and checkpoint export.

## Fixed-batch DDP scaling

The comparable sweep fixes global batch size at 1,024 and runs 200 optimizer steps. Each result is one fresh
persistent trainer group and includes the complete rank quantum, including checkpoint publication and the final
barrier, but excludes rank-process initialization.

| DDP GPUs | Local batch | Devices | Replay rows/s | Training samples/s | Speedup over one GPU |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1,024 | `[0]` | 2,202 | 1,854 | 1.00x |
| 2 | 512 | `[6,7]` | 4,142 | 3,001 | 1.62x |
| 4 | 256 | `[0,1,2,3]` | 7,664 | 4,449 | 2.40x |
| 4 | 256 | `[4,5,6,7]` | 7,866 | 4,448 | 2.40x |
| 8 | 128 | `[0,1,2,3,4,5,6,7]` | 14,348 | 5,541 | 2.99x |

The alternate four-GPU placement differs by less than 0.1% in training throughput, so the cross-NUMA GPU 0 is not
the observed slowdown. Eight ranks add only 24.5% over four because local batch 128 provides too little work per
rank relative to DDP synchronization and fixed per-step overhead.

## Self-play contention

The contention workload is the production active subset: three self-play processes per GPU on GPUs `[4,5,6,7]`,
1,024 games per process, two inference workers, batch 64, and two outstanding batches per worker. DDP uses the same
production replay and batch 1,024.

| DDP case | Optimizer steps | Replay rows/s | Training samples/s | Change from isolated |
| --- | ---: | ---: | ---: | ---: |
| 4 GPUs `[0,1,2,3]` | 200 | 6,223 | 3,594 | -19.2% |
| 4 GPUs `[0,1,2,3]` | 500 | 6,269 | 3,707 | -16.5% |
| 8 GPUs, sharing four self-play GPUs | 200 | 11,354 | 3,652 | -34.1% |

At the configured batch, eight-GPU DDP under contention is only 1.6% faster than four-GPU DDP and also occupies the
self-play devices. The 12-worker self-play harness sustained 108,917 searches/s across a 374-second mixed-contention
window, with mean inference batch 61.98/64, 2,508% aggregate process CPU, and 28,055 MiB summed peak RSS. The window
contains the controlled four- and eight-GPU training intervals, so it is evidence for continued high self-play
throughput under the experiment load, not an isolated self-play baseline or a clean measurement of the eight-GPU
DDP effect on self-play.

## Batch-size sweep

The batch sweep keeps the total presented samples at 409,600 for batch sizes 2,048 and above, so it uses 200, 100,
and 50 optimizer steps respectively. The batch-1,024 reference uses 200 steps and 204,800 samples.

| DDP GPUs | Global batch | Local batch | Training samples/s | Change from batch 1,024 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1,024 | 256 | 4,449 | reference |
| 4 | 2,048 | 512 | 5,236 | +17.7% |
| 4 | 4,096 | 1,024 | 5,654 | +27.1% |
| 8 | 1,024 | 128 | 5,541 | reference |
| 8 | 2,048 | 256 | 7,947 | +43.4% |
| 8 | 4,096 | 512 | 9,876 | +78.2% |
| 8 | 8,192 | 1,024 | 10,429 | +88.2% |

Batch 8,192 adds only 5.6% over 4,096 on eight GPUs, so the hardware throughput curve is already flattening. These
are systems measurements, not authorization to change optimization semantics: a larger batch changes update count,
gradient noise, replay consumption per step, and likely the appropriate learning-rate schedule.

## End-to-end production timing

Generation 70 reported 5,367 hot-loop samples/s and 14.6 seconds of credit wait. Its 512,000 presentations imply a
95.4-second trainer quantum. The log interval from generation 69 to 70 was 153 seconds, leaving approximately
43 seconds outside both reported training and credit wait. The coordinator path shows that this uninstrumented time
is the barrier that pauses selected workers, a final replay ingestion, applying the new checkpoint to all 24
self-play workers, collecting generation statistics, resetting search trees, and checkpoint retention.

The effective generation-70 wall-clock presentation rate was therefore about 3,346 samples/s, far below the
reported hot-loop rate. A production self-play worker batch takes roughly 25 seconds in the contention harness, so
waiting for every worker to reach a model-refresh boundary plausibly explains most of the 43-second gap. Exact phase
timers should be added before attributing the whole gap to one operation.

The run accumulated a presentation backlog early: available presentations reached roughly 10 million and were
7.85 million at generation 70, while observed replay ratio was 6.563 rather than the configured 8. The backlog was
already falling over the final generations as scheduled searches became more expensive, so the trainer was catching
up at shutdown even though it had not met the cumulative target ratio.

## Conclusions

1. The current four-GPU placement is not defective; its topology is effectively tied with the alternate grouping.
2. Global batch 1,024 is too small for efficient eight-rank DDP and leaves measurable throughput on the table even
   with four ranks.
3. CPU/self-play contention costs the four-rank trainer about 17-19%, primarily visible as reduced replay-loader
   throughput. The loader is not the isolated ceiling: at eight GPUs it supplies about 14-15k rows/s while training
   consumes at most 10.4k samples/s.
4. Eight-rank DDP at batch 1,024 is not worthwhile while self-play remains active. Eight ranks become useful at
   global batch 4,096 or larger, preferably with all self-play workers paused during the training phase.
5. The next production-safe experiment is four ranks with global/local batch 2,048/512, followed by a learning-curve
   check because this restores the original authored batch but changes the r2 optimization recipe.
6. The next orchestration experiment is a longer optimizer quantum, such as 1,000 steps, to amortize the roughly
   43-second model-refresh barrier. This trades lower overhead for older self-play models and less frequent
   checkpoints and therefore needs elapsed-strength validation.
7. Add explicit coordinator timers for pause acknowledgement, final ingestion, DDP, checkpoint application,
   statistics collection, and retention. Mixed precision and `torch.compile` are promising separate experiments;
   neither was measured here and neither should be presented as an established gain.

Raw JSON files in this directory contain every summarized DDP result and the complete self-play manifest and worker
summary. The authoritative remote evidence remains under `/workspace/training-benchmarks` while the ephemeral node
exists.
