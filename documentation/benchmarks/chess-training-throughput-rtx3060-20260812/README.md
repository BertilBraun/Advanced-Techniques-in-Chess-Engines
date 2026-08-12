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

The original contention workload is the r2 production active subset: three self-play processes per GPU on GPUs
`[4,5,6,7]`, 1,024 games per process, two inference workers, batch 64, and two outstanding batches per worker. DDP
uses the same production replay and batch 1,024.

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

The selected topology instead kept one self-play worker active on every GPU while four 500-step, eight-rank DDP
trials ran. Those eight workers sustained 169,157 searches/s over 1,209 seconds, completed 204.6 million searches,
and averaged 60.66 positions per inference call. This is the directly relevant self-play result for the proposed
pause-16/keep-8-active topology.

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

## Mixed precision and compilation

The selected-mode sweep fixes eight DDP ranks, global/local batch 2,048/256, and 500 optimizer steps. Exactly one
self-play worker remained active on every GPU for the entire sweep. Each trial starts fresh from generation 70 and
publishes a generation-71 eager and JIT checkpoint.

| Precision | `torch.compile` | Quantum seconds | Training samples/s | Change from eager FP32 |
| --- | --- | ---: | ---: | ---: |
| FP32 | disabled | 178.85 | 5,726 | reference |
| BF16 autocast | disabled | 163.80 | 6,252 | +9.2% |
| FP32 | default | 220.57 | 4,643 | -18.9% |
| BF16 autocast | default | 218.35 | 4,690 | -18.1% |

BF16 autocast is the fastest tested mode and saves 15.1 seconds per quantum. Compilation is a clear regression on
this network and hardware. The compiled trials also reported a DDP gradient-stride mismatch for a 1x1 parameter and
that the RTX 3060 has too few streaming multiprocessors for max-autotune GEMM. Compilation remains configurable for
future models or PyTorch releases, but it should be disabled for this run. Training compilation does not replace or
alter JIT inference export: rank zero saves the canonical eager model, and checkpoint publication scripts a fresh
inference model exactly as before.

## End-to-end production timing

Generation 70 reported 5,367 hot-loop samples/s and 14.6 seconds of credit wait. Its 512,000 presentations imply a
95.4-second trainer quantum. The log interval from generation 69 to 70 was 153 seconds, leaving approximately
43 seconds outside both reported training and credit wait.

A direct reproduction with the real 24-worker group and generation-69/70 JIT checkpoints accounts for essentially
all of it. The r2 pause-12 topology spent 17.33 seconds waiting for pause acknowledgements and 24.50 seconds applying
the next checkpoint to all workers: 41.84 seconds combined. The proposed pause-16 topology measured 17.96 seconds
and 17.74 seconds respectively: 35.71 seconds combined. Workers only inspect commands between whole self-play
batches, so the coordinator is waiting for batch boundaries; it is not losing time in DDP. Applying a new checkpoint
also validates and loads the JIT model, collects generation statistics, and rebuilds worker search state.

The effective generation-70 wall-clock presentation rate was therefore about 3,346 samples/s, far below the
reported hot-loop rate. Coordinator telemetry now records pause acknowledgement, final ingestion, checkpoint
application, retention, and total transition time independently so subsequent production generations expose this
cost directly.

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
4. Eight-rank DDP is useful at global batch 2,048 when only one worker remains active per GPU: eager BF16 reaches
   6,252 samples/s while those workers collectively sustain 169,157 searches/s.
5. The selected training mode is eight ranks, global/local batch 2,048/256, BF16 autocast, compilation disabled,
   pause worker IDs `[1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23]`, and the unchanged 500-step quantum.
6. The missing time is a synchronous self-play transition barrier, not an unidentified trainer stall. Removing
   acknowledgement without another consistency protocol would allow games from stale generations to cross the
   checkpoint boundary. Reducing worker batch granularity is the safer follow-up experiment.
7. The checked-in `vast-chess-8gpu-1d-r3.yaml` captures the selected recipe. It remains an explicitly approved clean
   run and was not launched during benchmarking.

Raw JSON files in this directory contain every summarized DDP result and the complete self-play manifest and worker
summary. The authoritative remote evidence remains under `/workspace/training-benchmarks` while the ephemeral node
exists.
