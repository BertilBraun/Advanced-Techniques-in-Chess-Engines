# Neural-network training parallelism benchmark

This benchmark compares single-GPU training, single-process PyTorch
`DataParallel`, and multi-process `DistributedDataParallel` (DDP) for the
3.24-million-parameter 12×112 chess network. All measurements use BF16 and
resident synthetic batches on four NVIDIA GeForce RTX 4070 SUPER GPUs, so the
results isolate model execution and gradient communication from data loading.

![Training throughput](training-throughput.png)

DDP is the only multi-GPU strategy that improves throughput. At a global batch
of 2,048, four-GPU DDP reaches 53.3 thousand synthetic samples/s in the short
comparison, 2.40× the 22.2 thousand samples/s single-GPU batch-1,024 baseline.
Four-GPU `DataParallel` reaches only 13.3 thousand samples/s because its
single-process scatter, replication, gather, and primary-device reduction
overheads dominate this relatively small network.

The DDP ranks intentionally received identical samples in this model-only
benchmark. The throughput represents executed samples, but the duplicated
inputs provide only one rank's batch diversity. Production DDP must give each
rank a distinct partition to obtain a genuine global batch of 2,048.

![Sustained DDP GPU utilization](ddp-gpu-utilization.png)

The longer 800-batch DDP run sustains 38.6 thousand synthetic samples/s. Mean
streaming-multiprocessor utilization is 56.9% across the four GPUs, while
memory-controller utilization remains between 15% and 18%. The workload is
therefore neither compute-saturated nor memory-bandwidth-saturated. Short
kernels, optimizer work, and gradient synchronization leave capacity that may
be useful for concurrent self-play, although a combined production-path
benchmark is required to measure contention.

GPU 3, which hosted rank zero in this device ordering, averaged 39.3% SM
utilization; the other GPUs averaged 61.7%–63.2%. This observation does not by
itself distinguish rank-specific overhead from hardware or PCIe-topology
effects.

The complete measurements and provenance are in
[`results.json`](results.json). Regenerate both PNG and SVG figures with:

```powershell
python documentation/benchmarks/ddp-model-throughput-20260720/plot_results.py
```
