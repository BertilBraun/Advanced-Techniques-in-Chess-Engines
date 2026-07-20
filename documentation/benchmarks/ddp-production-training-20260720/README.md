# Production DDP training benchmark

These measurements validate the persistent production `TrainerProcess` on the
four-GPU compute node without starting `chess_clean_main` or creating a training
approval. The authoritative isolated result uses replay files whose metadata
matches the current 1,880-action, 29-plane chess representation.

## Isolated sustained run

The run used NCCL ranks on devices `(3, 2, 1, 0)`, a global batch of 2,048, and a
local batch of 512. Ten recorded replay iterations were staged in deterministic
directory and filename order until at least 1,638,400 samples were available.
The production loader found 1,641,347 samples, retained 801 complete global
batches (1,640,448 samples), and dropped the final 899 samples. Each rank
processed 410,112 distinct samples in 801 optimizer steps. The sampler adds no
padding, so the rank partitions have no overlap and no duplicated samples.

The complete training phase sustained 22,599 samples/s, while the synchronized
optimizer interval sustained 23,097 samples/s. Peak aggregate resident memory
for the coordinator and four rank processes was 12,404.9 MiB. Per-GPU peak
memory was 1,279 MiB on rank zero and 1,015 MiB on each other rank.

The command was:

```bash
cd /workspace/chess/source/py
PYTHONPATH=. /venv/main/bin/python tools/smoke_production_ddp.py \
  --device-ids 3 2 1 0 \
  --samples 1638400 \
  --replay-source-directories \
    training_data/complete-training-run-v2/memory_2 \
    training_data/complete-training-run-v2/memory_0 \
    training_data/complete-training-run-v2/memory_3 \
    training_data/complete-training-run-v2/memory_5 \
    training_data/complete-training-run-v2/memory_4 \
    training_data/complete-training-run-v2/memory_6 \
    training_data/complete-training-run-v2/memory_1 \
    training_data/complete-training-run-v2/memory_7 \
    training_data/complete-training-run-v2/memory_8 \
    training_data/complete-training-run-v2/memory_10 \
  --work-directory /tmp/ddp-production-real-replay \
  --output-path /tmp/ddp-production-real-replay.json \
  --monitor-interval-seconds 0.5
```

Each persistent spawned rank owns an eager, read-only
`RollingSelfPlayBuffer`. This deliberately uses four replay-buffer copies
because its Python byte lists and ragged NumPy policy arrays cannot be shared
zero-copy safely across persistent spawned processes. The measured 12.1 GiB
process-tree peak includes that tradeoff. The coordinator no longer holds a
fifth replay copy.

## Half-self-play contention run

The contention run launched 20 isolated self-play benchmark workers: five per
GPU, three Monte Carlo tree search threads and 96 parallel games per process,
600 full searches, 100 fast searches, and inference caching disabled. While
they ran, the real production DDP coordinator completed 200 independent
partitioned optimizer steps.

Under contention, DDP sustained 15,026 samples/s over the complete phase and
15,890 samples/s inside the optimizer interval. Combined GPU utilization
averaged 96.6–97.1% in the 120-second self-play measurement. The self-play
workers completed 278 games and 3,628 generated samples, and performed
16,200,932 inference evaluations in 385,472 model calls.

The DDP process-tree peak was 9,819.8 MiB and the sum of the 20 workers'
individual peak resident sets was 27,192.4 MiB. These are different sampling
domains and must not be added as a simultaneous host-memory peak. The resource
monitor observed a 25.875% peak for total host RAM. GPU memory peaks, including
both workloads, ranged from 2,441 to 5,252 MiB.

## Interpretation and limitations

The real-replay result supersedes the earlier 38,573 samples/s resident-batch
DDP measurement for production planning because it includes deterministic
global shuffling, non-overlapping partition construction, vectorized decoding,
pinning, host-to-device transfer, and rank synchronization. The resident-batch
result remains useful as a model/collective ceiling, and the isolated
DataParallel result remains recorded at 13,346 samples/s.

The benchmark reuses compatible historical replay shards rather than samples
from the not-yet-started v4 run. It exercises the production training path but
does not exercise Commander orchestration; pause/resume cleanup is covered by
focused tests. The contention run used the generated replay fixture, so its DDP
host-memory number is not representative of the recorded-replay run. No
long-running training process was started.
