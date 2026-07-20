# Training Optimization

## Improved Training Target

The value target is typically hard for the model to learn, as the value target is only a binary signal of the game result. Wrong choises during the game search can completely change the outcome of the game. To have a better value target, we augment the value target with the results of the MCTS search. Just as we base the policy target on the MCTS search, we also base the value target on the MCTS search. The value target is then then a linar interpolation between the value of the MCTS search and the game result. The interpolation factor is based on the iteration, as in early iterations, the Neural Network is not yet trained and the value head is not yet reliable. In later iterations, the value head is more reliable and therefore also the MCTS search is more reliable. The interpolation factor therefore grows from 0 to ~0.3 over the first 10% of the training iterations. This way, the model is trained on a better value target and the training is more stable. The value target is then calculated as:

```
value_target = (1 - interpolation_factor) * game_result + interpolation_factor * mcts_value
```

Where `mcts_value` is the value of the MCTS search and `game_result` is the result of the game (1 for win, 0 for draw, -1 for loss). The interpolation factor is a value between 0 and 1 that grows over the first 10% of the training iterations. This way, the model is trained on a better value target and the training is more stable.

## Deduplication

The same state can be reached by different move sequences as well as symmetries of the board or simply by parallel self-play games. By analyzing the samples on which to train, we can deduplicate the samples and only train on the unique samples. Given the same state, the action probabilities and the value target are averaged. In theory, the model should be able to learn this by itself, but in practice, this deduplication step greatly improves the training speed and the quality of the model.

## Sliding Window

The training samples are based on the most recent iterations of training. We do not only use the samples from the most recent iteration, but also from the previous iterations. The window size is based on the current iteration and the number of iterations. The window size grows linearly with the number of iterations, but is capped at a maximum window size. This way, the model is trained on a diverse dataset and the training is more stable. Removing the oldest samples relatively early in the training process provides a large boost in training performance, as these early samples are most akin to random noise (as the model is not yet trained).

## Shuffling / Smart Data Loading

The self-players write out their games in lists of roughly 2000 samples with the iteration used to generate them as metadata. During training, the samples are loaded and the trainer waits until a predefined number of samples for the most recent iteration are available. This ensures that the self-players have generated enough representative samples for the current iteration. The samples are thereby kept in memory to allow for fast access and shuffling during training. We hereby use a rolling buffer to keep in the latest N number of iterations while also limiting to a max number of samples to train on (typically 4 million samples). This way, the training is more stable and the model is trained on a diverse dataset. The samples are shuffled before training to ensure that the model does not overfit on the samples and learns a more general representation of the game.

### Dataset

To reduce the IO and memory footprint, the samples are stored in a custom binary format of the canonical board states. As these are large, sparse binary arrays, they are losslessly compressed by encoding them into 64bit integers as bitfields. This way, the samples are stored in a more compact format and the IO and memory footprint is reduced. Only for training, the samples are decoded back into the original format. The encoding/decoding is hereby very performant, as it is reduced to 2 numpy bit operations in total.

## 1 Cycle learning rate policy

For stability during training, since only 1 epoch (at most 2) is used, the 1 cycle learning rate policy is used. This policy starts with a low learning rate, then increases the learning rate linearly to a maximum learning rate and then decreases the learning rate linearly to a minimum learning rate. Empirically, this policy has been shown to work well for training neural networks and is used in this project to stabilize the training process.

## Mixed Precision Training

To further improve the training speed, mixed precision training is used. This means that the model is trained with `torch.amp.autocast` with `torch.bfloat16` numbers instead of `torch.bfloat32` numbers. This greatly reduces the memory footprint and increases the training speed while preserving the accuracy on required operations, therefore minimally affecting the model performance. This way, the model can be trained with a larger batch size and the training speed is increased.

## Multi-GPU Training

Single-process PyTorch `DataParallel` is a poor fit for the 3.24-million-parameter
chess network. Its primary-device scatter, per-step replication, output gather,
and gradient reduction overheads outweigh the additional GPU compute.
Multi-process `DistributedDataParallel` (DDP) instead keeps one persistent model
replica per GPU and synchronizes bucketed gradients with NCCL.

![Single GPU, DataParallel, and DDP throughput](../benchmarks/ddp-model-throughput-20260720/training-throughput.png)

In a model-only benchmark with resident BF16 batches, four-GPU DDP processes
53.3 thousand synthetic samples/s, compared with 22.2 thousand samples/s for
the single-GPU batch-1,024 baseline and 13.3 thousand samples/s for four-GPU
`DataParallel`. The production trainer reaches 22.6 thousand samples/s with
real replay loading, independent deterministic rank partitions, vectorized
decoding, checkpointing, and retention.

![Four-GPU DDP utilization](../benchmarks/ddp-model-throughput-20260720/ddp-gpu-utilization.png)

Isolated production DDP uses 32.7%–58.3% mean SM utilization by GPU and about
10% memory-controller capacity. Running five self-play processes per GPU during
optimization raises mean SM utilization to 96.6%–97.1%, while DDP continues at
15.0 thousand samples/s and self-play continues generating games. The
production sampler assigns disjoint rank-local partitions, drops only the
incomplete global batch, and performs no duplicate padding.

See the [benchmark artifact](../benchmarks/ddp-model-throughput-20260720/README.md)
for the comparison and plots, and the
[production benchmark](../benchmarks/ddp-production-training-20260720/README.md)
for the authoritative replay, resource, and contention measurements.
