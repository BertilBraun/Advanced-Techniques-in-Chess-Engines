# Training Optimization

## Improved Training Target

The value target is typically hard for the model to learn, as the value target is only a binary signal of the game result. Wrong choises during the game search can completely change the outcome of the game. To have a better value target, we augment the value target with the results of the MCTS search. Just as we base the policy target on the MCTS search, we also base the value target on the MCTS search. The value target is then the average of the value of the MCTS search and the game result. This way, the model has a better value target and the training is more stable.

## Deduplication

The same state can be reached by different move sequences as well as symmetries of the board or simply by parallel self-play games. By analyzing the samples on which to train, we can deduplicate the samples and only train on the unique samples. Given the same state, the action probabilities and the value target are averaged. In theory, the model should be able to learn this by itself, but in practice, this deduplication step greatly improves the training speed and the quality of the model.

## Sliding Window

The training samples are based on the most recent iterations of training. We do not only use the samples from the most recent iteration, but also from the previous iterations. The window size is based on the current iteration and the number of iterations. The window size grows linearly with the number of iterations, but is capped at a maximum window size. This way, the model is trained on a diverse dataset and the training is more stable. Removing the oldest samples relatively early in the training process provides a large boost in training performance, as these early samples are most akin to random noise (as the model is not yet trained).

## Shuffling / Smart Data Loading

The self-players write out their games in lists of roughly 2000 samples with the iteration used to generate them as metadata. During training, the samples are loaded and the trainer waits until a predefined number of samples for the most recent iteration are available. This ensures that the self-players have generated enough representative samples for the current iteration. The samples are then shuffled and written to disk in chunks of roughly 1GB. During training, multiple of these chunks are loaded and shuffled in memory. The trainer trains on the samples in order and once all samples are used, the next chunks are loaded. This way, the samples are not all loaded into memory at once but the loaded samples are shuffled, so that not only are samples which were generated close by another (like the symmetries of the board) are not all loaded at once, but also the samples are loaded from different iterations. This way, the model is trained on a diverse dataset and the training is more stable.

### Dataset

To reduce the IO and memory footprint, the samples are stored in a custom binary format of the canonical board states. As these are large, sparse binary arrays, they are losslessly compressed by encoding them into 64bit integers as bitfields. This way, the samples are stored in a more compact format and the IO and memory footprint is reduced. Only for training, the samples are decoded back into the original format. The encoding/decoding is hereby very performant, as it is reduced to 2 numpy bit operations in total.

## 1 Cycle learning rate policy

For stability during training, since only 1 epoch (at most 2) is used, the 1 cycle learning rate policy is used. This policy starts with a low learning rate, then increases the learning rate linearly to a maximum learning rate and then decreases the learning rate linearly to a minimum learning rate. Empirically, this policy has been shown to work well for training neural networks and is used in this project to stabilize the training process.

## Mixed Precision Training

To further improve the training speed, mixed precision training is used. This means that the model is trained with 16-bit floating point numbers instead of 32-bit floating point numbers. This greatly reduces the memory footprint and increases the training speed.
