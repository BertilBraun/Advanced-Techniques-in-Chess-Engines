# Implementation Details

We only provide a brief overview of the key components and classes in the AlphaZero-Clone project. For a more detailed explanation, please refer to the source code and documentation in the repository. Since the project is under active development, the implementation details may change over time.

In general, the main implementation is currently in Python using PyTorch. Different [Architectures](optimizations/architecture.md) and [Optimizations](../README.md#optimizations) are tried out and documented in the repository. The project is planned to be translated to C++ for better scalability and performance.

## [Neural Network Model](network.md)

Describes the neural network architecture used in the AlphaZero-Clone project. The model consists of a shared representation Res-Net followed by policy and value heads. The model is trained using self-play data generated by the Monte Carlo Tree Search (MCTS) algorithm.

## [Game and Board](games.md)

To ensure flexibility and support for multiple games, abstract base classes for `Game` and `Board` are defined. These classes enforce a consistent interface across different game implementations. To implement a new game, one must subclass these abstract classes and provide the necessary methods. Other than that, only hyperparameters and game-specific logic need to be defined, the rest of the pipeline remains the same.

## [Encodings for Neural Networks](encodings.md)

Describes the encodings used to represent the game state for the neural network.

## [Chess Framework](chess/README.md)

Details the implementation of the chess framework in C++ and the performance improvements achieved through the translation of the framework from Python to C++.

## [Parallelization](parallelization/README.md)

Explains the parallelization strategy used to distribute the training data generation and training processes across multiple nodes and GPUs on the cluster.
