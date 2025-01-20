# Chess Engine C++ Port Documentation

This README outlines the rationale, the performance considerations, and the key architectural decisions behind porting our chess framework from Python to C++, focusing on the integration with the Monte Carlo Tree Search (MCTS) algorithm.

## Transition to C++

The initiative to reimplement our chess framework in C++ emerged from the anticipation of enhanced efficiency and speed, crucial for processing complex algorithms like MCTS. Post-transition benchmarks confirmed a significant performance boost, with a 50-100x speedup in execution over the Python version, demonstrating the substantial benefits of adopting C++.

## Performance Problems in Python Implementation

In the original Python implementation, the performance of the MCTS algorithm was notably hindered by its slow execution speed. This was particularly problematic when paired with a neural network for board state evaluation, where the MCTS steps, instead of the neural network evaluation, consumed a disproportionate amount of computational time.

## Simplified Chess Framework Requirements

The chess framework was streamlined to include only essential functionalities needed by the MCTS algorithm. This approach not only facilitated performance improvements but also established a flexible foundation for adapting the framework to other board games. The engine's requirements are as follows:

- **Copy the entire board state:** For evaluating potential moves without altering the current state.
- **Generate legal moves:** To explore possible future game states.
- **Push a move (Make a move):** For simulating game progressions.
- **Get the current player:** To determine the active player.
- **Check if the game is over:** To identify endgame scenarios.
- **Get the game result:** For evaluating the outcome of the game.
- **Piece At / Get Representation of all pieces on the Board:** Essential for neural network evaluations of the board.

These requirements ensure that the framework is not only optimized for chess but also adaptable for other games, such as Go, with minimal modifications.

## Reference Implementation

The chess framework ported to C++ was based on the Python chess package, essentially a one-to-one translation aimed at simplicity and efficiency. While not as optimized as specialized C implementations like Stockfish, the ported version results in a compact and reasonably fast chess engine, encapsulated in a single header file (`chess.hpp`) with about 2.5k lines of code. This balance between simplicity and performance makes it an ideal reference for integrating with our MCTS framework.

## GPU Utilization and Parallelization

A notable advantage of the C++ implementation is the efficient utilization of GPU resources. We achieved consistent 20% GPU usage, a significant improvement from the Python version. Moreover, by employing 5 threads per GPU, we managed to maximize CPU usage, ensuring nearly 100% utilization. This parallelization strategy is detailed in our parallelization [README](../parallelization/README.md), offering insights into achieving optimal performance.
