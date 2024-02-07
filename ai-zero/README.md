# Project: ChessAI-Zero

## Overview

ChessAI-Zero is an ambitious project aiming to replicate and explore the AlphaZero approach to computer chess. The project focuses on developing a chess engine that uses a combination of deep neural networks and Monte Carlo Tree Search (MCTS) to evaluate positions and select moves without reliance on traditional chess engines' databases and heuristics. By learning from self-play, ChessAI-Zero aspires to discover and refine its understanding of chess strategy, tactics, and endgames, pushing the boundaries of artificial chess intelligence.

## Goals

- To implement a neural network that can evaluate chess positions (value) and suggest move probabilities (policy) based on the current board state.
- To integrate the neural network with Monte Carlo Tree Search (MCTS) for effective move selection and game exploration.
- To train the neural network using self-play, allowing the system to learn and improve its chess-playing capabilities iteratively.
- To assess the performance of ChessAI-Zero against various benchmarks, including traditional engines and human players.

## Architecture

ChessAI-Zero comprises two main components: the Neural Network (NN) and the Monte Carlo Tree Search (MCTS).

### Neural Network

- **Input Layer**: Encodes the board state, including piece positions, player to move, castling rights, and en passant possibilities.
- **Processing Layers**: Multiple layers (convolutional neural networks or other suitable architectures) extract features and learn game patterns.
- **Output Layers**:
  - **Policy Head**: Outputs a probability distribution over all legal moves from the current position (64x73 possible moves).
  - **Value Head**: Outputs a single value estimating the likelihood of winning from the current position.

### Monte Carlo Tree Search (MCTS)

- Utilizes the policy head to guide exploration of the game tree.
- Employs the value head to evaluate board positions, aiding in the selection and backpropagation phases.

## Development Plan

1. **Neural Network Implementation**
   - Design and implement the neural network architecture with policy and value heads.
   - Develop a method for encoding chess board states as network inputs.
2. **MCTS Integration**
   - Implement the MCTS algorithm, integrating NN outputs to guide the search.
3. **Self-Play Training System**
   - Create a self-play mechanism to generate training data.
   - Implement training routines for the neural network using self-play data.
4. **Evaluation and Testing**
   - Develop benchmarks and testing protocols to evaluate the performance of ChessAI-Zero.
   - Compare performance against other engines and track improvement over time.

## Technologies

- **Programming Language**: Python 3.x, for its rich ecosystem of libraries and tools for machine learning and chess programming.
- **Machine Learning Frameworks**: TensorFlow or PyTorch, for designing and training the neural network.
- **Chess Library**: python-chess, for handling chess game mechanics, legal move generation, and game state management.

## Getting Started

- Setup instructions, training procedures, and how to run the engine will be detailed in the project documentation.
