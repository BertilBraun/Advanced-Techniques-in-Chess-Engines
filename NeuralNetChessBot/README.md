# Project: AINeural Net Chess Bot

## Overview

AINeural Net Chess Bot  is a sophisticated chess engine project that aims to implement a blend of advanced chess programming techniques with the power of neural network-based evaluation. This project builds on the foundation of a basic chess engine by incorporating neural network (NN) evaluators to replace and enhance traditional evaluation methods where possible. The goal is to leverage the depth of machine learning to understand and evaluate chess positions dynamically, improving the engine's strategic and tactical decision-making capabilities.

## Goals

- To integrate advanced chess engine features such as move ordering, quiescence search, and iterative deepening with a neural network-based evaluation function.
- To design and train a neural network capable of understanding complex chess positions, thereby enhancing the traditional handcrafted evaluation components like material count, piece-square tables, and pawn structure analysis.
- To evaluate the performance impact of replacing traditional evaluation metrics with a neural network, focusing on the engine's playing strength, decision-making process, and computational efficiency.

## Architecture

The project architecture is divided into the Chess Engine Framework and the Neural Network Evaluator, with a focus on how the NN evaluator integrates and enhances the engine's capabilities.

### Chess Engine Framework

- **Advanced Features**:
  - **Move Ordering**: Prioritization of moves to improve search efficiency, initially guided by simple heuristics and enhanced by NN insights.
  - **Quiescence Search**: Extends search at quiet positions to avoid the horizon effect.
  - **Iterative Deepening**: Dynamic adjustment of search depth for optimal time management.
  - **Transposition Table**: Storing previously evaluated positions to avoid redundant calculations.

### Neural Network Evaluator

- **Input Layer**: Encodes the board state into a format suitable for NN processing. This includes piece positions, player to move, castling rights, en passant possibilities, and possibly a simplified representation of the game phase.
- **Hidden Layers**: Multiple layers, potentially including convolutional layers to capture spatial relationships and fully connected layers for high-level strategy learning.
- **Output**: A single value representing the board's evaluation from the current player's perspective, normalized to a range (e.g., -1 to 1).

## Neural Network Architecture for Evaluation

- **Convolutional Neural Network (CNN)** layers at the beginning to process spatial patterns on the chessboard. These layers can identify piece configurations, control of key squares, and potential threats.
- **Fully Connected (FC)** layers following the CNN layers to integrate the spatial information and learn higher-level strategies, evaluating the position globally.
- **Activation Functions**: Use ReLU (Rectified Linear Unit) for hidden layers to introduce non-linearity, allowing the network to learn complex patterns.
- **Output Layer**: A single neuron with a tanh activation function to output a value between -1 and 1, indicating the evaluated score of the position for the current player.

## Integration with Chess Engine

- The NN evaluator is called during the search process to evaluate positions, replacing traditional handcrafted evaluation functions.
- Move ordering may initially use simple heuristics but will be adjusted based on insights from the NN, especially after moves have been evaluated, to prioritize exploration of more promising branches.

## Development Plan

1. **Framework Implementation**: Use the groundwork layed out by the handcrafted chess bot.
2. **Neural Network Design and Training**:
   - Design the NN architecture suitable for chess position evaluation.
   - Train the NN using a dataset of high-quality games, focusing on learning a broad understanding of various positions.
3. **Integration and Testing**:
   - Integrate the NN evaluator with the chess engine framework.
   - Benchmark performance against versions with traditional evaluation functions.

## Technologies

- **Programming Language**: Python 3.x.
- **Machine Learning Frameworks**: TensorFlow or PyTorch.
- **Chess Library**: python-chess.

## Getting Started

- Setup instructions, training procedures, and how to run the engine will be detailed in the project documentation.

## References

- [NNBot](https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results/blob/main/Bots/Bot_529.cs) by Jamie Whiting: A simple chess engine using a neural network for evaluation.
