# Chess Bot Projects Repository

## Overview

This repository hosts a collection of four distinct chess bot projects, each exploring different approaches to chess AI development, ranging from handcrafted algorithms to advanced machine learning and beyond. The addition of the AI-ZeroChessBot-C++ project marks a significant milestone in our exploration, representing the most developed and sophisticated chess engine in our collection. It is a direct port of the AI-ZeroChessBot to C++, further developed to leverage the performance benefits of the language. From foundational frameworks to cutting-edge techniques, these projects aim to cover a comprehensive spectrum of AI strategies in the domain of chess. Each project is contained within its own subfolder, allowing for focused development and exploration.

## Projects

### 0. [Framework](/Framework/README.md) (Subfolder: `Framework`)

Provides the basic infrastructure for evaluating and playing chess games, serving as the foundation for the other projects. Includes essential classes such as `ChessBot`, `GameManager`, `HumanPlayer`, and `Tournament`.

### 1. [HandcraftedChessBot](/HandcraftedChessBot/README.md) (Subfolder: `HandcraftedChessBot`)

Focuses on traditional chess bot design principles, employing handcrafted evaluation functions and optimization techniques without machine learning.

**Key Features**:

- Alpha-beta pruning and iterative deepening.
- Handcrafted evaluation functions focusing on material count, piece-square tables, and pawn structure.

### 2. [NeuralNetChessBot](/NeuralNetChessBot/README.md) (Subfolder: `NeuralNetChessBot`)

Explores the integration of a neural network-based evaluator within a chess bot framework, aiming to assess chess positions dynamically through machine learning.

**Key Features**:

- Neural network for position evaluation.
- Training of the neural network with chess game datasets.

### 3. [AI-ZeroChessBot](/AIZeroChessBot/README.md) (Subfolder: `AI-ZeroChessBot`)

Inspired by AlphaZero, this project combines deep neural networks with Monte Carlo Tree Search (MCTS) for move selection and position evaluation, learning from self-play.

**Key Features**:

- Deep neural network with policy and value heads.
- MCTS for strategic move selection.
- Self-play for continuous learning.

### 4. [AI-ZeroChessBot-C++](/AIZeroChessBot-C++/README.md) (Subfolder: `AI-ZeroChessBot-C++`)

A significant evolution of the AI-ZeroChessBot, ported to C++ and further developed to harness the language's performance capabilities. This project represents the pinnacle of our chess engine development, emphasizing efficiency, advanced strategies, and self-improvement through deep learning and MCTS.

**Key Features**:

- Ported to C++ for enhanced performance and efficiency.
- Advanced integration of deep learning and MCTS for superior strategic depth.
- Focus on self-improvement through extensive self-play and continuous learning.

### 5. [AI-ZeroConnect4Bot](/AIZeroConnect4Bot/README.md) (Subfolder: `AI-ZeroConnect4Bot`)

This project simplifies the game to Connect 4 and applies the AI-Zero approach to develop a Connect 4 bot. The project aims to verify the correctness and effectiveness of the AI-Zero approach in a simpler game setting. Once the Connect 4 bot is successfully developed, the project will be extended to attempt the same approach in more complex games like Chess.

**Key Features**:

- Connect 4 game environment.
- Optimizations for Inference like Batch Inference and Caching.
- Optimized Hyperparameters and actual good achieved performance.

## Repository Structure

```text
ChessBot/
│
├── Framework/
│   └── README.md
│
├── HandcraftedChessBot/
│   └── README.md
│
├── NeuralNetChessBot/
│   └── README.md
│
├── AI-ZeroChessBot/
│   └── README.md
│
└── AI-ZeroChessBot-C++/
    └── README.md
```

## Getting Started

To get started with any of the projects:

1. Clone this repository to your local machine.
2. Navigate to the project subfolder of interest (`HandcraftedChessBot`, `NeuralNetChessBot`, `AI-ZeroChessBot`, or `AI-ZeroChessBot-C++`).
3. Follow the instructions in the project-specific README.md for setup and running the bot.
4. To explore the basic chess bot framework, refer to the `Framework` subfolder and try it out via `python -m Framework`.
