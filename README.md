# Chess Engine Projects Repository

## Overview

This repository hosts a collection of three distinct chess engine projects, each exploring different approaches to chess AI development. From handcrafted algorithms to cutting-edge machine learning techniques, these projects aim to cover a broad spectrum of AI strategies in the realm of chess. Each project is contained within its own subfolder, allowing for focused development and exploration.

## Projects

### 1. HandcraftedChessBot (Subfolder: `HandcraftedChessBot`)

A traditional chess engine that relies on handcrafted evaluation functions, search algorithms, and optimization techniques without the use of machine learning. This project focuses on the fundamentals of chess engine design, including move generation, board evaluation, and search strategies like alpha-beta pruning and iterative deepening.

**Key Features**:

- Efficient move generation and legal move checking.
- Alpha-beta pruning and iterative deepening for search optimization.
- Handcrafted evaluation functions including material count, piece-square tables, and pawn structure analysis.

### 2. NeuralNetChessBot (Subfolder: `NeuralNetChessBot`)

This project introduces a neural network-based evaluator within a basic chess engine framework. It aims to explore the impact of replacing traditional, handcrafted evaluation methods with a machine learning model trained to understand and assess chess positions.

**Key Features**:

- Basic chess engine capabilities with move generation and game state management.
- Integration of a neural network for dynamic position evaluation.
- Training and implementation of the neural network using chess game datasets.

### 3. AI-ZeroChessBot (Subfolder: `AI-ZeroChessBot`)

Inspired by AlphaZero, this ambitious project seeks to implement a chess engine that combines deep neural networks with Monte Carlo Tree Search (MCTS) to guide move selection and position evaluation. It focuses on learning from self-play without relying on pre-existing game databases or handcrafted evaluation heuristics.

**Key Features**:

- Deep neural network with policy and value heads to evaluate positions and suggest moves.
- Monte Carlo Tree Search integration for explorative and strategic move selection.
- Self-play training mechanism for continuous learning and improvement.

## Repository Structure

```text
ChessBot/
│
├── HandcraftedChessBot/
│   └── README.md
│
├── NeuralNetChessBot/
│   └── README.md
│
└── AI-ZeroChessBot/
    └── README.md
```

## Getting Started

To get started with any of the projects:

1. Clone this repository to your local machine.
2. Navigate to the project subfolder of interest (`HandcraftedChessBot`, `NeuralNetChessBot`, or `AI-ZeroChessBot`).
3. Follow the instructions in the project-specific README.md for setup and running the engine.
