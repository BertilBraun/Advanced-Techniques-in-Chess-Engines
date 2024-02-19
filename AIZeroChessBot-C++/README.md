# Project: AI-ZeroChessBot-C++

## Current State

The current state of the project is that everything is reimplemented in C++ including the base Chess Framework, the MCTS algorithm, and the Neural Network. The translation of the Chess Framework lead to 50-100x speedup. The project is currently deployed on the cluster and should be training there using 1 Tesla H100 for now.

### Upcoming To Dos

- Profile the performance of the C++ implementation of the MCTS algorithm and the Neural Network compared to the Python implementation. Is the C++ implementation more than 10% in the neural network evaluation step?
- Create a interactive Notebook to evaluate the performance of the bot on the cluster with access to the cluster's GPU.
  - Evaluate the performance of the bot after the first 12 hours of training.
  - Evaluate the performance of the bot against a baseline chess bot (continuously).
- Continue training the bot on the cluster with multiple GPUs and multiple nodes (as already implemented) for multiple days if the performance after 12 hours is promising.

## Overview

AI-Zero Chess Bot is an ambitious project aiming to replicate and explore the AlphaZero approach to computer chess. The project focuses on developing a chess bot that uses a combination of deep neural networks and Monte Carlo Tree Search (MCTS) to evaluate positions and select moves without reliance on traditional chess bots' databases and heuristics. By learning from self-play, the AI-Zero Chess Bot aspires to discover and refine its understanding of chess strategy, tactics, and endgames, pushing the boundaries of artificial chess intelligence.

## Goals

- To implement a neural network that can evaluate chess positions (value) and suggest move probabilities (policy) based on the current board state.
- To integrate the neural network with Monte Carlo Tree Search (MCTS) for effective move selection and game exploration.
- To train the neural network using self-play, allowing the system to learn and improve its chess-playing capabilities iteratively.
- To assess the performance of the AI-Zero Chess Bot against various benchmarks, including traditional bots and human players.

## Architecture

AI-Zero Chess Bot comprises two main components: the Neural Network (NN) and the Monte Carlo Tree Search (MCTS).

### Neural Network

- **Input Layer**: Encodes the board state, including piece positions, player to move, castling rights, and en passant possibilities.
- **Processing Layers**: Multiple ResNet layers extract features and learn game patterns.
- **Output Layers**:
  - **Policy Head**: Outputs a probability distribution over all legal moves from the current position (1968 possible moves).
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
   - Compare performance against other bots and track improvement over time.

## Technologies

- **Programming Language**: C++17
- **Machine Learning Frameworks**: LibTorch (PyTorch C++ API)
- **Chess Library**: Python-Chess (self ported to C++)

## Getting Started

To run your `AIZeroChessBot` project after setting it up with CMake and compiling it with Visual Studio Code (VSCode) on Windows, follow these steps. This guide assumes you've already configured your `CMakeLists.txt` and have the necessary build scripts in place.

### Step 1: Build the Project

1. **Open VSCode** and navigate to your project folder (`AIZeroChessBot`).

2. **Run the Setup Script** (if you haven't already) to download LibTorch and generate the CMake build system. In the VSCode terminal, navigate to your project's root directory and run:

    ```cmd
    .\setup_build.bat
    ```

   This script prepares the build environment, including downloading LibTorch if necessary and generating build files.

3. **Build the Project** using CMake. If you're using the CMake Tools extension in VSCode, you can build the project directly from the VSCode command palette (`Ctrl+Shift+P`) by typing "CMake: Build" and selecting the build target. Alternatively, you can build from the terminal in the `build` directory:

    ```cmd
    cd build
    cmake --build . --config Release
    ```

### Step 2: Running the Executable

After building the project, an executable file named `AIZeroChessBot` (or `AIZeroChessBot.exe` on Windows) will be created in the `build` directory, inside a `Release` or `Debug` subdirectory, depending on your build configuration.

To run your project:

1. **Navigate to the Executable Directory** in the terminal using `cd`:

    ```cmd
    cd Release  # or Debug, depending on your build config
    ```

2. **Run the Executable** by typing its name in the terminal:

    ```cmd
    .\AIZeroChessBot.exe
    ```

   This command executes your program. Any command-line arguments required by your application should follow the executable name.

## Performance Problems

The Python implementation of the MCTS algorithm is slow and inefficient, especially when combined with a neural network evaluation function which should be taking longer to evaluate the board state than the expansion and simulation steps of the MCTS algorithm. This is a performance bottleneck that needs to be addressed.

![MCTS Performance](/AIZeroChessBot/documentation/performance_analysis.png)

Only the small pink section in the performance analysis graph is the time taken to evaluate the board state using the neural network. The rest of the time is spent on the other steps of the MCTS algorithm.

### Reimplementation Plan

**C++ Implementation**: Reimplement the MCTS algorithm in C++ to improve performance.

**What does a Board need to provide FAST:**

- Copy the entire board state
- Generate Legal Moves
- Push a Move (Make a move)
- Get the current player
- Check if the game is over
- Get the game result
- Piece At / Get Representation of all pieces on the Board

Nothing else is needed for the MCTS algorithm to work.
  
For a chess framework, take a look at [Stockfish](https://github.com/official-stockfish/Stockfish) and [Chess-Coding-Adventure](https://github.com/SebLague/Chess-Coding-Adventure/tree/Chess-V2-UCI) for inspiration. For simplicity, the Python-Chess library was ported to C++ and stripped down to the bare minimum needed for the MCTS algorithm to work.

**Machine Learning Frameworks:**

Everything around Machine Learning and Neural Networks can be done in C++ using PyTorch C++ API (LibTorch) or TensorFlow C++ API.

## References

- [AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815.pdf)
- [AlphaZero Chess: How It Works, What Sets It Apart, and What It Can Tell Us](https://towardsdatascience.com/alphazero-chess-how-it-works-what-sets-it-apart-and-what-it-can-tell-us-4ab3d2d08867)
- [AlphaZero Explained](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)
- [AlphaGo Zero Cheat Sheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)
- [AlphaZero from Scratch](https://www.youtube.com/watch?v=wuSQpLinRB4&ab_channel=freeCodeCamp.org)
