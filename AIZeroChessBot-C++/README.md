# Project: AI-ZeroChessBot-C++

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

- **Input Layer**: Encodes the board state, including piece positions.
- **Processing Layers**: Multiple ResNet layers extract features and learn game patterns.
- **Output Layers**:
  - **Policy Head**: Outputs a probability distribution over all legal moves from the current position (1968 possible moves).
  - **Value Head**: Outputs a single value estimating the likelihood of winning from the current position.

For this NN architecture, we require a board encoding scheme that translates the chess board into a tensor representation, along with methods for move encoding and handling moves with associated probabilities. The design choices for the board encoding scheme are documented in [Chess Encoding for Neural Networks](/AIZeroChessBot-C++/documentation/encodings/README.md).

### Monte Carlo Tree Search (MCTS)

- Utilizes the policy head to guide exploration of the game tree.
- Employs the value head to evaluate board positions, aiding in the selection and back propagation phases of MCTS.

## Documentation

The project documentation is organized into several categories, each focusing on a specific aspect of the project. The documentation provides more detailed information about specific components, development processes, and performance analysis. The documentation categories are as follows:

- **[Chess Encoding for Neural Networks](/AIZeroChessBot-C++/documentation/encodings/README.md)**: Describes the board encoding scheme used to represent chess board states as inputs to the neural network.
- **[Chess Framework](/AIZeroChessBot-C++/documentation/chess/README.md)**: Details the implementation of the chess framework in C++ and the performance improvements achieved through the translation of the framework from Python to C++.
- **[Pre-Training System](/AIZeroChessBot-C++/documentation/pretraining/README.md)**: Discusses the pre-training system used to generate training data for the neural network using grandmaster games and stockfish evaluations.
- **[Parallelization](/AIZeroChessBot-C++/documentation/parallelization/README.md)**: Explains the parallelization strategy used to distribute the training data generation and training processes across multiple nodes and GPUs on the cluster.

## Technologies

- **Programming Language**: C++17
- **Machine Learning Frameworks**: LibTorch (PyTorch C++ API)
- **Chess Library**: Python-Chess (self ported to C++)

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

## Getting Started

To run your `AIZeroChessBot-C++` project after cloning the repository, follow these steps:

### Step 1: Build the Project

1. **Run the Setup Script** (if you haven't already) to download LibTorch and generate the CMake build system. In the terminal, navigate to your project's root directory and run:

    ```cmd
    setup_build.bat
    ./setup_build.sh
    ```

   This script prepares the build environment, including downloading LibTorch if necessary and generating build files.

2. **Build the Project** using CMake.

    ```cmd
    cd build
    cmake --build . --config Release
    ```

    Or alternatively:

    ```cmd
    cd build
    make
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
    AIZeroChessBot <train|generate> <num_workers>
    ```

   This command executes your program.

   - The first argument specifies the mode (`train` or `generate`).
     - `train` mode trains the neural network while simultaneously generating self-play data.
     - `generate` mode generates data for training the neural network from datasets of grandmaster games and stockfish evaluations. Read [Self-Play Pre-Training System](/AIZeroChessBot-C++/documentation/pretraining/README.md) for more information.
   - The second argument specifies the number of workers to use for the specified mode.

**On the Cluster**: To run the project on the cluster, you can use the provided `train/train.sh` script to submit a job to the cluster. The script will handle the build and execution of the project on the cluster. To submit a job, run the following command:

```bash
cd train
sbatch train.sh
```

### Step 3: Interacting with the Project

For evaluating the performance of the AI-Zero Chess Bot, we have a jupyter notebook that can be used to interactively evaluate the bot's performance against a baseline chess bot and track its improvement over time. The notebook will also provide visualizations and metrics to assess the bot's learning progress. The notebook will be deployed on the cluster and will have access to the cluster's GPU for running the bot evaluations. In the notebook, we will be able to play against the bot and observe its moves and strategies in real-time.

To run the evaluation notebook, follow these steps:

1. **Setup the eval Build**: The evaluation notebook requires a new compiled build with the evaluation mode enabled. Run the following commands to set up the evaluation build:

    ```bash
    cd eval
    ./setup_eval_build.sh
    ```

    This will create a new build with the evaluation mode enabled.
2. **Open the Evaluation Notebook**: Open the `eval.ipynb` notebook in Jupyter.
3. **[Optional] Download Model Weights**: If you want to use pre-trained model weights for the evaluation, download the model weights from [here](/documentation/) (TODO: Add link) and place them in the `train` directory.
4. **Run the Notebook**: Execute the cells in the notebook to start the evaluation process. The notebook will guide you through the evaluation steps and display the bot's performance metrics and visualizations.

## References

- [AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815.pdf)
- [AlphaZero Chess: How It Works, What Sets It Apart, and What It Can Tell Us](https://towardsdatascience.com/alphazero-chess-how-it-works-what-sets-it-apart-and-what-it-can-tell-us-4ab3d2d08867)
- [AlphaZero Explained](https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/)
- [AlphaGo Zero Cheat Sheet](https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0)
- [AlphaZero from Scratch](https://www.youtube.com/watch?v=wuSQpLinRB4&ab_channel=freeCodeCamp.org)
