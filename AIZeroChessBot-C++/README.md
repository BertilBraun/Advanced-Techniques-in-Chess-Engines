# Project: AI-ZeroChessBot-C++

## Current State

The current state of the project is that everything is reimplemented in C++ including the base Chess Framework, the MCTS algorithm, and the Neural Network. The translation of the Chess Framework lead to 50-100x speedup. The project is currently deployed on the cluster and should be training there using 1 Tesla H100 for now.

### Upcoming To Dos

- [x] Profile the performance of the C++ implementation of the MCTS algorithm and the Neural Network compared to the Python implementation. Is the C++ implementation more than 10% in the neural network evaluation step?
- [x] Create a interactive Notebook to evaluate the performance of the bot on the cluster with access to the cluster's GPU.
  - Evaluate the performance of the bot after the first 12 hours of training.
  - Evaluate the performance of the bot against a baseline chess bot (continuously).
- [ ] Continue training the bot on the cluster with multiple GPUs and multiple nodes (as already implemented) for multiple days if the performance after 12 hours is promising.

### **Current Model Problems:**

We currently have a problem, that the self-play games at the beginning are not very good. This is because the model is not trained yet and therefore predicts bad moves as well as bad evaluations. This means, that many of the expanded nodes in the MCTS are evaluated by the model instead of the endgame score. This means, that we are training the model with random data, which is not very useful. AlphaZero solves this problem by simply searching more iterations per move, which more often leads to the endgame score being used. However, this is not viable for us, because we are using way less computational resources than AlphaZero.

My idea to overcome this problem is to use grandmaster games and stockfish evaluations to generate the training data for the first few iterations. This way, we can train the model with good data from the beginning and therefore improve the self-play games. This should lead to a better model and therefore better self-play games. After a few  iterations, the model should be good enough to generate good training data by itself, so that the self improvement loop can start.

- Grandmaster games can be found here: [https://database.nikonoel.fr/](https://database.nikonoel.fr/)
- Lichess evaluations can be found here: [https://database.lichess.org/#evals](https://database.lichess.org/#evals)
- Stockfish can be found here: [https://stockfishchess.org/download/](https://stockfishchess.org/download/)

(We are using Stockfish 8 on the cluster, because it is the only version that compiles there)

- The model seems to have collapsed on the value head. The value head is always predicting `-0.1435`. This is a problem that needs to be addressed. For now, I restarted training with a new model and a new optimizer and added a way larger search space for the best move in the self-play algorithm.
- There is little to no coherence between the policy of our model and stockfish's evaluation of the board state. The hope is that this will improve with more training.

The hope is, that the teacher like initialization for the model, will also counteract the problem of the collapsed value head as the model is initially trained on good data with diverse evaluations.

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
- **Parallel Computing**: MPI (mpi4py)

## Parallel Computing

The project is currently deployed on the cluster and should have access to multiple GPUs and multiple nodes. The project is implemented using MPI for parallel computing. The training data is generated using self-play and is distributed across the nodes. We have one master node and multiple worker nodes. The master node is responsible for training the neural network and the worker nodes are responsible for generating the training data.

### Training Time Estimation Formula

To achieve a balance where $T_{gen} = T_{train}$, we'll need to determine the number of workers required for sample generation and training processes so that their times are equal. This balance ensures that the time spent generating samples through self-play is equal to the time spent training on those samples, optimizing resource utilization.

Given:

- $T_{gen}$ is the time to generate one sample.
- $W_{gen}$ is the number of workers dedicated to sample generation.
- $T_{batch}$ is the time to process one training batch.
- $E$ is the number of epochs.
- $D$ is the desired dataset size.
- $B$ is the batch size.

### Definitions for Parallel Processing

For parallel processing, the effective time to generate the dataset and the time to train on it are influenced by the number of workers in each process:

1. **Effective Time for Sample Generation with $W_{gen}$ Workers**:

    The total time to generate $D$ samples with $W_{gen}$ workers is:
    $$T_{total\_gen} = \frac{D \times T_{gen}}{W_{gen}}$$

2. **Time for Training**:

    $$T_{total\_train} = E \times \frac{D}{B} \times T_{batch}$$

### Balancing $T_{gen}$ and $T_{train}$

To balance $T_{gen}$ and $T_{train}$, we set $T_{total\_gen} = T_{total\_train}$ and solve for $W_{gen}$, the number of workers needed for sample generation:

$$\frac{D \times T_{gen}}{W_{gen}} = E \times \frac{D}{B} \times T_{batch}$$

Solving for $W_{gen}$:

$$W_{gen} = \frac{D \times T_{gen}}{E \times \frac{D}{B} \times T_{batch}} = \frac{T_{gen} \times B}{E \times T_{batch}}$$

This formula gives you the number of workers for sample generation needed to match the training time, assuming optimal parallelization and no significant overhead for increasing workers.

### Considerations

- **Parallelization Efficiency**: In practice, the efficiency of adding more workers may decrease due to overhead and communication costs. The actual number of workers needed could differ.

### Example Calculation

Our current setup has the following parameters:

- `1` Worker with `800` Iterations per Move with `64` Games in parallel and a Batch size of `64` generated `9806 samples` in `80.4` min
- Training took `13` min for `654912` Samples

With these parameters, we can calculate the number of workers needed for sample generation to balance the generation and training times:

- $T_{gen} = 0.492$ seconds
- $T_{batch} = 0.0019$ second
- $E = 20$ epochs
- $B = 64$ batch size

Find $W_{gen}$:

$$W_{gen} = \frac{0.492sec \times 64}{40 \times 0.0019sec} = \frac{31.488}{0.076} = 414.32$$

Rounding up, you would need about 415 workers dedicated to sample generation to balance the generation and training times under these conditions.

## Getting Started

To run your `AIZeroChessBot-C++` project after setting it up with CMake and compiling it with Visual Studio Code (VSCode) on Windows, follow these steps. This guide assumes you've already configured your `CMakeLists.txt` and have the necessary build scripts in place.

### Step 1: Build the Project

1. **Open VSCode** and navigate to your project folder (`AIZeroChessBot-C++`).

2. **Run the Setup Script** (if you haven't already) to download LibTorch and generate the CMake build system. In the VSCode terminal, navigate to your project's root directory and run:

    ```cmd
    .\setup_build.bat
    ```

   This script prepares the build environment, including downloading LibTorch if necessary and generating build files.

3. **Build the Project** using CMake.

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
    .\AIZeroChessBot.exe <root|worker> <rank> <num_workers>
    ```

   This command executes your program. Any command-line arguments required by your application should follow the executable name.

Alternatively, you can run the train setup script to start the training process:

```cmd
cd ../train
sbatch train_setup.sh
```

## Evaluation

For evaluating the performance of the AI-Zero Chess Bot, we have a jupyter notebook that can be used to interactively evaluate the bot's performance against a baseline chess bot and track its improvement over time. The notebook will also provide visualizations and metrics to assess the bot's learning progress. The notebook will be deployed on the cluster and will have access to the cluster's GPU for running the bot evaluations. In the notebook, we will be able to play against the bot and observe its moves and strategies in real-time.

To run the evaluation notebook, follow these steps:

1. **Connect to the Cluster**: Connect to the cluster using SSH and navigate to the project directory.
2. **Setup the eval Build**: The evaluation notebook requires a new compiled build with the evaluation mode enabled. Run the following commands to set up the evaluation build:

    ```bash
    cd train
    ./setup_eval_build.sh
    ```

    This will create a new build with the evaluation mode enabled.
3. **Open the Evaluation Notebook**: Open the `eval.ipynb` notebook in Jupyter.
4. **[Optional] Download Model Weights**: If you want to use pre-trained model weights for the evaluation, download the model weights from [here](TODO: Add link) and place them in the `train` directory.
5. **Run the Notebook**: Execute the cells in the notebook to start the evaluation process. The notebook will guide you through the evaluation steps and display the bot's performance metrics and visualizations.

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
