# Project: AI-ZeroConnect4Bot

An implementation of the AlphaZero algorithm for learning Connect4. This project aims to test the implementation of AlphaZero on a simpler game before applying it to more complex games like Chess.

## Features

- **Monte Carlo Tree Search (MCTS)**: Integrated MCTS for decision making during self-play.
- **Iterative Self-Play and Training**: The model improves through cycles of self-play and training iterations.
- **Batching of Self-Play Games**: Efficiently runs multiple games in parallel to accelerate data generation.
- **Multi-Node Cluster Setup**: Supports training and self-play across multiple GPUs for scalability.
- **Neural Network Model**: Utilizes a 10-layer deep Residual Neural Network (ResNet) for function approximation.
- **Dirichlet Noise**: Adds exploration noise to the prior probabilities to encourage exploration of the action space.
- **Temperature Parameter**: Controls exploration by adjusting the temperature parameter for the first 30 moves.
- **Symmetric Variations for Data Augmentation**: Employs board symmetry to augment training data.
- **Neural Network Evaluation Caching**: Caches evaluations to avoid redundant computations of board positions.
- **Torch Compilation**: Compiles the PyTorch model and certain functions for faster execution.
- **Optimized Datatype**: Uses bfloat16 for faster computation and reduced memory usage.
- **Zobrist Hashing**: Implements Zobrist hashing to efficiently detect duplicate board positions.
- **Deduplication of Positions in Evaluations and Training Data**: Avoids duplicate computations and data by averaging priors and result values for identical positions.
- **1-Cycle Learning Rate Policy**: Adjusts the learning rate cyclically to facilitate efficient training.
- **Adaptive Sampling Window**: Gradually increases the sampling window to phase out early data and stabilize training.

## Future Work

- **Optimized Inference**: Implement quantization to reduce the model size and improve inference speed as well as using TensorRT for optimized inference.
- **Evaluation of Model Performance**: Evaluate the model against a baseline agent to assess its performance. Evaluate how much of the model's strength comes from the neural network and how much from the MCTS. Do so by comparing the model's performance with 1 NN evaluation per move and 800 NN evaluations per move.

## Installation

```bash
# Clone the repository
git clone https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git

# Navigate to the project directory
cd Advanced-Techniques-in-Chess-Engines

# Install required dependencies
pip install -r requirements.txt
pip install -r AIZeroConnect4Bot/requirements.txt
```

## Usage

```bash
# Train the model
python -m AIZeroConnect4Bot.train

# Evaluate the model
python -m AIZeroConnect4Bot.eval
```

## Implementation Details

### Monte Carlo Tree Search (MCTS)

Uses MCTS for exploring possible moves during self-play, balancing exploration and exploitation to improve policy and value estimates.

### Neural Network Architecture

- **Input**: Board state represented as a tensor.
- **Architecture**: 10-layer ResNet to approximate the policy and value function.
- **Output**: Policy vector over possible moves and a scalar value estimating the probability of winning.

### Training Process

- **Self-Play**: The current model plays against itself to generate training data.
- **Data Augmentation**: Applies symmetric transformations to the board to augment the dataset.
- **Caching and Deduplication**: Caches neural network evaluations and deduplicates positions to optimize performance.
- **Learning Rate Schedule**: Implements a 1-cycle learning rate policy for efficient convergence.
- **Sampling Window**: Adaptively adjusts the sampling window size to improve training stability.

## References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815)
- [Lessons from AlphaZero: Connect Four](https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68)
- [Lessons from AlphaZero Part 3: Parameter Tweaking](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5)
- [Lessons from AlphaZero Part 4: Improving the Training Target](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)
- [Lessons from AlphaZero Part 5: Performance Optimization](https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e)
- [Lessons from AlphaZero Part 6: Hyperparameter Tuning](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)
- [AlphaZero Explained](https://www.youtube.com/watch?v=wuSQpLinRB4)

## Results

Results will be added here after completion of training and evaluation.
