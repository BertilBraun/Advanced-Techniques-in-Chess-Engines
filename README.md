# AlphaZero-Clone: Deep Reinforcement Learning for Board Games

Welcome to the **AlphaZero-Clone** repository! This project replicates the deep reinforcement learning approaches pioneered by AlphaZero, focusing initially on Chess and extending to other classic board games. The goal is to create an optimized, scalable, and maintainable framework for training AI agents capable of mastering complex games through self-play and iterative learning. This implementation is **optimized for distributed medium-to-large scale training** on multi-node multi-GPU clusters.

## Table of Contents

- [AlphaZero-Clone: Deep Reinforcement Learning for Board Games](#alphazero-clone-deep-reinforcement-learning-for-board-games)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
    - [Training Pipeline](#training-pipeline)
  - [Optimizations](#optimizations)
  - [Implementation Details](#implementation-details)
    - [Supported Games](#supported-games)
  - [Future Work](#future-work)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
  - [Contributing](#contributing)
  - [References](#references)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Project Overview

**AlphaZero-Clone** is a Python-based project inspired by the AlphaZero algorithm, aiming to replicate its success in mastering games like Go, Chess, and Shogi. Starting with simpler implementations of Tic-Tac-Toe, Connect Four, and Checkers, the project incrementally scales up to Chess, ensuring correctness and optimization at each step.

Key objectives include:

- Implementing self-play mechanisms using Monte Carlo Tree Search (MCTS).
- Facilitating iterative training with continuous model updates.
- Optimizing performance through parallel processing and efficient algorithms.
- Maintaining code readability and modularity for ease of understanding and extension.
- Supporting multiple games with varying complexities and rule sets.
- Scalability for distributed training across multiple nodes and GPUs.

### Training Pipeline

The training pipeline follows these steps:

1. **Self-Play Generation:**
   - Multiple workers generate self-play games using the current neural network model.
   - MCTS guides the move selection during self-play.

2. **Data Collection:**
   - Game states, move probabilities, and game outcomes are stored for training.

3. **Training:**
   - The neural network is trained on the collected data to predict move probabilities (policy) and game outcomes (value).
   - Training is performed in parallel across multiple nodes to accelerate learning.

4. **Model Evaluation:**
   - Periodically, new models are evaluated against previous versions to ensure improvement.
   - Best-performing models are promoted for continued training and self-play.

## Optimizations

Several optimizations have been implemented to enhance performance and scalability and to scale the training process across multiple nodes and GPUs. Refer to the following sections for more details:

- [Inference Optimization](documentation/optimizations/inference.md)
- [Inference Architecture Optimization](documentation/optimizations/architecture.md)
- [Training Optimization](documentation/optimizations/training.md)
- [Evaluation](documentation/optimizations/evaluation.md)
- [Hyperparameter Optimization](documentation/optimizations/hyperparameters.md)

## Implementation Details

For a detailed overview of the project's implementation, refer to the following section: [Implementation Details](documentation/implementation.md).

The main implementation is currently written in Python and can be found in the `py` directory. The main 3 scripts are:

- `train.py`: The main training script that orchestrates the self-play, training, and evaluation processes.
- `eval.py`: A script to play against the trained model or evaluate the model against itself.
- `opt.py`: A script to optimize the hyperparameters and architecture of the model.

A rewrite in C++ is planned for at least the Self-Play part to improve performance and scalability. The work in progress can be found in the `cpp` directory.

### Supported Games

- **Tic-Tac-Toe:** Basic implementation for initial testing and verification.
- **Connect Four:** Intermediate complexity to ensure correct game state handling.
- **Checkers:** Transitioning to more complex game mechanics and strategies.
- **Chess:** The primary focus for deploying the AlphaZero-like algorithms.

## Future Work

For a list of planned enhancements and future work, refer to the following section: [Future Work](documentation/future.md).

## Getting Started

To get started with the Python implementation, follow the steps below:

### Prerequisites

Ensure you have the following installed:

- **Python 3.11+**
- **PyTorch 2.0+** (or another compatible deep learning framework)
- **Dependencies:** Listed in `requirements.txt`

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/BertilBraun/Advanced-Techniques-in-Chess-Engines.git
   cd Advanced-Techniques-in-Chess-Engines
   cd py # Navigate to the Python implementation
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Configure the Environment:**

   Edit the `src/settings.py` file to set parameters such as the game, search and train settings.

2. **Start Self-Play and Training:**

   ```bash
   python train.py
   ```

## Contributing

Contributions are welcome! Whether you're fixing bugs, improving documentation, or adding new features, your help is appreciated.

## References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815)
- [Minigo: A Case Study in Reproducing Reinforcement Learning Research](https://openreview.net/pdf?id=H1eerhIpLV)
- [Lessons from AlphaZero: Connect Four](https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68)
- [Lessons from AlphaZero Part 3: Parameter Tweaking](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5)
- [Lessons from AlphaZero Part 4: Improving the Training Target](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)
- [Lessons from AlphaZero Part 5: Performance Optimization](https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e)
- [Lessons from AlphaZero Part 6: Hyperparameter Tuning](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)
- [AlphaZero Explained](https://www.youtube.com/watch?v=wuSQpLinRB4)

## License

This project is licensed under the [MIT License](./LICENSE).

## Acknowledgements

- Inspired by [DeepMind's AlphaZero](https://deepmind.com/research/case-studies/alphazero-the-story-so-far)
- Utilizes [PyTorch](https://pytorch.org/) for deep learning implementations
- Contributions from the open-source community

---

Stay tuned for more updates! If you have any questions or suggestions, feel free to open an issue or contact the maintainer.
