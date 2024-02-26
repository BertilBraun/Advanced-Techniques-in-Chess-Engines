# Project: AINeural Net Chess Bot

## Overview

AINeural Net Chess Bot  is a sophisticated chess bot project that aims to implement a blend of advanced chess programming techniques with the power of neural network-based evaluation. This project builds on the foundation of a basic chess bot by incorporating neural network (NN) evaluators to replace and enhance traditional evaluation methods where possible. The goal is to leverage the depth of machine learning to understand and evaluate chess positions dynamically, improving the bot's strategic and tactical decision-making capabilities.

## Goals

- To integrate advanced chess bot features such as move ordering, quiescence search, and iterative deepening with a neural network-based evaluation function.
- To design and train a neural network capable of understanding complex chess positions, thereby enhancing the traditional handcrafted evaluation components like material count, piece-square tables, and pawn structure analysis.
- To evaluate the performance impact of replacing traditional evaluation metrics with a neural network, focusing on the bot's playing strength, decision-making process, and computational efficiency.

## Architecture

The project architecture is divided into the Chess Bot Framework and the Neural Network Evaluator, with a focus on how the NN evaluator integrates and enhances the bot's capabilities.

### Chess Bot Framework

- **Advanced Features**:
  - **Move Ordering**: Prioritization of moves to improve search efficiency, initially guided by simple heuristics and enhanced by NN insights.
  - **Quiescence Search**: Extends search at quiet positions to avoid the horizon effect.
  - **Iterative Deepening**: Dynamic adjustment of search depth for optimal time management.
  - **Transposition Table**: Storing previously evaluated positions to avoid redundant calculations.

### Neural Network Evaluator

Two different neural network architectures are considered for evaluation:

#### Shallow Neural Network Architecture

A simple neural network architecture with at most a single hidden layer, suitable for basic evaluation tasks. The reasoning here is, that the network should be able to capture simple patterns and relationships in the chess position, not much more complex than piece count and material balance at this stage. Also, the techniques of the handcrafted chess bot require a good evaluation function but also a very fast one. A shallow neural network should be able to provide a good evaluation in a very short time.

The architecture of the shallow neural network is as follows:

- **Input Layer**: Encodes the board state into a format suitable for NN processing. This includes piece positions, player to move, castling rights, en passant possibilities, and possibly a simplified representation of the game phase.
- **Hidden Layer**: A single layer with a small number of neurons to capture simple patterns and relationships.
- **Output**: A single value representing the board's evaluation from the current player's perspective, normalized to a range (e.g., -1 to 1).

#### Deep Neural Network Architecture

A more complex neural network architecture with multiple hidden layers, suitable for capturing complex spatial relationships and high-level strategic patterns. The deep neural network is expected to provide a more nuanced evaluation of the chess position, potentially outperforming the handcrafted evaluation functions in terms of strategic understanding. However, the trade-off is the computational cost, as deeper networks require more time to evaluate positions, which leads to less evaluated positions in the search tree.

The architecture of the deep neural network is as follows:

- **Input Layer**: Encodes the board state into a format suitable for NN processing. This includes piece positions, player to move, castling rights, en passant possibilities, and possibly a simplified representation of the game phase.
- **Convolutional Neural Network (CNN)** layers at the beginning to process spatial patterns on the chessboard. These layers can identify piece configurations, control of key squares, and potential threats.
- **Fully Connected (FC)** layers following the CNN layers to integrate the spatial information and learn higher-level strategies, evaluating the position globally.
- **Activation Functions**: Use ReLU (Rectified Linear Unit) for hidden layers to introduce non-linearity, allowing the network to learn complex patterns.
- **Output Layer**: A single neuron with a tanh activation function to output a value between -1 and 1, indicating the evaluated score of the position for the current player.

## Implementation

### Dataset

The dataset is a collection of chess games. The games are fetched from the official [lichess](https://database.lichess.org/) database.
Only games which have a evaluation score, which the NN is supposed to learn, are included.
This reduces the size of the dataset to about 11% of the original.
The dataset is normalized to the range [-1, 1] using a scaled `tanh` function.
Since many evaluations are very close to 0 and -1/1, a percentage of these are dropped aswell, to reduce the variation in the dataset.

### Data representation

Each move is a datapoint to train the network on.
The input for the NN are 12 arrays of size 8x8 (so called [Bitboards](https://en.wikipedia.org/wiki/Bitboard)), one for each of the 6x2 different pieces and colors.
The output of the NN is a single scalar value between -1 and 1, representing the evaluation of the board state. -1 meaning the board is heavily favored for the black player, 1 meaning the board is heavily favored for the white player.

### The neural network

The neural network is implemented using [Keras](https://keras.io/).

- #### First iteration

    The network consists of a convolutional neural network (CNN).
    The CNN is used to extract features from the board, then passed to a dense network to reduce to an evaluation.

    The CNN is implemented as follows:

    ```python
    model = Sequential()
    model.add(Reshape((12, 8, 8, 1), input_shape=(12 * 64,)))
    model.add(Conv2D(256, (3, 3), activation='relu',
                        padding='same', input_shape=(12, 8, 8, 1)))

    for _ in range(10):
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    # model.add(Rescaling(scale=1 / 10., offset=0))
    model.add(Dense(units=1, activation='tanh'))
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.01),
        # metrics=['accuracy', 'mse']
    )
    ```

- #### Second iteration

    Since the CNN did not seem to converge, I replaced it with a pure dense network.

    This is implemented as follows:

    ```python
    model = Sequential()
    model.add(Dense(2048, input_shape=(12 * 8 * 8,), activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.001),
        # metrics=['accuracy', 'mse']
    )
    ```

### Training

The training is done using [Keras](https://keras.io/).
Multiple sets of 50k-500k moves are used to train the network.
The network is trained for 20 epochs on each move set with a batchsize of 64 and 10% of moves are used for validation.

Afterwards the learning rate is adjusted by `0.001 / (index + 1)`.

A [checkpoint](https://keras.io/callbacks/#checkpoint-callback) is used to save the model after each epoch, allowing manual testing of the model.

### Evaluation

  To be determined, once the network is trained.

### Play

The search space of the possible moves for the computer is explored using [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search). The NN is thereby fed the current board state and depending on the evaluation provided by the NN, the next move is chosen.

## Issues

The NN currently does not learn anything. It converges within a few epochs to a average evaluation of the dataset and does not predict anything depending on the board state.

## Ideas

- The NN is way too large. It should be reduced to a much smaller size, since evaluation must be very fast and the evaluation of the board state is not that complex.
- Output of the NN should be a linear layer, since the evaluation in the dataset is (probably) linear.
- Only implementing a NN to evaluate the board state, would not lead to a very good chess AI.
  - The move search would either be done with an algorithm like [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) or MinMax with [Alpha-Beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) ([ref](https://www.youtube.com/watch?v=ffzvhe97J4Q)).
  - Alternatively the move search could be done with a NN, which would be trained to predict the next move, given the current board state. This would be similar to [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero). This would require a totally different approach and a lot more data and a stronger model. Also the training would be much more complex and time intensive.

## Installation

Run `pip install -r requirements.txt` to install the dependencies of the project.

The file `preprocessor.py` is used to extract the moves from the lichess database and write them in proper format to a `.csv` file.

The file `training.py` is used to start the training of the NN. Change `GAME_FILE` to "../dataset/nm_games small.csv" if you have not previously extracted the moves from the lichess database.

The file `testing.py` is used to test the current state of the NN on some parts of the dataset.

The file `play.py` will at some point in the future hopefully allow you to play against the NN.

## Integration with Chess Bot

- The NN evaluator is called during the search process to evaluate positions, replacing traditional handcrafted evaluation functions.
- Move ordering may initially use simple heuristics but will be adjusted based on insights from the NN, especially after moves have been evaluated, to prioritize exploration of more promising branches.

## Development Plan

1. **Framework Implementation**: Use the groundwork layed out by the handcrafted chess bot.
2. **Neural Network Design and Training**:
   - Design the NN architecture suitable for chess position evaluation.
   - Train the NN using a dataset of high-quality games, focusing on learning a broad understanding of various positions.
3. **Integration and Testing**:
   - Integrate the NN evaluator with the chess bot framework.
   - Benchmark performance against versions with traditional evaluation functions.

## Technologies

- **Programming Language**: Python 3.x.
- **Machine Learning Frameworks**: TensorFlow or PyTorch.
- **Chess Library**: python-chess.

## Getting Started

- Setup instructions, training procedures, and how to run the bot will be detailed in the project documentation.

## References

- [NNBot](https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results/blob/main/Bots/Bot_529.cs) by Jamie Whiting: A simple chess bot using a neural network for evaluation.
- https://github.com/ryanp73/ChessAI
- https://www.chess.com/blog/the_real_greco/understanding-alphazero-a-basic-chess-neural-network
- https://towardsdatascience.com/creating-a-chess-ai-using-deep-learning-d5278ea7dcf
- https://erikbern.com/2014/11/29/deep-learning-for-chess
- https://arxiv.org/abs/1712.01815
- https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188
- https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a
- https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
- https://database.lichess.org/
- http://www.ficsgames.org/download.html
