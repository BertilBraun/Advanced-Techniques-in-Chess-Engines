# Chess AI

The idea of this project is to create a chess AI that can play chess against a human player.
This will be done by using a neural network to learn how to evaluate the current board state and genereate the next move using [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search).

# Implementation

## Dataset

The dataset is a collection of chess games. The games are fetched from the official [lichess](https://database.lichess.org/) database.
Only games which have a evaluation score, which the NN is supposed to learn, are included.
This reduces the size of the dataset to about 11% of the original.
The dataset is normalized to the range [-1, 1] using a scaled `tanh` function.
Since many evaluations are very close to 0 and -1/1, a percentage of these are dropped aswell, to reduce the variation in the dataset.

## Data representation

Each move is a datapoint to train the network on.
The input for the NN are 12 arrays of size 8x8 (so called (Bitboards)[https://en.wikipedia.org/wiki/Bitboard]), one for each of the 6x2 different pieces and colors.
The output of the NN is a single scalar value between -1 and 1, representing the evaluation of the board state. -1 meaning the board is heavily favored for the black player, 1 meaning the board is heavily favored for the white player.

## The neural network

The neural network is implemented using [Keras](https://keras.io/).

- ### First iteration

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

- ### Second iteration

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

## Training

The training is done using [Keras](https://keras.io/).
Multiple sets of 50k-500k moves are used to train the network.
The network is trained for 20 epochs on each move set with a batchsize of 64 and 10% of moves are used for validation.

Afterwards the learning rate is adjusted by `0.001 / (index + 1)`.

A [checkpoint](https://keras.io/callbacks/#checkpoint-callback) is used to save the model after each epoch, allowing manual testing of the model.

## Evaluation

  To be determined, once the network is trained.

## Play

The searchspace of the possible moves for the computer is explored using [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search). The NN is thereby fed the current board state and depending on the evaluation provided by the NN, the next move is chosen.

# Issues

The NN currently does not learn anything. It converges within a few epochs to a average evaluation of the dataset and does not predict anything depending on the board state.

# Installation

Run `pip install -r requirements.txt` to install the dependencies of the project.

The file `preprocessor.py` is used to extract the moves from the lichess database and write them in proper format to a `.csv` file.

The file `training.py` is used to start the training of the NN.

The file `testing.py` is used to test the current state of the NN on some parts of the dataset.

The file `play.py` will at some point in the future hopefully allow you to play against the NN.


# References

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

