import multiprocessing
import os
from bz2 import BZ2File
from io import StringIO
from time import time
from typing import List, TextIO, Tuple

import chess
import chess.pgn
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Conv3D, Dense,
                                     Flatten, Rescaling, Reshape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import Callback, History, TensorBoard


def board_to_bitfields(board: chess.Board, turn: chess.Color) -> np.ndarray:

    pieces_array = []
    colors = [chess.WHITE, chess.BLACK]
    for c in colors if turn == chess.WHITE else colors[::-1]:
        for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            pieces_array.append(board.pieces_mask(p, c))

    return np.array(pieces_array).astype(np.int64)


def bitfield_to_nums(bitfield: np.int64, white: bool) -> np.ndarray:

    board_array = np.zeros(64).astype(np.float32)

    for i in np.arange(64).astype(np.int64):
        if bitfield & (1 << i):
            board_array[i] = 1. if white else -1.

    return board_array


def bitfields_to_nums(bitfields: np.ndarray) -> np.ndarray:
    bitfields = bitfields.astype(np.int64)

    boards = []

    for i, bitfield in enumerate(bitfields):
        boards.append(bitfield_to_nums(bitfield, i < 6))

    return np.array(boards).astype(np.float32)


def board_to_nums(board: chess.Board, turn: chess.Color) -> np.ndarray:

    return bitfields_to_nums(board_to_bitfields(board, turn))


def pager(in_file: TextIO, lines_per_page=20):
    assert lines_per_page > 1 and lines_per_page == int(lines_per_page)

    lin_ctr = 0
    current = ""
    for lin in in_file:
        lin_ctr += 1
        current += lin.decode("utf-8") + "\n"
        if lin_ctr % lines_per_page == 0:
            yield current
            current = ""
            if lin_ctr % (lines_per_page * 5000) == 0:
                print("Next:" + str(lin_ctr // lines_per_page))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_training_data(dataset: DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    def drop(indices, fract):
        drop_index = np.random.choice(
            indices,
            size=int(len(indices) * fract),
            replace=False)
        dataset.drop(drop_index, inplace=True)

    drop(dataset[abs(dataset[12] / 10.) > 30].index, fract=0.80)
    drop(dataset[abs(dataset[12] / 10.) < 0.1].index, fract=0.90)
    drop(dataset[abs(dataset[12] / 10.) < 0.15].index, fract=0.10)

    y = dataset[12].values
    X = dataset.drop(12, axis=1)

    def transform(row):
        return list(np.concatenate(bitfields_to_nums(row)))
    X = X.apply(transform, axis=1, result_type='expand')

    # move into range of -1 to 1
    y = y.astype(np.float32)
    y = np.tanh(y / 10.)
    # y = sigmoid(y / 10.)
    print(min(y), max(y))
    print(X.shape, y.shape)

    return X, y


def plot_history(history: History, index):
    plot(history.history['loss'], history.history['val_loss'], 'loss', index)
    # plot(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', index)


def plot(data: List, val_data: List, type: str, index):
    import matplotlib.pyplot as plt

    plt.plot(data)
    plt.plot(val_data)
    plt.title(f'model {type}')
    plt.ylabel(type)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'training/{type}{index}.png')

    plt.clf()


def load_last_training_weights_file(model: Sequential) -> None:

    # get the last filename in the sorted directory "training"
    last_files = sorted([
        f for f in os.listdir("training") if f.endswith('.h5')
    ])

    if len(last_files) > 0:
        model.load_weights(f"training/{last_files[-1]}")


out_files = {}


def create_dataset(game: chess.pgn.Game, out: TextIO) -> None:
    state = game.end()
    while state:
        evaluation = state.eval()

        if evaluation is not None:
            nums = board_to_bitfields(state.board(), state.turn())
            evaluation = evaluation.relative.score(mate_score=10) / 100
            evaluation = evaluation if state.turn() == chess.WHITE else -evaluation

            out.write(','.join(map(str, nums)) + ',' + str(evaluation) + '\n')

        state = state.parent


def process_game(lines: str, out_path: str) -> None:
    game = chess.pgn.read_game(StringIO(lines))
    if game is None:
        return

    cp = multiprocessing.current_process()
    if cp.pid not in out_files:
        out_files[cp.pid] = open(
            f'{out_path[:-4]}.{cp.pid}{out_path[-4:]}', 'w', buffering=1024*1024)

    # if "WhiteElo" in game.headers and "BlackElo" in game.headers and \
    #         int(game.headers["WhiteElo"]) > 2200 and int(game.headers["BlackElo"]) > 2200:
    #     create_dataset(game, out_files[cp.pid])
    create_dataset(game, out_files[cp.pid])


def preprocess(in_path: str, out_path: str) -> None:

    with BZ2File(in_path, "rb") as in_file:
        with multiprocessing.Pool(os.cpu_count()-1) as pool:
            for i, lines in enumerate(pager(in_file)):
                pool.apply_async(process_game, args=(str(lines), out_path))


def unite(dir: str, out: str) -> None:
    """
    Unites all files in a directory into a single file.
    """
    with open(out, 'w') as outfile:
        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename)) as inFile:
                outfile.write(inFile.read())


class Plotter(Callback):
    batch_loss = []  # loss at given batch

    def __init__(self, batches):
        super(Plotter, self).__init__()
        self.batches = batches
        self.current_batch = 0

    def on_train_batch_end(self, batch, logs=None):
        self.current_batch += 1

        Plotter.batch_loss.append(logs.get('loss'))

        if self.current_batch % self.batches == 0:
            plot(Plotter.batch_loss, Plotter.batch_loss, 'loss', '')


def gen_model() -> Sequential:
    """ model = Sequential()
    model.add(Dense(2048, input_shape=(12 * 8 * 8,), activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.001),
        # metrics=['accuracy', 'mse']
    )

    return model """

    model = Sequential()
    model.add(Reshape((12, 8, 8, 1), input_shape=(12 * 64,)))
    model.add(Conv3D(64, (12, 3, 3), activation='relu', padding='same'))

    for _ in range(15):
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(BatchNormalization())

    """model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())"""
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3)))
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
    return model


def train(model: Sequential, X, y, index: int):
    model.optimizer.learning_rate = 0.0005 / ((index + 1) * 2)

    history: History = model.fit(
        X,
        y,
        epochs=25,
        batch_size=64,
        validation_split=0.1,
        callbacks=[
            ModelCheckpoint('training/' + f'{index:03d}' + 'weights{epoch:08d}.h5',
                            save_weights_only=True, save_freq='epoch'),
            Plotter(batches=100),
            # access via tensorboard --logdir training/logs
            TensorBoard(log_dir=f'training/logs/{time()}.log')
        ]
    )

    plot_history(history, index)


def test_model():
    model = gen_model()
    load_last_training_weights_file(model)

    # test the model

    for chunk in pd.read_csv("dataset/nm_games.csv", header=None, chunksize=200):
        X, y = create_training_data(chunk)

        predictions = model.predict(X)

        # print the results and evaluate the error
        for v, p in zip(y, predictions):
            print(f"actual: {v} prediction: {p} - loss: {abs(v - p)}")

        break


def prep():
    try:
        preprocess("dataset/lichess_db_standard_rated_2021-10.pgn.bz2",
                   "dataset/processed_games/nm_games.csv")
    except KeyboardInterrupt:
        pass
    finally:
        for out_file in out_files.values():
            out_file.close()


def combine():
    unite("dataset/processed_games", "dataset/nm_games.csv")


def learn():
    model = gen_model()
    model.summary()
    # load_last_training_weights_file(model)

    for i, chunk in enumerate(pd.read_csv("dataset/nm_games.csv", header=None, chunksize=50000)):
        X, y = create_training_data(chunk)
        train(model, X, y, i)

        model.save(f'training/model{i:03d}.h5')


def main():
    prep()
    combine()
    learn()
    test_model()


if __name__ == "__main__":
    main()
