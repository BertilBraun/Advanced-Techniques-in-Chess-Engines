import multiprocessing
import os
from bz2 import BZ2File
from io import StringIO
from time import time
from typing import TextIO

import chess
import chess.pgn
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Conv3D, Dense,
                                     Flatten, Reshape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import Callback, History, TensorBoard

from util import *

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

    try:
        with BZ2File(in_path, "rb") as in_file:
            with multiprocessing.Pool(os.cpu_count()-1) as pool:
                for i, lines in enumerate(pager(in_file)):
                    pool.apply_async(process_game, args=(str(lines), out_path))
    except KeyboardInterrupt:
        pass
    finally:
        for out_file in out_files.values():
            out_file.close()


def unite(dir: str, out: str) -> None:
    """
    Unites all files in a directory into a single file.
    """
    with open(out, 'w') as outfile:
        for filename in os.listdir(dir):
            with open(os.path.join(dir, filename)) as inFile:
                outfile.write(inFile.read())


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


def learn():
    model = gen_model()
    model.summary()
    # load_last_training_weights_file(model)

    for i, chunk in enumerate(pd.read_csv("dataset/nm_games.csv", header=None, chunksize=50000)):
        X, y = create_training_data(chunk)
        train(model, X, y, i)

        model.save(f'training/model{i:03d}.h5')


def main():
    preprocess("dataset/lichess_db_standard_rated_2021-10.pgn.bz2",
               "dataset/processed_games/nm_games.csv")
    unite("dataset/processed_games", "dataset/nm_games.csv")
    learn()
    test_model()


if __name__ == "__main__":
    main()
