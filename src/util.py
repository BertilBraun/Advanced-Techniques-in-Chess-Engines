import os
from typing import List, TextIO, Tuple
import chess
import numpy as np
from pandas.core.frame import DataFrame
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import History


def board_to_bitfields(board: chess.Board) -> np.ndarray:

    pieces_array = []
    for c in (chess.WHITE, chess.BLACK):
        for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            pieces_array.append(board.pieces_mask(p, c))

    return np.array(pieces_array).astype(np.int64)


def bitfield_to_nums(bitfield: np.int64, white: bool) -> np.ndarray:

    board_array = np.zeros(64)

    for i in range(64):
        board_array[i] = (1 if white else -
                          1) if bitfield & (1 << i) != 0 else 0

    return board_array.astype(np.float32)


def bitfields_to_nums(bitfields: np.ndarray) -> np.ndarray:
    bitfields = bitfields.astype(np.int64)

    boards = []

    for i, bitfield in enumerate(bitfields):
        boards.append(bitfield_to_nums(bitfield, i < 8))

    return np.array(boards)


def board_to_nums(board: chess.Board) -> np.ndarray:

    return bitfields_to_nums(board_to_bitfields(board))


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

    # drop(dataset[abs(dataset[12] / 10.) > 15].index, fract=0.80)
    drop(dataset[abs(dataset[12] / 10.) < 0.1].index, fract=0.90)
    drop(dataset[abs(dataset[12] / 10.) < 0.15].index, fract=0.10)

    y = dataset[12].values
    X = dataset.drop(12, axis=1)

    def transform(row):
        return list(np.concatenate([bitfield_to_nums(e, i < 8) for i, e in enumerate(row)]))
    X = X.apply(transform, axis=1, result_type='expand')
    X = X.astype(np.float32)

    # move into range of 0 to 1
    y = y.astype(np.float32)
    y = sigmoid(y / 10.)
    print(min(y), max(y))

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
    plt.savefig(f'../training/{type}{index}.png')

    plt.clf()


def load_last_training_weights_file(model: Sequential) -> None:

    # get the last filename in the sorted directory "../training"
    last_files = sorted([
        f for f in os.listdir("../training") if f.endswith('.h5')
    ])

    if len(last_files) > 0:
        model.load_weights(f"../training/{last_files[-1]}")
