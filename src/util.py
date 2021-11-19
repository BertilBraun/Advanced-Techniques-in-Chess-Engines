from typing import TextIO, Tuple
import chess
import numpy as np
from tensorflow.python.keras.callbacks import History


def board_to_bitfields(board: chess.Board) -> np.ndarray:

    pieces_array = []
    for c in (chess.WHITE, chess.BLACK):
        for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            pieces_array.append(board.pieces_mask(p, c))

    return np.array(pieces_array).astype(np.int64)


def bitfield_to_nums(bitfield: np.int64) -> np.ndarray:

    board_array = np.zeros(64)

    for i in range(64):
        board_array[i] = 1 if bitfield & (1 << i) != 0 else 0

    return board_array.astype(np.float32)


def bitfields_to_nums(bitfields: np.ndarray) -> np.ndarray:
    bitfields = bitfields.astype(np.int64)

    return np.array(bitfields.map(bitfield_to_nums))


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


def create_training_data(dataset) -> Tuple[np.ndarray, np.ndarray]:
    y = dataset[12].values
    X = dataset.drop(12, axis=1)

    def transform(row):
        return list(np.concatenate([bitfield_to_nums(e) for e in row]))
    X = X.apply(transform, axis=1, result_type='expand')

    # move into range of 0 - 1
    y = np.vectorize(lambda v: v / 250)(y)
    print(min(y), max(y))

    return X, y


def plot_history(history: History, index: int):
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'../training/loss{index:04d}.png')

    plt.clf()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f'../training/accuracy{index:04d}.png')
