import numpy as np
from numpy.typing import NDArray

from Framework import *


def board_result_to_score(board: Board) -> float:
    """
    Converts the result of a chess board into a score.

    The result of a chess board is a string representing the result of the game.
    The score is either 1.0 if the result is a checkmate, or 0.0 otherwise (draw).

    :param board: The chess board to convert the result to a score.
    :return: The score of the result.
    """
    if board.is_checkmate():
        return 1.0

    return 0.0


ENCODING_CHANNELS = 6 + 6


def encode_board(board: Board) -> NDArray[np.float32]:
    """
    Encodes a chess board into a 12x8x8 numpy array.

    Each layer in the first dimension represents one of the 12 distinct
    piece types (6 for each color). Each cell in the 8x8 board for each layer
    is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.

    :param board: The chess board to encode.
    :return: A 12x8x8 numpy array representing the encoded board.
    """

    # TODO ensure, that this is turn agnostic, I.e. that the board is encoded in such a way that the first 6 layers contain the pieces of the current player and the second 6 layers contain the pieces of the other player

    encoded_board = np.zeros((12, 8, 8), dtype=np.float32)
    pieces = [board.pieces(piece_type, color) for color in COLORS for piece_type in PIECE_TYPES]

    for index, bitboard in enumerate(pieces):
        squares = np.array(list(bitboard))
        rows, cols = np.divmod(squares, 8)
        encoded_board[index, rows, cols] = 1

    # If board.turn is BLACK, mirror the encoding
    if board.turn == BLACK:
        encoded_board = np.flip(encoded_board, axis=2)  # Flip columns for mirror

    return encoded_board


def encode_boards(boards: list[Board]) -> NDArray[np.float32]:
    """
    Encodes a list of chess boards into a Nx12x8x8 numpy array.

    Each layer in the first dimension represents one of the 12 distinct
    piece types (6 for each color). Each cell in the 8x8 board for each layer
    is 1 if a piece of the layer's type is present at that cell, and 0 otherwise.

    :param boards: The chess boards to encode.
    :return: A Nx12x8x8 numpy array representing the encoded boards.
    """
    return np.stack([encode_board(board) for board in boards])
