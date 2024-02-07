

import chess
import chess.pgn
import chess.svg
import numpy as np
from numba import njit


def board_to_bitfields(board: chess.Board, turn: chess.Color) -> np.ndarray:

    pieces_array = []
    colors = [chess.WHITE, chess.BLACK]
    for c in colors if turn == chess.WHITE else colors[::-1]:
        for p in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            pieces_array.append(board.pieces_mask(p, c))

    return np.array(pieces_array, copy=False).astype(np.int64)


@njit(cache=True)
def bitfield_to_nums(bitfield: np.int64, white: bool) -> np.ndarray:
    """ board_array = np.unpackbits(
        np.array([bitfield], dtype=np.int64).view(np.uint8)
    ).flatten().astype(np.float32)

    return board_array if white else np.negative(board_array) """

    board_array = np.zeros(64, dtype=np.float32)

    for i in np.arange(64).astype(np.int64):
        if bitfield & (1 << i):
            board_array[i] = 1. if white else -1.

    return board_array


def bitfields_to_nums(bitfields: np.ndarray) -> np.ndarray:
    boards = [None] * 12

    for i, bitfield in enumerate(bitfields):
        boards[i] = bitfield_to_nums(bitfield, i < 6)

    return np.array(boards, dtype=np.float32, copy=False)


def board_to_nums(board: chess.Board, turn: chess.Color) -> np.ndarray:

    return bitfields_to_nums(board_to_bitfields(board, turn))
