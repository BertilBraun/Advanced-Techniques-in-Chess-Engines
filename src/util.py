import chess
import numpy as np


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
