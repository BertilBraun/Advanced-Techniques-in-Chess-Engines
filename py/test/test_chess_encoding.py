import chess
import numpy as np

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.ChessGame import ChessGame


KNIGHT_CYCLE = ('g1f3', 'g8f6', 'f3g1', 'f6g8')


def test_encoding_contains_en_passant_and_halfmove_clock() -> None:
    board = ChessBoard()
    board.board.push_uci('e2e4')

    encoded = ChessGame().get_canonical_board(board)

    assert encoded.shape == (29, 8, 8)
    assert np.count_nonzero(encoded[19]) == 1
    assert encoded[19].reshape(-1)[chess.E6] == 1
    assert np.all(encoded[28] == 0)


def test_encoding_distinguishes_second_and_third_occurrences() -> None:
    board = ChessBoard()
    for move_uci in KNIGHT_CYCLE:
        board.board.push_uci(move_uci)

    second_occurrence = ChessGame().get_canonical_board(board)
    assert np.all(second_occurrence[20] == 1)
    assert np.all(second_occurrence[21] == 0)

    for move_uci in KNIGHT_CYCLE:
        board.board.push_uci(move_uci)

    third_occurrence = ChessGame().get_canonical_board(board)
    assert np.all(third_occurrence[20] == 1)
    assert np.all(third_occurrence[21] == 1)
