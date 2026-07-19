from __future__ import annotations

import chess

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.repetition_history import REPETITION_HISTORY_PLIES, bounded_repetition_history


KNIGHT_CYCLE = (
    'g1f3',
    'g8f6',
    'f3g1',
    'f6g8',
)

EIGHT_PLY_CYCLE = (
    'g1f3',
    'g8f6',
    'b1c3',
    'b8c6',
    'f3g1',
    'f6g8',
    'c3b1',
    'c6b8',
)


def play_moves(board: chess.Board, moves_uci: tuple[str, ...]) -> None:
    for move_uci in moves_uci:
        board.push_uci(move_uci)


def test_bounded_repetition_history_reconstructs_current_position() -> None:
    board = chess.Board()
    moves = KNIGHT_CYCLE * 3
    play_moves(board, moves)

    history = bounded_repetition_history(board, maximum_plies=REPETITION_HISTORY_PLIES)
    reconstructed = chess.Board(history.starting_fen)
    play_moves(reconstructed, history.moves_uci)

    assert len(history.moves_uci) == 8
    assert reconstructed.fen() == board.fen()
    assert reconstructed.is_repetition(3)


def test_threefold_ends_after_third_occurrence_not_before_claimable_move() -> None:
    board = ChessBoard()
    play_moves(board.board, KNIGHT_CYCLE)
    play_moves(board.board, KNIGHT_CYCLE[:-1])

    assert board.board.can_claim_threefold_repetition()
    assert not board.board.is_repetition(3)
    assert not board.is_game_over()

    board.board.push_uci(KNIGHT_CYCLE[-1])

    assert board.board.is_repetition(3)
    assert board.is_game_over()
    assert board.check_winner() is None


def test_chess_board_copy_preserves_only_bounded_history() -> None:
    board = ChessBoard()
    play_moves(board.board, KNIGHT_CYCLE * 3)

    copied = board.copy()

    assert len(copied.board.move_stack) == REPETITION_HISTORY_PLIES
    assert copied.board.fen() == board.board.fen()
    assert copied.board.is_repetition(3)


def test_current_fifty_move_position_is_a_draw_without_prospective_claims() -> None:
    board = ChessBoard()
    board.board = chess.Board('8/8/8/8/8/2k5/5R2/4K3 w - - 100 51')

    assert board.board.is_fifty_moves()
    assert board.is_game_over()
    assert board.check_winner() is None


def test_bounded_history_declares_longer_cycle_limitation() -> None:
    board = chess.Board()
    play_moves(board, EIGHT_PLY_CYCLE * 2)
    assert board.is_repetition(3)

    history = bounded_repetition_history(board, maximum_plies=REPETITION_HISTORY_PLIES)
    reconstructed = chess.Board(history.starting_fen)
    play_moves(reconstructed, history.moves_uci)

    assert reconstructed.fen() == board.fen()
    assert not reconstructed.is_repetition(3)
