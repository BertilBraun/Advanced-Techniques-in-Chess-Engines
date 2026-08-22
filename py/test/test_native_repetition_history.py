from __future__ import annotations

import pytest

AlphaZeroCpp = pytest.importorskip('AlphaZeroCpp')

STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
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


def test_native_threefold_ends_on_third_occurrence() -> None:
    second_occurrence = AlphaZeroCpp.new_root_with_history(STARTING_FEN, KNIGHT_CYCLE, 1)
    third_occurrence = AlphaZeroCpp.new_root_with_history(STARTING_FEN, KNIGHT_CYCLE * 2, 1)

    assert second_occurrence.repetition_count == 1
    assert not second_occurrence.is_terminal
    assert third_occurrence.repetition_count == 2
    assert third_occurrence.is_terminal


def test_native_history_supports_longer_cycles() -> None:
    root = AlphaZeroCpp.new_root_with_history(STARTING_FEN, EIGHT_PLY_CYCLE * 2, 1)

    assert root.repetition_count == 2
    assert root.is_terminal


def test_native_irreversible_move_resets_history() -> None:
    root = AlphaZeroCpp.new_root_with_history(STARTING_FEN, KNIGHT_CYCLE + ('e2e4',), 1)

    assert root.repetition_count == 0
    assert not root.is_terminal


def test_native_history_is_bounded_to_fifty_move_window() -> None:
    reversible_moves = EIGHT_PLY_CYCLE * 13
    root = AlphaZeroCpp.new_root_with_history(STARTING_FEN, reversible_moves, 1)

    assert root.is_terminal


def test_history_blind_root_misses_repetition_the_aware_root_sees() -> None:
    import chess

    board = chess.Board()
    for move in KNIGHT_CYCLE * 2:
        board.push_uci(move)
    history_blind = AlphaZeroCpp.new_root(board.fen(), 1)
    history_aware = AlphaZeroCpp.new_root_with_history(STARTING_FEN, KNIGHT_CYCLE * 2, 1)
    assert not history_blind.is_terminal
    assert history_aware.is_terminal


def test_history_replay_reproduces_castled_fen() -> None:
    import chess

    castling_moves = ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'e1g1')
    board = chess.Board()
    for move in castling_moves:
        board.push_uci(move)
    root = AlphaZeroCpp.new_root_with_history(STARTING_FEN, castling_moves, 1)
    assert root.fen == board.fen()
