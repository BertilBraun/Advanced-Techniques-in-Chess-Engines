from types import SimpleNamespace

import pytest

from src.games.chess.self_play import new_game_from_encoded_prefix, policy_search_disagreement
from src.games.chess.contract import CHESS_STATE_CONTRACT


def test_encoded_prefix_restores_complete_legal_move_history() -> None:
    board = CHESS_STATE_CONTRACT.initial_position()
    first_move = board.get_valid_moves()[0]
    first_encoded = CHESS_STATE_CONTRACT.encode_move(first_move, board)
    board.make_move(first_move)
    second_move = board.get_valid_moves()[0]
    second_encoded = CHESS_STATE_CONTRACT.encode_move(second_move, board)

    restored = new_game_from_encoded_prefix((first_encoded, second_encoded))

    assert restored.encoded_moves == [first_encoded, second_encoded]
    assert tuple(move.uci() for move in restored.board.board.move_stack) == (
        first_move.uci(),
        second_move.uci(),
    )


def test_policy_search_disagreement_uses_raw_prior() -> None:
    root = SimpleNamespace(
        children=(
            SimpleNamespace(visits=75, raw_policy=0.25),
            SimpleNamespace(visits=25, raw_policy=0.75),
        )
    )

    assert policy_search_disagreement(root) == pytest.approx(0.75 * 1.5849625007 + 0.25 * -1.5849625007)
