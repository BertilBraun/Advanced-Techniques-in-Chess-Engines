from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

from src.games.chess.ChessBoard import ChessBoard
from src.self_play.SelfPlayCpp import SelfPlayCpp, SelfPlayGame


def self_play_client(shortcut_strength: float) -> SelfPlayCpp:
    client = object.__new__(SelfPlayCpp)
    client.args = SimpleNamespace(num_moves_after_which_to_play_greedy=50)
    client.endgame_shortcut_strength = shortcut_strength
    return client


def game_from_fen(fen: str, played_move_count: int = 50) -> SelfPlayGame:
    game = SelfPlayGame()
    game.board = ChessBoard.from_fen(fen)
    game.played_moves = [chess.Move.null()] * played_move_count
    return game


@pytest.mark.parametrize(
    ('strength', 'random_sample', 'expected'),
    (
        (1.0, 0.99, True),
        (0.5, 0.49, True),
        (0.5, 0.50, False),
        (0.0, 0.00, False),
    ),
)
def test_fast_endgame_playout_fades_out(
    monkeypatch: pytest.MonkeyPatch,
    strength: float,
    random_sample: float,
    expected: bool,
) -> None:
    client = self_play_client(strength)
    game = game_from_fen('8/8/8/8/8/4K3/7P/4k3 w - - 0 1')
    monkeypatch.setattr('src.self_play.SelfPlayCpp.random.random', lambda: random_sample)

    assert client._should_force_fast_endgame_playout(game) is expected


def test_fast_endgame_playout_keeps_normal_search_before_greedy_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = self_play_client(1.0)
    game = game_from_fen('8/8/8/8/8/4K3/7P/4k3 w - - 0 1', played_move_count=49)
    monkeypatch.setattr('src.self_play.SelfPlayCpp.random.random', lambda: 0.0)

    assert not client._should_force_fast_endgame_playout(game)


@pytest.mark.parametrize(
    ('strength', 'random_sample', 'expected'),
    (
        (1.0, 0.19, True),
        (1.0, 0.20, False),
        (0.5, 0.09, True),
        (0.5, 0.10, False),
        (0.0, 0.00, False),
    ),
)
def test_low_material_termination_probability_fades_out(
    monkeypatch: pytest.MonkeyPatch,
    strength: float,
    random_sample: float,
    expected: bool,
) -> None:
    client = self_play_client(strength)
    game = game_from_fen('8/8/8/8/8/4K3/7P/4k3 w - - 0 1')
    monkeypatch.setattr('src.self_play.SelfPlayCpp.random.random', lambda: random_sample)

    assert client._should_terminate_low_material_game(game) is expected


def test_low_material_termination_requires_one_side_below_four_pieces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = self_play_client(1.0)
    game = game_from_fen('8/p7/8/8/8/4K3/PP5p/R3k2r w - - 0 1')
    monkeypatch.setattr('src.self_play.SelfPlayCpp.random.random', lambda: 0.0)

    assert not client._should_terminate_low_material_game(game)
