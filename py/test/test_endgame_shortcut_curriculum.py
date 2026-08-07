from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.self_play import ChessSelfPlayPolicy, SelfPlayGame
from src.self_play.statistics import SelfPlayStatistics


def self_play_client(shortcut_strength: float) -> ChessSelfPlayPolicy:
    client = object.__new__(ChessSelfPlayPolicy)
    client.args = SimpleNamespace(
        num_moves_after_which_to_play_greedy=50,
        endgame_continuation_start_plies=None,
        low_material_termination_minimum_plies=120,
        low_material_termination_piece_threshold_per_player=4,
        low_material_termination_probability=0.7,
    )
    client.statistics = SelfPlayStatistics()
    client.endgame_shortcut_strength = shortcut_strength
    client.iteration = 0
    return client


def game_from_fen(fen: str, played_move_count: int = 120) -> SelfPlayGame:
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
    monkeypatch.setattr('src.games.chess.self_play.random.random', lambda: random_sample)

    assert client._should_force_fast_endgame_playout(game) is expected


def test_fast_endgame_playout_keeps_normal_search_before_greedy_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = self_play_client(1.0)
    game = game_from_fen('8/8/8/8/8/4K3/7P/4k3 w - - 0 1', played_move_count=49)
    monkeypatch.setattr('src.games.chess.self_play.random.random', lambda: 0.0)

    assert not client._should_force_fast_endgame_playout(game)


@pytest.mark.parametrize(
    ('played_move_count', 'expected'),
    ((249, False), (250, True), (399, True)),
)
def test_configured_endgame_continuation_forces_fast_searches(
    played_move_count: int,
    expected: bool,
) -> None:
    client = self_play_client(0.0)
    client.args.endgame_continuation_start_plies = 250
    game = game_from_fen(chess.STARTING_FEN, played_move_count=played_move_count)

    assert client._should_force_fast_endgame_playout(game) is expected


@pytest.mark.parametrize(
    ('random_sample', 'expected'),
    (
        (0.69, True),
        (0.70, False),
    ),
)
def test_low_material_termination_uses_its_configured_one_shot_probability(
    monkeypatch: pytest.MonkeyPatch,
    random_sample: float,
    expected: bool,
) -> None:
    client = self_play_client(0.0)
    game = game_from_fen('8/8/8/8/8/4K3/7P/4k3 w - - 0 1')
    monkeypatch.setattr('src.games.chess.self_play.random.random', lambda: random_sample)

    assert client._should_terminate_low_material_game(game) is expected
    assert not client._should_terminate_low_material_game(game)
    assert game.low_material_termination_evaluated


def test_low_material_termination_decline_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    client = self_play_client(0.0)
    game = game_from_fen('8/8/8/8/8/4K3/7P/4k3 w - - 0 1')
    random_samples = iter((0.70, 0.0))
    monkeypatch.setattr('src.games.chess.self_play.random.random', lambda: next(random_samples))

    assert not client._should_terminate_low_material_game(game)
    assert not client._should_terminate_low_material_game(game)
    assert game.low_material_termination_evaluated


def test_low_material_termination_requires_one_side_below_four_pieces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = self_play_client(1.0)
    game = game_from_fen('8/p7/8/8/8/4K3/PP5p/R3k2r w - - 0 1')
    monkeypatch.setattr('src.games.chess.self_play.random.random', lambda: 0.0)

    assert not client._should_terminate_low_material_game(game)


def test_low_material_termination_waits_for_its_configured_minimum_ply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = self_play_client(0.0)
    game = game_from_fen('8/8/8/8/8/4K3/7P/4k3 w - - 0 1', played_move_count=119)
    monkeypatch.setattr('src.games.chess.self_play.random.random', lambda: 0.0)

    assert not client._should_terminate_low_material_game(game)
    assert not game.low_material_termination_evaluated

    game.played_moves.append(chess.Move.null())

    assert client._should_terminate_low_material_game(game)
