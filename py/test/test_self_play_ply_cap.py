from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

from src.self_play.SelfPlayCpp import SelfPlayCpp, SelfPlayGame
from src.self_play.SelfPlayDataset import SelfPlayDataset


def self_play_client(iteration: int, final_maximum_game_plies: int | None = None) -> SelfPlayCpp:
    client = object.__new__(SelfPlayCpp)
    client.args = SimpleNamespace(
        maximum_game_plies=200,
        maximum_game_plies_until_iteration=50,
        final_maximum_game_plies=final_maximum_game_plies,
        low_material_termination_minimum_plies=0,
        low_material_termination_piece_threshold_per_player=0,
        low_material_termination_probability=0.0,
    )
    client.iteration = iteration
    client.dataset = SelfPlayDataset()
    return client


def game_at_200_plies() -> SelfPlayGame:
    game = SelfPlayGame()
    game.played_moves = [chess.Move.null()] * 200
    return game


def game_at_300_plies() -> SelfPlayGame:
    game = SelfPlayGame()
    game.played_moves = [chess.Move.null()] * 300
    return game


def test_iteration_49_caps_game_at_200_plies_with_material_result(monkeypatch: pytest.MonkeyPatch) -> None:
    client = self_play_client(iteration=49)
    game = game_at_200_plies()
    replacement = SelfPlayGame()
    handled_outcomes: list[float] = []

    def handle_end_of_game(handled_game: SelfPlayGame, game_outcome: float, _reason: object) -> SelfPlayGame:
        assert handled_game is game
        handled_outcomes.append(game_outcome)
        return replacement

    monkeypatch.setattr(client, '_handle_end_of_game', handle_end_of_game)

    assert client._finish_game_after_move(game, native_game_over=False) is replacement
    assert handled_outcomes == [0.0]
    assert client.dataset.stats.num_too_long_games == 1
    assert client.dataset.stats.capped_game_material_scores == [0.0]


def test_cap_uses_material_result(monkeypatch: pytest.MonkeyPatch) -> None:
    client = self_play_client(iteration=50, final_maximum_game_plies=300)
    game = game_at_300_plies()
    replacement = SelfPlayGame()

    monkeypatch.setattr(game.board, 'get_approximate_result_score', lambda: 0.5)
    monkeypatch.setattr(
        client,
        '_handle_end_of_game',
        lambda _, __, ___: replacement,
    )

    assert client._finish_game_after_move(game, native_game_over=False) is replacement
    assert client.dataset.stats.num_too_long_games == 1
    assert client.dataset.stats.capped_game_material_scores == [0.5]


def test_iteration_50_does_not_cap_game_at_200_plies(monkeypatch: pytest.MonkeyPatch) -> None:
    client = self_play_client(iteration=50)
    game = game_at_200_plies()

    def unexpected_end_of_game(_: SelfPlayGame, __: float) -> SelfPlayGame:
        pytest.fail('The ply cap must be disabled at iteration 50.')

    monkeypatch.setattr(client, '_handle_end_of_game', unexpected_end_of_game)

    assert client._finish_game_after_move(game, native_game_over=False) is None


def test_empty_terminal_search_result_finishes_game(monkeypatch: pytest.MonkeyPatch) -> None:
    client = self_play_client(iteration=50)
    game = SelfPlayGame()
    replacement = SelfPlayGame()

    monkeypatch.setattr(client, '_handle_end_of_game', lambda _, __, ___: replacement)

    assert client._finish_terminal_search_root(game, SimpleNamespace(is_terminal=True)) is replacement


def test_empty_non_terminal_search_result_fails_clearly() -> None:
    client = self_play_client(iteration=50)

    with pytest.raises(RuntimeError, match='no visit counts for a non-terminal root'):
        client._finish_terminal_search_root(SelfPlayGame(), SimpleNamespace(is_terminal=False))


def test_unexpanded_reused_root_is_rebuilt_with_history() -> None:
    client = self_play_client(iteration=50)
    replacement = SimpleNamespace(is_terminal=False, is_expanded=False)
    root_factory = SimpleNamespace(
        new_root_with_history=lambda starting_fen, moves_uci: replacement,
    )
    client.mcts = root_factory
    game = SelfPlayGame()
    game.already_expanded_node = SimpleNamespace(is_terminal=False, is_expanded=False)

    client._prepare_search_root(game)

    assert game.already_expanded_node is replacement


@pytest.mark.parametrize(
    'retained_root',
    (
        SimpleNamespace(is_terminal=True, is_expanded=False),
        SimpleNamespace(is_terminal=False, is_expanded=True),
    ),
)
def test_terminal_or_expanded_reused_root_is_retained(retained_root: SimpleNamespace) -> None:
    client = self_play_client(iteration=50)
    client.mcts = SimpleNamespace(
        new_root_with_history=lambda *_: pytest.fail('Reusable root must not be rebuilt.'),
    )
    game = SelfPlayGame()
    game.already_expanded_node = retained_root

    client._prepare_search_root(game)

    assert game.already_expanded_node is retained_root


@pytest.mark.parametrize(
    ('iteration', 'expected_maximum'),
    (
        (0, 200),
        (40, 225),
        (79, 249),
        (80, 250),
        (300, 250),
    ),
)
def test_maximum_game_plies_increases_from_200_to_250(
    iteration: int,
    expected_maximum: int,
) -> None:
    client = self_play_client(iteration, final_maximum_game_plies=250)
    client.args.maximum_game_plies_until_iteration = 80

    assert client._maximum_game_plies() == expected_maximum
