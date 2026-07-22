from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

from src.self_play.SelfPlayCpp import SelfPlayCpp, SelfPlayGame


def self_play_client(iteration: int, final_maximum_game_plies: int | None = None) -> SelfPlayCpp:
    client = object.__new__(SelfPlayCpp)
    client.args = SimpleNamespace(
        maximum_game_plies=200,
        maximum_game_plies_until_iteration=50,
        final_maximum_game_plies=final_maximum_game_plies,
    )
    client.iteration = iteration
    return client


def game_at_200_plies() -> SelfPlayGame:
    game = SelfPlayGame()
    game.played_moves = [chess.Move.null()] * 200
    return game


def test_iteration_49_caps_game_at_200_plies_as_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    client = self_play_client(iteration=49)
    game = game_at_200_plies()
    replacement = SelfPlayGame()
    handled_outcomes: list[float] = []

    def handle_end_of_game(handled_game: SelfPlayGame, game_outcome: float) -> SelfPlayGame:
        assert handled_game is game
        handled_outcomes.append(game_outcome)
        return replacement

    monkeypatch.setattr(client, '_handle_end_of_game', handle_end_of_game)

    assert client._finish_game_after_move(game, native_game_over=False) is replacement
    assert handled_outcomes == [0.0]


def test_iteration_50_does_not_cap_game_at_200_plies(monkeypatch: pytest.MonkeyPatch) -> None:
    client = self_play_client(iteration=50)
    game = game_at_200_plies()

    def unexpected_end_of_game(_: SelfPlayGame, __: float) -> SelfPlayGame:
        pytest.fail('The ply cap must be disabled at iteration 50.')

    monkeypatch.setattr(client, '_handle_end_of_game', unexpected_end_of_game)

    assert client._finish_game_after_move(game, native_game_over=False) is None


@pytest.mark.parametrize(
    ('iteration', 'expected_maximum'),
    (
        (0, 200),
        (40, 300),
        (79, 397),
        (80, 400),
        (300, 400),
    ),
)
def test_maximum_game_plies_increases_from_200_to_400(
    iteration: int,
    expected_maximum: int,
) -> None:
    client = self_play_client(iteration, final_maximum_game_plies=400)
    client.args.maximum_game_plies_until_iteration = 80

    assert client._maximum_game_plies() == expected_maximum
