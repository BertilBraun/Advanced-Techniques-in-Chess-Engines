from __future__ import annotations
import random

import numpy as np
from typing import Callable
from dataclasses import dataclass

from src.Network import Network
from src.alpha_zero.SelfPlay import sample_move
from src.mcts.MCTS import MCTS
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import CURRENT_BOARD, CURRENT_GAME
from src.games.Game import Player


@dataclass
class Results:
    wins: int
    losses: int
    draws: int

    def __add__(self, other: Results) -> Results:
        return Results(
            wins=self.wins + other.wins,
            losses=self.losses + other.losses,
            draws=self.draws + other.draws,
        )

    def update(self, result: Player | None, main_player: Player) -> None:
        if result is None:
            self.draws += 1
        elif result == main_player:
            self.wins += 1
        else:
            self.losses += 1

    def __neg__(self) -> Results:
        return Results(self.losses, self.wins, self.draws)

    def __str__(self) -> str:
        return f'Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}'


EvaluationModel = Callable[[list[CURRENT_BOARD]], list[np.ndarray]]


class ModelEvaluation:
    # This class provides functionallity to evaluate only the models performance without any search, to be used in the training loop to evaluate the model against itself

    def play_vs_random(self, model: Network, num_games: int = 64, num_searches_per_turn: int = 20) -> Results:
        # Random vs Random has a result of: 60% Wins, 28% Losses, 12% Draws
        results = Results(0, 0, 0)

        mcts_args = MCTSArgs(
            num_searches_per_turn=num_searches_per_turn,
            c_param=2,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=1.0,
        )

        def model1(boards: list[CURRENT_BOARD]) -> list[np.ndarray]:
            return MCTS(model, mcts_args).search(boards)

        def model2(boards: list[CURRENT_BOARD]) -> list[np.ndarray]:
            def get_random_policy(board: CURRENT_BOARD) -> np.ndarray:
                return CURRENT_GAME.encode_moves([random.choice(board.get_valid_moves())])

            return [get_random_policy(board) for board in boards]

        results += self._play_two_models_search(model1, model2, num_games // 2)
        results += -self._play_two_models_search(model2, model1, num_games // 2)

        return results

    def play_two_models_search(
        self, current_model: Network, previous_model: Network, num_games: int = 64, num_searches_per_turn: int = 20
    ) -> Results:
        results = Results(0, 0, 0)

        mcts_args = MCTSArgs(
            num_searches_per_turn=num_searches_per_turn,
            c_param=2,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=1.0,
        )

        def model1(boards: list[CURRENT_BOARD]) -> list[np.ndarray]:
            return MCTS(current_model, mcts_args).search(boards)

        def model2(boards: list[CURRENT_BOARD]) -> list[np.ndarray]:
            return MCTS(previous_model, mcts_args).search(boards)

        results += self._play_two_models_search(model1, model2, num_games // 2)
        results += -self._play_two_models_search(model2, model1, num_games // 2)

        return results

    def _play_two_models_search(self, model1: EvaluationModel, model2: EvaluationModel, num_games: int) -> Results:
        results = Results(0, 0, 0)

        games = [CURRENT_BOARD() for _ in range(num_games)]
        while games:
            assert all(game.current_player == games[0].current_player for game in games)
            if games[0].current_player == 1:
                policies = model1(games)
            else:
                policies = model2(games)

            for game, policy in zip(games, policies):
                game.make_move(sample_move(policy))

                if game.is_game_over():
                    results.update(game.check_winner(), 1)

            games = [game for game in games if not game.is_game_over()]

        return results
