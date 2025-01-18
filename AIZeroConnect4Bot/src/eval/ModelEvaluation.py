from __future__ import annotations
import random

import numpy as np
from typing import Callable, Coroutine
from dataclasses import dataclass

from src.cluster.InferenceClient import InferenceClient
from src.mcts.MCTS import MCTS
from src.mcts.MCTSArgs import MCTSArgs
from src.settings import TRAINING_ARGS, CurrentBoard, CurrentGame
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

    def __sub__(self, other: Results) -> Results:
        return Results(
            wins=self.wins + other.losses,
            losses=self.losses + other.wins,
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


EvaluationModel = Callable[[list[CurrentBoard]], Coroutine[None, None, list[np.ndarray]]]


class ModelEvaluation:
    """This class provides functionallity to evaluate only the models performance without any search, to be used in the training loop to evaluate the model against itself"""

    def __init__(self, iteration: int, num_games: int = 64, num_searches_per_turn: int = 20) -> None:
        self.iteration = iteration
        self.num_games = num_games
        self.num_searches_per_turn = num_searches_per_turn

        self.mcts_args = MCTSArgs(
            num_searches_per_turn=num_searches_per_turn,
            num_parallel_searches=8,
            c_param=2,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=1.0,
        )

    async def play_vs_random(self) -> Results:
        # Random vs Random has a result of: 60% Wins, 28% Losses, 12% Draws

        async def random_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            def get_random_policy(board: CurrentBoard) -> np.ndarray:
                return CurrentGame.encode_moves([random.choice(board.get_valid_moves())])

            return [get_random_policy(board) for board in boards]

        return await self.play_vs_evaluation_model(random_evaluator)

    async def play_two_models_search(self, previous_model_iteration: int) -> Results:
        previous_model = InferenceClient(0, TRAINING_ARGS)
        previous_model.update_iteration(previous_model_iteration)

        async def previous_model_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            results = await MCTS(previous_model, self.mcts_args).search(boards)
            return [result[0] for result in results]

        return await self.play_vs_evaluation_model(previous_model_evaluator)

    async def play_vs_evaluation_model(self, evaluation_model: EvaluationModel) -> Results:
        results = Results(0, 0, 0)

        current_model = InferenceClient(0, TRAINING_ARGS)
        current_model.update_iteration(self.iteration)

        async def model1(boards: list[CurrentBoard]) -> list[np.ndarray]:
            results = await MCTS(current_model, self.mcts_args).search(boards)
            return [result[0] for result in results]

        results += await self._play_two_models_search(model1, evaluation_model, self.num_games // 2)
        results -= await self._play_two_models_search(evaluation_model, model1, self.num_games // 2)

        return results

    async def _play_two_models_search(
        self, model1: EvaluationModel, model2: EvaluationModel, num_games: int
    ) -> Results:
        results = Results(0, 0, 0)

        games = [CurrentBoard() for _ in range(num_games)]
        while games:
            assert all(game.current_player == games[0].current_player for game in games)
            if games[0].current_player == 1:
                policies = await model1(games)
            else:
                policies = await model2(games)

            for game, policy in zip(games, policies):
                move = CurrentGame.decode_move(np.argmax(policy).item())
                game.make_move(move)

                if game.is_game_over():
                    results.update(game.check_winner(), 1)

            games = [game for game in games if not game.is_game_over()]

        return results
