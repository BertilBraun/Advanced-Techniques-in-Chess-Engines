from __future__ import annotations
from itertools import chain
from os import PathLike
import os
from pathlib import Path
import random

import numpy as np
from typing import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.Network import Network
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.train.TrainingArgs import MCTSParams, TrainingArgs
from src.cluster.InferenceClient import InferenceClient
from src.mcts.MCTS import MCTS, action_probabilities
from src.settings import USE_GPU, CurrentBoard, CurrentGame
from src.games.Game import Player
from src.util.save_paths import load_model, model_save_path
from src.util.tensorboard import log_text


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
        return f'W/D/L: {self.wins}/{self.draws}/{self.losses}'


EvaluationModel = Callable[[list[CurrentBoard]], list[np.ndarray]]

# The device to use for evaluation. Since Training is done on device 0, we can use device 1 for evaluation
EVAL_DEVICE = 0


def policy_evaluator(current_model: InferenceClient) -> EvaluationModel:
    def evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
        results = current_model.inference_batch(boards)
        policies = []
        for board, (visit_counts, value) in zip(boards, results):
            policy = np.zeros(CurrentGame.action_size, dtype=np.float32)
            for move, score in visit_counts:
                policy[move] = score
            assert policy.sum() >= 0, f'Policy for board {board} has negative sum: {policy.sum()}'
            policies.append(policy / policy.sum())
        return policies

    return evaluator


class ModelEvaluation:
    """This class provides functionallity to evaluate only the models performance without any search, to be used in the training loop to evaluate the model against itself"""

    def __init__(
        self, iteration: int, args: TrainingArgs, num_games: int = 64, num_searches_per_turn: int = 20
    ) -> None:
        self.iteration = iteration
        self.num_games = num_games
        self.args = args

        self.mcts_args = MCTSParams(
            num_searches_per_turn=num_searches_per_turn,
            num_parallel_searches=args.self_play.mcts.num_parallel_searches,
            c_param=2,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=1.0,
            min_visit_count=0,
            num_threads=2,
            percentage_of_node_visits_to_keep=1.0,
        )

    def evaluate_model_vs_dataset(self, dataset: SelfPlayDataset) -> tuple[float, float, float, float]:
        device = torch.device(f'cuda:{EVAL_DEVICE}' if USE_GPU else 'cpu')
        model = load_model(model_save_path(self.iteration, self.args.save_path), self.args.network, device)

        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        return self._evaluate_model_vs_dataset(model, dataloader)

    @staticmethod
    def _evaluate_model_vs_dataset(model: Network, dataloader: DataLoader) -> tuple[float, float, float, float]:
        model.eval()

        total_top1_correct = 0
        total_top5_correct = 0
        total_top10_correct = 0
        total_policy_total = 0
        total_value_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                board, moves, outcome = batch
                board = board.to(model.device)
                moves = moves.to(model.device)
                outcome = outcome.to(model.device).unsqueeze(1)

                policy_pred, value_output = model(board)

                for i in range(len(moves)):
                    top1 = policy_pred[i].topk(1).indices
                    top5 = policy_pred[i].topk(5).indices
                    top10 = policy_pred[i].topk(10).indices
                    true_moves = moves[i].nonzero().squeeze()

                    if torch.any(top1 == true_moves):
                        total_top1_correct += 1
                    if torch.any(top5.unsqueeze(1) == true_moves):
                        total_top5_correct += 1
                    if torch.any(top10.unsqueeze(1) == true_moves):
                        total_top10_correct += 1

                    total_policy_total += 1

                total_value_loss += F.mse_loss(value_output, outcome).item()

        top1_accuracy = total_top1_correct / total_policy_total
        top5_accuracy = total_top5_correct / total_policy_total
        top10_accuracy = total_top10_correct / total_policy_total
        avg_value_loss = total_value_loss / len(dataloader)

        return top1_accuracy, top5_accuracy, top10_accuracy, avg_value_loss

    def play_vs_random(self) -> Results:
        # Random vs Random has a result of: 60% Wins, 28% Losses, 12% Draws

        def random_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            def get_random_policy(board: CurrentBoard) -> np.ndarray:
                return CurrentGame.encode_moves([random.choice(board.get_valid_moves())], board)

            return [get_random_policy(board) for board in boards]

        return self.play_vs_evaluation_model(random_evaluator, 'random')

    def play_two_models_search(self, model_path: str | PathLike) -> Results:
        if not Path(model_path).exists():
            print(f'Model path {model_path} does not exist. Skipping evaluation.')
            return Results(self.num_games, 0, 0)

        opponent = InferenceClient(EVAL_DEVICE, self.args.network, self.args.save_path)
        opponent.load_model(model_path)

        def opponent_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            results = MCTS(opponent, self.mcts_args).search(boards)
            return [action_probabilities(result.visit_counts) for result in results]

        # opponent_evaluator = policy_evaluator(opponent)

        return self.play_vs_evaluation_model(opponent_evaluator, os.path.basename(model_path))

    def play_policy_vs_random(self) -> Results:
        current_model = InferenceClient(EVAL_DEVICE, self.args.network, self.args.save_path)
        current_model.update_iteration(self.iteration)

        policy_model = policy_evaluator(current_model)

        def random_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            def get_random_policy(board: CurrentBoard) -> np.ndarray:
                return CurrentGame.encode_moves([random.choice(board.get_valid_moves())], board)

            return [get_random_policy(board) for board in boards]

        results = Results(0, 0, 0)

        results += _play_two_models_search(
            self.iteration, policy_model, random_evaluator, self.num_games // 2, 'policy_vs_random'
        )
        results -= _play_two_models_search(
            self.iteration, random_evaluator, policy_model, self.num_games // 2, 'random_vs_policy'
        )

        return results

    def play_vs_evaluation_model(self, eval_model: EvaluationModel, name: str) -> Results:
        client = InferenceClient(EVAL_DEVICE, self.args.network, self.args.save_path)
        client.update_iteration(self.iteration)

        def current_model(boards: list[CurrentBoard]) -> list[np.ndarray]:
            results = MCTS(client, self.mcts_args).search(boards)
            return [action_probabilities(result.visit_counts) for result in results]

        # model1 = policy_evaluator(current_model)

        results = Results(0, 0, 0)

        results += _play_two_models_search(
            self.iteration, current_model, eval_model, self.num_games // 2, name + '_vs_current'
        )
        results -= _play_two_models_search(
            self.iteration, eval_model, current_model, self.num_games // 2, 'current_vs_' + name
        )

        return results


def _new_game() -> tuple[CurrentBoard, list[str]]:
    game = CurrentBoard()
    move_history: list[str] = []

    # Play a random moves to start the game in different states
    random_moves_to_play = int(random.random() * 4) + 1  # Play between 1 and 4 random moves
    for _ in range(random_moves_to_play):
        move = random.choice(game.get_valid_moves())
        move_history.append(str(CurrentGame.encode_move(move, game)))
        game.make_move(move)
        if game.is_game_over():
            # If the game is over, start a new game
            return _new_game()

    return game, move_history


def _play_two_models_search(
    iteration, model1: EvaluationModel, model2: EvaluationModel, num_games: int, name: str
) -> Results:
    results = Results(0, 0, 0)

    games = [CurrentBoard() for _ in range(num_games)]

    game_move_histories: list[list[str]] = [[] for _ in range(num_games)]

    # start from different starting positions, as the players are deterministic
    for i in range(num_games):
        games[i], game_move_histories[i] = _new_game()

    while games:
        games_for_player1 = [game for game in games if game.current_player == 1]
        games_for_player1_indices = [i for i, game in enumerate(games) if game.current_player == 1]
        games_for_player2 = [game for game in games if game.current_player == -1]
        games_for_player2_indices = [i for i, game in enumerate(games) if game.current_player == -1]

        policies1 = model1(games_for_player1)
        policies2 = model2(games_for_player2)

        for game, policy, game_index in chain(
            zip(games_for_player1, policies1, games_for_player1_indices),
            zip(games_for_player2, policies2, games_for_player2_indices),
        ):
            # decrease the probability of playing the last 5 moves again by deviding the probability by 5, 4, 3, 2, 1
            for i, move in enumerate(game_move_histories[game_index][-10:]):
                policy[int(move)] /= i + 1

            policy /= policy.sum()

            move = np.argmax(policy).item()
            game.make_move(CurrentGame.decode_move(move, game))
            game_move_histories[game_index].append(str(move))

            if game.is_game_over() or len(game_move_histories[game_index]) >= 200:
                results.update(game.check_winner(), main_player=1)

                moves = ','.join(game_move_histories[game_index])
                log_text(
                    f'evaluation_moves/{iteration}/{name}',
                    str(game.check_winner()) + ':' + moves,
                )

        games = [game for i, game in enumerate(games) if not game.is_game_over() and len(game_move_histories[i]) < 200]

    return results


if __name__ == '__main__':
    from src.settings import TRAINING_ARGS

    evaluation = ModelEvaluation(0, TRAINING_ARGS, 100, 400)
    print('Evaluation vs Random:', evaluation.play_vs_random())
