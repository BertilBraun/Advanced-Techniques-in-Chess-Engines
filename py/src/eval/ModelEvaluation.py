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
from src.settings import CURRENT_GAME, USE_GPU, CurrentBoard, CurrentGame
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
        return f'Wins: {self.wins}, Losses: {self.losses}, Draws: {self.draws}'


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
            results = MCTS(opponent, self.mcts_args).search([(board, None) for board in boards])
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

        results += self._play_two_models_search(policy_model, random_evaluator, self.num_games // 2, 'policy_vs_random')
        results -= self._play_two_models_search(random_evaluator, policy_model, self.num_games // 2, 'random_vs_policy')

        return results

    def play_vs_evaluation_model(self, eval_model: EvaluationModel, name: str) -> Results:
        client = InferenceClient(EVAL_DEVICE, self.args.network, self.args.save_path)
        client.update_iteration(self.iteration)

        def current_model(boards: list[CurrentBoard]) -> list[np.ndarray]:
            results = MCTS(client, self.mcts_args).search([(board, None) for board in boards])
            return [action_probabilities(result.visit_counts) for result in results]

        # model1 = policy_evaluator(current_model)

        results = Results(0, 0, 0)

        results += self._play_two_models_search(current_model, eval_model, self.num_games // 2, name + '_vs_current')
        results -= self._play_two_models_search(eval_model, current_model, self.num_games // 2, 'current_vs_' + name)

        return results

    def _play_two_models_search(
        self, model1: EvaluationModel, model2: EvaluationModel, num_games: int, name: str
    ) -> Results:
        results = Results(0, 0, 0)

        games = [CurrentBoard() for _ in range(num_games)]

        game_move_histories: list[list[str]] = [[] for _ in range(num_games)]
        game_to_index = {game: i for i, game in enumerate(games)}

        # start from different starting positions, as the players are deterministic
        if CURRENT_GAME == 'chess':
            opening_fens = [
                'rnbqkb1r/pppp1ppp/5n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3',  # Ruy-Lopez (Spanish Game)
                'rnbqkb1r/pppp1ppp/5n2/4p3/B3P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3',  # Italian Game
                'rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 3 3',  # Scotch Game
                'rnbqkb1r/pp1ppppp/5n2/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 2 2',  # Sicilian Defense
                'rnbqkb1r/pppp1ppp/4pn2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 2 3',  # French Defense
                'rnbqkb1r/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR w KQkq c6 2 2',  # Caro-Kann Defense
                'rnbqkb1r/ppp1pppp/3p4/8/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3',  # Pirc Defense
                'rnbqkb1r/ppp1pppp/3p4/8/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 2 3',  # Modern Defense
                'rnbqkb1r/pppp1ppp/8/4p3/3Pn3/5N2/PPP2PPP/RNBQKB1R w KQkq - 3 3',  # Alekhine’s Defense
                'rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 2 3',  # King's Indian Defense
                'rnbqkb1r/pppp1ppp/5n2/4p3/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq - 2 3',  # Grünfeld Defense
                'rnbqkb1r/pp1ppppp/5n2/2p5/3PP3/8/PPP2PPP/RNBQKBNR w KQkq c6 2 2',  # Queen’s Gambit Declined
                'rnbqkb1r/pp1ppppp/5n2/8/2pPP3/8/PP3PPP/RNBQKBNR w KQkq - 0 3',  # Queen’s Gambit Accepted
                'rnbqkb1r/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR w KQkq c6 2 2',  # Slav Defense
                'rnbqkb1r/pppp1ppp/4pn2/8/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 2 3',  # Nimzo-Indian Defense
                'rnbqkb1r/pppp1ppp/5n2/4p3/2P5/5NP1/PP1PPP1P/RNBQKB1R b KQkq - 2 3',  # Catalan Opening
                'rnbqkb1r/pppppppp/5n2/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 2 2',  # English Opening
                'rnbqkb1r/pppppppp/5n2/8/4P2p/8/PPPP1PPP/RNBQKBNR w KQkq - 2 2',  # Dutch Defense
                'rnbqkb1r/pppppppp/5n2/8/3PP3/4B3/PPP2PPP/RN1QKBNR b KQkq - 3 3',  # London System
                'rnbqkb1r/pppppppp/5n2/8/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 2',  # Réti Opening
            ]

            for game, fen in zip(games, opening_fens):
                game.set_fen(fen)
                game_move_histories[game_to_index[game]].append(f'FEN"{fen}"')
        else:
            for game in games:
                for _ in range(3):
                    move = random.choice(game.get_valid_moves())
                    game_move_histories[game_to_index[game]].append(str(CurrentGame.encode_move(move, game)))
                    game.make_move(move)

        while games:
            games_for_player1 = [game for game in games if game.current_player == 1]
            games_for_player2 = [game for game in games if game.current_player == -1]

            policies1 = model1(games_for_player1)
            policies2 = model2(games_for_player2)

            for game, policy in chain(zip(games_for_player1, policies1), zip(games_for_player2, policies2)):
                # decrease the probability of playing the last 5 moves again by deviding the probability by 5, 4, 3, 2, 1
                # for i, move in enumerate(game_move_histories[game_to_index[game]][-10:]):
                #     if not move.startswith('FEN'):
                #         policy[int(move)] /= i + 1

                policy /= policy.sum()

                move = np.argmax(policy).item()
                game.make_move(CurrentGame.decode_move(move, game))
                game_move_histories[game_to_index[game]].append(str(move))

                if game.is_game_over():
                    results.update(game.check_winner(), main_player=1)

                    moves = ','.join(game_move_histories[game_to_index[game]])
                    log_text(
                        f'evaluation_moves/{self.iteration}/{name}',
                        str(game.check_winner()) + ':' + moves,
                    )

            games = [game for game in games if not game.is_game_over()]

        return results


if __name__ == '__main__':
    from src.settings import TRAINING_ARGS

    evaluation = ModelEvaluation(0, TRAINING_ARGS, 100, 400)
    print('Evaluation vs Random:', evaluation.play_vs_random())
