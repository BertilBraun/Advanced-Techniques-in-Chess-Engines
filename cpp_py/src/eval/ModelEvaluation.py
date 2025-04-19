from __future__ import annotations
from itertools import chain
from os import PathLike
import os
from pathlib import Path
import chess
import random

import numpy as np
from typing import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.Network import Network
from src.dataset.SelfPlayTrainDataset import SelfPlayTrainDataset
from src.eval.Bot import check_winner
from src.train.TrainingArgs import TrainingArgs
from src.settings import USE_GPU, TORCH_DTYPE
from src.util.save_paths import load_model, model_save_path
from src.util.tensorboard import log_text

import AlphaZeroCpp


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

    def update(self, result: chess.Color | None, main_player: chess.Color) -> None:
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


EvaluationModel = Callable[[list[chess.Board]], list[list[int]]]

# The device to use for evaluation. Since Training is done on device 0, we can use device 1 for evaluation
EVAL_DEVICE = 0


class ModelEvaluation:
    """This class provides functionallity to evaluate only the models performance without any search, to be used in the training loop to evaluate the model against itself"""

    def __init__(self, iteration: int, args: TrainingArgs, num_games: int = 64) -> None:
        self.iteration = iteration
        self.num_games = num_games
        self.args = args

    def evaluate_model_vs_dataset(self, dataset: SelfPlayTrainDataset) -> tuple[float, float, float, float]:
        device = torch.device(f'cuda:{EVAL_DEVICE}' if USE_GPU else 'cpu')
        model = load_model(model_save_path(self.iteration, self.args.save_path), self.args.network, device)

        return self._evaluate_model_vs_dataset(model, dataset.as_dataloader(batch_size=128, num_workers=1))

    @staticmethod
    def _evaluate_model_vs_dataset(model: Network, dataloader: DataLoader) -> tuple[float, float, float, float]:
        model.eval()

        total_top1_correct = 0
        total_top5_correct = 0
        total_top10_correct = 0
        total_policy_total = 0
        total_value_loss = 0.0

        with torch.no_grad():
            for boards, moves_list, outcomes in dataloader:
                boards = boards.to(device=model.device, dtype=TORCH_DTYPE)
                moves_list = moves_list.to(device=model.device, dtype=TORCH_DTYPE)
                outcomes = outcomes.to(device=model.device, dtype=TORCH_DTYPE).unsqueeze(1)

                policy_outputs, value_outputs = model(boards)

                policy_preds = torch.softmax(policy_outputs, dim=1)

                for i in range(len(boards)):
                    top1 = policy_preds[i].topk(1).indices
                    top5 = policy_preds[i].topk(5).indices
                    top10 = policy_preds[i].topk(10).indices
                    true_moves = moves_list[i].nonzero().squeeze()

                    if torch.any(top1 == true_moves):
                        total_top1_correct += 1
                    if torch.any(top5.unsqueeze(1) == true_moves):
                        total_top5_correct += 1
                    if torch.any(top10.unsqueeze(1) == true_moves):
                        total_top10_correct += 1

                    total_policy_total += 1

                total_value_loss += F.mse_loss(value_outputs, outcomes).item()

        top1_accuracy = total_top1_correct / total_policy_total
        top5_accuracy = total_top5_correct / total_policy_total
        top10_accuracy = total_top10_correct / total_policy_total
        avg_value_loss = total_value_loss / len(dataloader)

        return top1_accuracy, top5_accuracy, top10_accuracy, avg_value_loss

    def play_vs_random(self) -> Results:
        # Random vs Random has a result of: 60% Wins, 28% Losses, 12% Draws
        def encode_move_into_policy(move: chess.Move) -> list[int]:
            return AlphaZeroCpp.action_probabilities([(AlphaZeroCpp.encode_move(move), 1)])

        def random_evaluator(boards: list[chess.Board]) -> list[list[int]]:
            def get_random_policy(board: chess.Board) -> list[int]:
                return encode_move_into_policy(random.choice(list(board.legal_moves)))

            return [get_random_policy(board) for board in boards]

        return self.play_vs_evaluation_model(random_evaluator, 'random')

    def play_two_models_search(self, model_path: str | PathLike) -> Results:
        def opponent_evaluator(boards: list[chess.Board]) -> list[list[int]]:
            model_path_str = self._get_jit_model_path(model_path)
            results = AlphaZeroCpp.board_inference_main(model_path_str, [board.fen() for board in boards])
            return [AlphaZeroCpp.action_probabilities(visit_counts) for score, visit_counts in results]

        return self.play_vs_evaluation_model(opponent_evaluator, os.path.basename(model_path))

    def play_vs_evaluation_model(
        self, evaluation_model: EvaluationModel, name: str, num_games: int | None = None
    ) -> Results:
        results = Results(0, 0, 0)

        def model1(boards: list[chess.Board]) -> list[list[int]]:
            current_model_path = model_save_path(self.iteration, self.args.save_path)
            results = AlphaZeroCpp.board_inference_main(
                self._get_jit_model_path(current_model_path),
                [board.fen() for board in boards],
            )
            return [AlphaZeroCpp.action_probabilities(visit_counts) for score, visit_counts in results]

        num_games = num_games or self.num_games

        results += self._play_two_models_search(model1, evaluation_model, num_games // 2, name + '_vs_current')
        results -= self._play_two_models_search(evaluation_model, model1, num_games // 2, 'current_vs_' + name)

        return results

    def _get_jit_model_path(self, model_path: str | PathLike) -> str:
        model_path_str = str(Path(model_path).absolute())
        if model_path_str.endswith('.jit.pt'):
            return model_path_str
        if model_path_str.endswith('.pt'):
            return model_path_str[:-3] + '.jit.pt'
        assert False, f'Invalid model path: {model_path_str}'

    def _play_two_models_search(
        self, model1: EvaluationModel, model2: EvaluationModel, num_games: int, name: str
    ) -> Results:
        results = Results(0, 0, 0)

        games = [chess.Board() for _ in range(num_games)]

        # start from different starting positions, as the players are deterministic
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
            'rnbqkb1r/ppppppp1/5n2/8/4P2p/8/PPPP1PPP/RNBQKBNR w KQkq - 2 2',  # Dutch Defense
            'rnbqkb1r/pppppppp/5n2/8/3PP3/4B3/PPP2PPP/RN1QKBNR b KQkq - 3 3',  # London System
            'rnbqkb1r/pppppppp/5n2/8/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 2',  # Réti Opening
        ]

        for game, fen in zip(games, opening_fens):
            game.set_fen(fen)

        while games:
            games_for_player1 = [game for game in games if game.turn == chess.WHITE]
            games_for_player2 = [game for game in games if game.turn == chess.BLACK]

            policies1 = np.array(model1(games_for_player1))
            policies2 = np.array(model2(games_for_player2))

            for game, policy in chain(zip(games_for_player1, policies1), zip(games_for_player2, policies2)):
                # decrease the probability of playing the last 5 moves again by deviding the probability by 5, 4, 3, 2, 1
                for i, move in enumerate(game.move_stack[-10:]):
                    policy[AlphaZeroCpp.encode_move(move)] /= i + 1

                policy /= policy.sum()

                move = np.argmax(policy).item()
                game.push(AlphaZeroCpp.decode_move(move))

                if game.is_game_over():
                    game.result()
                    results.update(check_winner(game), chess.WHITE)

                    history: list[str] = []
                    while True:
                        try:
                            move = game.pop()
                            history.append(str(move))
                        except IndexError:
                            break
                    history.append(f'FEN"{game.fen()}"')
                    history.reverse()
                    moves = ','.join(history)
                    log_text(f'evaluation_moves/{self.iteration}/{name}', f'{check_winner(game)}:{moves}')

            games = [game for game in games if not game.is_game_over()]

        return results
