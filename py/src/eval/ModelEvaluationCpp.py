from __future__ import annotations
from os import PathLike
import os
from pathlib import Path
import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AlphaZeroCpp import MCTSParams

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.Network import Network
from src.eval.ModelEvaluationPy import _play_two_models_search, policy_evaluator, Results, EvaluationModel
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.train.TrainingArgs import TrainingArgs
from src.cluster.InferenceClient import InferenceClient
from src.mcts.MCTS import action_probabilities
from src.settings import USE_GPU, PLAY_C_PARAM, CurrentBoard, CurrentGame
from src.util.save_paths import load_model, model_save_path


class ModelEvaluation:
    """This class provides functionallity to evaluate only the models performance without any search, to be used in the training loop to evaluate the model against itself"""

    def __init__(
        self, iteration: int, args: TrainingArgs, device_id: int, num_games: int = 64, num_searches_per_turn: int = 20
    ) -> None:
        self.iteration = iteration
        self.num_games = num_games
        self.args = args
        self.num_searches_per_turn = num_searches_per_turn
        self.device_id = device_id

    @property
    def mcts_args(self) -> MCTSParams:
        from AlphaZeroCpp import MCTSParams

        return MCTSParams(
            num_parallel_searches=self.args.self_play.mcts.num_parallel_searches,
            c_param=PLAY_C_PARAM,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=1.0,
            min_visit_count=0,
            num_threads=self.num_games // 2,
            node_reuse_discount=1.0,
            num_full_searches=self.num_searches_per_turn,
            num_fast_searches=self.num_searches_per_turn,
        )

    def evaluate_model_vs_dataset(self, dataset: SelfPlayDataset) -> tuple[float, float, float, float]:
        device = torch.device(f'cuda:{self.device_id}' if USE_GPU else 'cpu')
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
        from AlphaZeroCpp import INVALID_NODE, InferenceClientParams, MCTS

        if not Path(model_path).exists():
            print(f'Model path {model_path} does not exist. Skipping evaluation.')
            return Results(self.num_games, 0, 0)

        opponent = MCTS(InferenceClientParams(self.device_id, str(model_path), 16), self.mcts_args)

        def opponent_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            assert self.args.evaluation is not None, 'Evaluation args must be set to use opponent evaluator'
            results = opponent.search(  # noqa: F821
                [(board.board.fen(), INVALID_NODE, False) for board in boards]
            )
            return [action_probabilities(result.visits) for result in results.results]

        res = self.play_vs_evaluation_model(opponent_evaluator, os.path.basename(model_path))

        del opponent  # Free the memory used by the MCTS client

        return res

    def play_policy_vs_random(self) -> Results:
        current_model = InferenceClient(self.device_id, self.args.network, self.args.save_path)
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

    def play_vs_stockfish(self, level: int) -> Results:
        import chess.engine

        engine = chess.engine.SimpleEngine.popen_uci('stockfish')
        engine.configure({'Skill Level': level})
        engine.configure({'Threads': 1})  # Limit to one thread for consistency
        engine.configure({'Hash': 1024})  # Set hash size to 1GB

        def stockfish_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            def get_stockfish_policy(board: CurrentBoard) -> np.ndarray:
                # Use Stockfish to get the best move
                result = engine.play(board.board, chess.engine.Limit(time=0.01, depth=level + 1))
                move = result.move

                if not move:
                    raise ValueError('Stockfish did not return a move.')

                if move.promotion is not None:
                    # Handle promotion moves
                    move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

                return CurrentGame.encode_moves([move], board)

            return [get_stockfish_policy(board) for board in boards]

        results = self.play_vs_evaluation_model(stockfish_evaluator, f'stockfish_level_{level}')

        engine.quit()  # Clean up the Stockfish engine

        return results

    def play_vs_evaluation_model(self, eval_model: EvaluationModel, name: str) -> Results:
        from AlphaZeroCpp import INVALID_NODE, InferenceClientParams, MCTS

        current = MCTS(
            InferenceClientParams(self.device_id, str(model_save_path(self.iteration, self.args.save_path)), 16),
            self.mcts_args,
        )

        def current_model(boards: list[CurrentBoard]) -> list[np.ndarray]:
            assert self.args.evaluation is not None, 'Evaluation args must be set to use opponent evaluator'
            results = current.search(  # noqa: F821
                [(board.board.fen(), INVALID_NODE, False) for board in boards]
            )
            return [action_probabilities(result.visits) for result in results.results]

        results = Results(0, 0, 0)

        results += _play_two_models_search(
            self.iteration, current_model, eval_model, self.num_games // 2, name + '_vs_current'
        )
        results -= _play_two_models_search(
            self.iteration, eval_model, current_model, self.num_games // 2, 'current_vs_' + name
        )

        del current  # Free the memory used by the MCTS client

        return results


if __name__ == '__main__':
    from src.settings import TRAINING_ARGS
    from src.eval.ModelEvaluationPy import EVAL_DEVICE

    evaluation = ModelEvaluation(0, TRAINING_ARGS, EVAL_DEVICE, 100, 400)
    print('Evaluation vs Random:', evaluation.play_vs_random())
