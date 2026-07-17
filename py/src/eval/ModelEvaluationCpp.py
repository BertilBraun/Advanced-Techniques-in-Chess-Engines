from __future__ import annotations
from os import PathLike
from pathlib import Path
import random

from typing import TYPE_CHECKING

import chess
import chess.engine

if TYPE_CHECKING:
    from AlphaZeroCpp import MCTSParams

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.Network import Network
from src.eval.ModelEvaluationPy import (
    _play_paired_models_search,
    _play_two_models_search,
    policy_evaluator,
    Results,
    EvaluationModel,
)
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.train.TrainingArgs import TrainingArgs
from src.cluster.InferenceClient import InferenceClient
from src.games.chess.ChessGame import normalize_move_for_action_space
from src.mcts.MCTS import action_probabilities
from src.settings import USE_GPU, PLAY_C_PARAM, CurrentBoard, CurrentGame
from src.util.save_paths import inference_model_path, load_model, model_save_path
from src.experiment.evaluation_protocol import (
    GameRecord,
    build_paired_schedule,
    load_opening_suite,
)


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
        if args.evaluation is None or args.evaluation.opening_suite_path is None:
            self.paired_schedule = None
        else:
            openings = load_opening_suite(Path(args.evaluation.opening_suite_path))
            self.paired_schedule = build_paired_schedule(openings)
            if len(self.paired_schedule) != num_games:
                raise ValueError(
                    f'Opening suite schedules {len(self.paired_schedule)} games, but evaluation requested {num_games}.'
                )

    @property
    def mcts_args(self) -> MCTSParams:
        from AlphaZeroCpp import MCTSParams

        return MCTSParams(
            num_parallel_searches=self.args.self_play.mcts.num_parallel_searches,
            c_param=PLAY_C_PARAM,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=1.0,
            min_visit_count=0,
            num_threads=self.args.evaluation.mcts_threads,
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
        results, _ = self.play_two_models_paired(model_path)
        return results

    def play_two_models_paired(
        self,
        model_path: str | PathLike,
    ) -> tuple[Results, tuple[GameRecord, ...]]:
        from AlphaZeroCpp import InferenceClientParams, MCTS, new_root

        opponent_inference_path = inference_model_path(model_path)
        if not opponent_inference_path.is_file():
            raise ValueError(f'Opponent inference model does not exist: {opponent_inference_path}')
        if self.paired_schedule is None or self.args.evaluation is None:
            raise ValueError('A fixed paired opening suite is required for model evaluation.')

        opponent = MCTS(
            InferenceClientParams(self.device_id, str(opponent_inference_path), 16),
            self.mcts_args,
        )

        def opponent_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            assert self.args.evaluation is not None, 'Evaluation args must be set to use opponent evaluator'
            results = opponent.search(  # noqa: F821
                [(new_root(board.board.fen()), False) for board in boards]
            )
            return [action_probabilities(result.visits) for result in results.results]

        res = self.play_vs_evaluation_model_paired(
            opponent_evaluator,
            opponent_inference_path.name,
        )

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

        evaluation = self.args.evaluation
        if evaluation is None or evaluation.stockfish_binary_path is None:
            raise ValueError('Stockfish skill-level evaluation requires a configured binary path.')
        engine = chess.engine.SimpleEngine.popen_uci(evaluation.stockfish_binary_path)
        engine.configure({'Skill Level': level})
        engine.configure({'Threads': 1})  # Limit to one thread for consistency
        engine.configure({'Hash': 1024})  # Set hash size to 1GB

        def stockfish_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
            def get_stockfish_policy(board: CurrentBoard) -> np.ndarray:
                # Use Stockfish to get the best move
                result = engine.play(board.board, chess.engine.Limit(time=0.01, depth=level * 2))
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

    def play_vs_stockfish_fixed_nodes(
        self,
        binary_path: Path,
        nodes_per_move: int,
        threads: int,
        hash_mib: int,
    ) -> tuple[Results, tuple[GameRecord, ...], str]:
        if nodes_per_move < 1:
            raise ValueError('Stockfish nodes_per_move must be positive.')
        if threads < 1:
            raise ValueError('Stockfish threads must be positive.')
        if hash_mib < 1:
            raise ValueError('Stockfish hash_mib must be positive.')
        if not binary_path.is_file():
            raise ValueError(f'Stockfish binary does not exist: {binary_path}')

        engine = chess.engine.SimpleEngine.popen_uci(str(binary_path))
        try:
            if 'name' not in engine.id:
                raise ValueError('Stockfish did not report an engine name.')
            engine_name = engine.id['name']
            engine.configure(
                {
                    'Threads': threads,
                    'Hash': hash_mib,
                }
            )

            def stockfish_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
                policies: list[np.ndarray] = []
                for board in boards:
                    result = engine.play(
                        board.board,
                        chess.engine.Limit(nodes=nodes_per_move),
                        ponder=False,
                    )
                    if result.move is None:
                        raise ValueError('Stockfish did not return a move.')
                    move = normalize_move_for_action_space(result.move, board)
                    policies.append(CurrentGame.encode_moves([move], board))
                return policies

            results, records = self.play_vs_evaluation_model_paired(
                stockfish_evaluator,
                'stockfish_fixed_nodes',
            )
            return results, records, engine_name
        finally:
            engine.quit()

    def play_vs_evaluation_model(self, eval_model: EvaluationModel, name: str) -> Results:
        results, _ = self.play_vs_evaluation_model_paired(eval_model, name)
        return results

    def play_vs_evaluation_model_paired(
        self,
        eval_model: EvaluationModel,
        name: str,
    ) -> tuple[Results, tuple[GameRecord, ...]]:
        from AlphaZeroCpp import new_root, InferenceClientParams, MCTS

        if self.paired_schedule is None or self.args.evaluation is None:
            raise ValueError('A fixed paired opening suite is required for model evaluation.')

        current_inference_path = inference_model_path(model_save_path(self.iteration, self.args.save_path))
        if not current_inference_path.is_file():
            raise ValueError(f'Candidate inference model does not exist: {current_inference_path}')
        current = MCTS(
            InferenceClientParams(self.device_id, str(current_inference_path), 16),
            self.mcts_args,
        )

        def current_model(boards: list[CurrentBoard]) -> list[np.ndarray]:
            assert self.args.evaluation is not None, 'Evaluation args must be set to use opponent evaluator'
            results = current.search(  # noqa: F821
                [(new_root(board.board.fen()), False) for board in boards]
            )
            return [action_probabilities(result.visits) for result in results.results]

        results, records = _play_paired_models_search(
            iteration=self.iteration,
            candidate_model=current_model,
            opponent_model=eval_model,
            schedule=self.paired_schedule,
            maximum_game_plies=self.args.evaluation.maximum_game_plies,
            name=name,
        )

        del current  # Free the memory used by the MCTS client

        return results, records


if __name__ == '__main__':
    from src.settings import TRAINING_ARGS
    from src.eval.ModelEvaluationPy import EVAL_DEVICE

    evaluation = ModelEvaluation(0, TRAINING_ARGS, EVAL_DEVICE, 100, 400)
    print('Evaluation vs Random:', evaluation.play_vs_random())
