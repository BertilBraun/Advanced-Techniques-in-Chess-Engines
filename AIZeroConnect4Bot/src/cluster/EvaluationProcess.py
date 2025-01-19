import asyncio
import numpy as np

from src.eval.ModelEvaluation import ModelEvaluation
from src.settings import log_scalar, CurrentGame, CurrentBoard
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.alpha_zero.train.TrainingArgs import EvaluationParams


def run_evaluation_process(args: EvaluationParams, iteration: int):
    evaluation_process = EvaluationProcess(args)
    with log_exceptions('Evaluation process'):
        asyncio.run(evaluation_process.run(iteration))


class EvaluationProcess:
    def __init__(self, args: EvaluationParams) -> None:
        self.args = args

    async def run(self, iteration: int):
        """Play two most recent models against each other."""
        if not self.args or iteration % self.args.every_n_iterations != 0 or iteration == 0:
            return

        model_evaluation = ModelEvaluation(
            iteration,
            self.args.num_games,
            self.args.num_searches_per_turn,
        )
        results = await model_evaluation.play_two_models_search(iteration - self.args.every_n_iterations)

        log(f'Results after playing two most recent models at iteration {iteration}:', results)

        log_scalar('win_loss_draw_vs_previous_model/wins', results.wins, iteration)
        log_scalar('win_loss_draw_vs_previous_model/losses', results.losses, iteration)
        log_scalar('win_loss_draw_vs_previous_model/draws', results.draws, iteration)

        results = await model_evaluation.play_vs_random()
        log(f'Results after playing vs random at iteration {iteration}:', results)

        log_scalar('win_loss_draw_vs_random/wins', results.wins, iteration)
        log_scalar('win_loss_draw_vs_random/losses', results.losses, iteration)
        log_scalar('win_loss_draw_vs_random/draws', results.draws, iteration)

        from src.games.chess.ChessGame import ChessGame
        from src.games.chess.ChessComparisonBot import ChessComparisonBot

        if isinstance(CurrentGame, ChessGame):
            comparison_bot = ChessComparisonBot()

            async def comparison_bot_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
                return [CurrentGame.encode_moves([comparison_bot.think(board)]) for board in boards]  # type: ignore

            results = await model_evaluation.play_vs_evaluation_model(comparison_bot_evaluator)
            log(f'Results after playing vs comparison bot at iteration {iteration}:', results)

            log_scalar('win_loss_draw_vs_comparison_bot/wins', results.wins, iteration)
            log_scalar('win_loss_draw_vs_comparison_bot/losses', results.losses, iteration)
            log_scalar('win_loss_draw_vs_comparison_bot/draws', results.draws, iteration)
