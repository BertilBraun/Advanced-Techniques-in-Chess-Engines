import asyncio
import numpy as np

from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.eval.ModelEvaluation import ModelEvaluation
from src.settings import log_scalar, CurrentGame, CurrentBoard, TensorboardWriter
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.train.TrainingArgs import TrainingArgs


def run_evaluation_process(args: TrainingArgs, iteration: int):
    evaluation_process = EvaluationProcess(args)
    with log_exceptions('Evaluation process'), TensorboardWriter('evaluation'):
        asyncio.run(evaluation_process.run(iteration))


class EvaluationProcess:
    """This class provides functionallity to evaluate the model against itself and other models to collect performance metrics for the model. The results are logged to tensorboard."""

    def __init__(self, args: TrainingArgs) -> None:
        assert args.evaluation, 'Evaluation parameters must be set.'
        self.args = args
        self.eval_args = args.evaluation

    async def run(self, iteration: int):
        """Play two most recent models against each other."""
        if not self.args or iteration % self.eval_args.every_n_iterations != 0 or iteration == 0:
            return

        model_evaluation = ModelEvaluation(
            iteration,
            self.args,
            self.eval_args.num_games,
            self.eval_args.num_searches_per_turn,
        )

        dataset = SelfPlayDataset.load(self.eval_args.dataset_path)
        (
            policy_accuracy_at_1,
            policy_accuracy_at_5,
            policy_accuracy_at_10,
            avg_value_loss,
        ) = model_evaluation.evaluate_model_vs_dataset(dataset)
        log(f'Evaluation results at iteration {iteration}:')
        log(f'    Policy accuracy @1: {policy_accuracy_at_1}')
        log(f'    Policy accuracy @5: {policy_accuracy_at_5}')
        log(f'    Policy accuracy @10: {policy_accuracy_at_10}')
        log(f'    Avg value loss: {avg_value_loss}')

        log_scalar('policy_accuracy@1', policy_accuracy_at_1, iteration)
        log_scalar('policy_accuracy@5', policy_accuracy_at_5, iteration)
        log_scalar('policy_accuracy@10', policy_accuracy_at_10, iteration)
        log_scalar('value_mse_loss', avg_value_loss, iteration)

        results = await model_evaluation.play_two_models_search(iteration - self.eval_args.every_n_iterations)

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

        if isinstance(CurrentGame, ChessGame):
            from src.games.chess.comparison_bots.HandcraftedBotV3 import HandcraftedBotV3

            comparison_bot = HandcraftedBotV3()

            async def comparison_bot_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
                return [CurrentGame.encode_moves([comparison_bot.think(board.board)]) for board in boards]  # type: ignore

            results = await model_evaluation.play_vs_evaluation_model(comparison_bot_evaluator)
            log(f'Results after playing vs comparison bot at iteration {iteration}:', results)

            # TODO for StockfishBot you have to call .cleanup() before quitting the process

            log_scalar('win_loss_draw_vs_comparison_bot/wins', results.wins, iteration)
            log_scalar('win_loss_draw_vs_comparison_bot/losses', results.losses, iteration)
            log_scalar('win_loss_draw_vs_comparison_bot/draws', results.draws, iteration)
