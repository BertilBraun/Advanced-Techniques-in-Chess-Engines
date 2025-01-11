import asyncio

from src.eval.ModelEvaluation import ModelEvaluation
from src.settings import log_scalar
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

        model_evaluation = ModelEvaluation()
        results = await model_evaluation.play_two_models_search(
            iteration,
            iteration - self.args.every_n_iterations,
            self.args.num_games,
            self.args.num_searches_per_turn,
        )

        log(f'Results after playing two most recent models at iteration {iteration}:', results)

        log_scalar('win_loss_draw_vs_previous_model/wins', results.wins, iteration)
        log_scalar('win_loss_draw_vs_previous_model/losses', results.losses, iteration)
        log_scalar('win_loss_draw_vs_previous_model/draws', results.draws, iteration)

        results = await model_evaluation.play_vs_random(
            iteration,
            self.args.num_games,
            self.args.num_searches_per_turn,
        )
        log(f'Results after playing vs random at iteration {iteration}:', results)

        log_scalar('win_loss_draw_vs_random/wins', results.wins, iteration)
        log_scalar('win_loss_draw_vs_random/losses', results.losses, iteration)
        log_scalar('win_loss_draw_vs_random/draws', results.draws, iteration)
