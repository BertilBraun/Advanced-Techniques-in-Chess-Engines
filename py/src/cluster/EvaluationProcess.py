import numpy as np

from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.eval.ModelEvaluation import ModelEvaluation
from src.settings import log_scalar, CurrentGame, CurrentBoard, TensorboardWriter
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.train.TrainingArgs import TrainingArgs
from src.util.save_paths import model_save_path
from src.util.tensorboard import log_scalars


def run_evaluation_process(run: int, args: TrainingArgs, iteration: int):
    evaluation_process = EvaluationProcess(args)
    with log_exceptions('Evaluation process'), TensorboardWriter(run, 'evaluation', postfix_pid=False):
        evaluation_process.run(iteration)


class EvaluationProcess:
    """This class provides functionallity to evaluate the model against itself and other models to collect performance metrics for the model. The results are logged to tensorboard."""

    def __init__(self, args: TrainingArgs) -> None:
        self.args = args
        self.eval_args = args.evaluation

    def run(self, iteration: int):
        """Play two most recent models against each other."""
        if not self.eval_args or iteration % self.eval_args.every_n_iterations != 0 or iteration == 0:
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
        log(f'    Policy accuracy @1: {policy_accuracy_at_1:.2%}')
        log(f'    Policy accuracy @5: {policy_accuracy_at_5:.2%}')
        log(f'    Policy accuracy @10: {policy_accuracy_at_10:.2%}')
        log(f'    Avg value loss: {avg_value_loss}')

        log_scalars(
            'evaluation/policy_accuracy',
            {
                '1': policy_accuracy_at_1,
                '5': policy_accuracy_at_5,
                '10': policy_accuracy_at_10,
            },
            iteration,
        )

        log_scalar('evaluation/value_mse_loss', avg_value_loss, iteration)

        previous_model_path = model_save_path(iteration - self.eval_args.every_n_iterations, self.args.save_path)
        results = model_evaluation.play_two_models_search(previous_model_path)

        log(f'Results after playing two most recent models at iteration {iteration}:', results)

        log_scalars(
            'evaluation/vs_previous_model',
            {
                'wins': results.wins,
                'losses': results.losses,
                'draws': results.draws,
            },
            iteration,
        )

        first_model_path = self.args.save_path + '/reference_model.pt'
        results = model_evaluation.play_two_models_search(first_model_path)

        log(f'Results after playing the current vs the reference at iteration {iteration}:', results)

        log_scalars(
            'evaluation/vs_reference_model',
            {
                'wins': results.wins,
                'losses': results.losses,
                'draws': results.draws,
            },
            iteration,
        )

        results = model_evaluation.play_vs_random()
        log(f'Results after playing vs random at iteration {iteration}:', results)

        log_scalars(
            'evaluation/vs_random',
            {
                'wins': results.wins,
                'losses': results.losses,
                'draws': results.draws,
            },
            iteration,
        )

        from src.games.chess.ChessGame import ChessGame

        if isinstance(CurrentGame, ChessGame) and False:
            from src.games.chess.comparison_bots.HandcraftedBotV4 import HandcraftedBotV4

            comparison_bot = HandcraftedBotV4()

            def comparison_bot_evaluator(boards: list[CurrentBoard]) -> list[np.ndarray]:
                comparison_bot.restart_clock()
                return [CurrentGame.encode_moves([comparison_bot.think(board)]) for board in boards]

            results = model_evaluation.play_vs_evaluation_model(
                comparison_bot_evaluator, 'comparison_bot', self.eval_args.num_games // 2
            )
            log(f'Results after playing vs comparison bot at iteration {iteration}:', results)

            # TODO for StockfishBot you have to call .cleanup() before quitting the process

            log_scalars(
                'evaluation/vs_comparison_bot',
                {
                    'wins': results.wins,
                    'losses': results.losses,
                    'draws': results.draws,
                },
                iteration,
            )


def evaluate_iteration(iteration):
    from src.settings import TRAINING_ARGS
    from src.util.save_paths import model_save_path

    if not model_save_path(iteration, TRAINING_ARGS.save_path).exists():
        return
    log(f'Running evaluation process for iteration {iteration}')
    run_evaluation_process(run_id, TRAINING_ARGS, iteration)


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')

    import torch  # noqa

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    from src.settings import get_run_id

    run_id = get_run_id()

    with mp.Pool(processes=4) as pool:
        pool.map(evaluate_iteration, range(1, 100))
