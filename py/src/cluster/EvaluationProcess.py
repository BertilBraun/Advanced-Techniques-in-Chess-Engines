import os
from torch import multiprocessing as mp

from src.eval.ModelEvaluation import ModelEvaluation
from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import log_scalar, TensorboardWriter
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.train.TrainingArgs import TrainingArgs
from src.util.save_paths import model_save_path
from src.util.tensorboard import log_scalars


def run_evaluation_process(run: int, args: TrainingArgs, iteration: int):
    evaluation_process = EvaluationProcess(args)
    with log_exceptions('Evaluation process'):
        evaluation_process.run(run, iteration)


def _eval_vs_dataset(run: int, model_evaluation: ModelEvaluation, iteration: int, dataset_path: str):
    with TensorboardWriter(run, 'evaluation', postfix_pid=False):
        dataset = SelfPlayDataset.load(dataset_path)
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


def _eval_vs_previous(run: int, model_evaluation: ModelEvaluation, iteration: int, save_path: str):
    if iteration < 2:
        return

    with TensorboardWriter(run, 'evaluation', postfix_pid=False):
        previous_model_path = model_save_path(iteration - 1, save_path)
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


def _eval_vs_five_previous(run: int, model_evaluation: ModelEvaluation, iteration: int, save_path: str):
    if iteration < 6:
        return

    with TensorboardWriter(run, 'evaluation', postfix_pid=False):
        previous_model_path = model_save_path(iteration - 5, save_path)
        results = model_evaluation.play_two_models_search(previous_model_path)
        log(f'Results after playing {iteration} vs {iteration - 5}:', results)

        log_scalars(
            'evaluation/vs_five_previous_model',
            {
                'wins': results.wins,
                'losses': results.losses,
                'draws': results.draws,
            },
            iteration,
        )


def _eval_vs_ten_previous(run: int, model_evaluation: ModelEvaluation, iteration: int, save_path: str):
    if iteration < 11:
        return

    with TensorboardWriter(run, 'evaluation', postfix_pid=False):
        previous_model_path = model_save_path(iteration - 10, save_path)
        results = model_evaluation.play_two_models_search(previous_model_path)
        log(f'Results after playing {iteration} vs {iteration - 10}:', results)

        log_scalars(
            'evaluation/vs_ten_previous_model',
            {
                'wins': results.wins,
                'losses': results.losses,
                'draws': results.draws,
            },
            iteration,
        )


def _eval_vs_reference(run: int, model_evaluation: ModelEvaluation, iteration: int, save_path: str):
    reference_model_path = save_path + '/reference_model.pt'
    if not os.path.exists(reference_model_path):
        log(f'Reference model not found at {reference_model_path}. Skipping evaluation.')
        return

    with TensorboardWriter(run, 'evaluation', postfix_pid=False):
        results = model_evaluation.play_two_models_search(reference_model_path)
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


def _eval_vs_random(run: int, model_evaluation: ModelEvaluation, iteration: int, _: str):
    with TensorboardWriter(run, 'evaluation', postfix_pid=False):
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


class EvaluationProcess:
    """This class provides functionallity to evaluate the model against itself and other models to collect performance metrics for the model. The results are logged to tensorboard."""

    def __init__(self, args: TrainingArgs) -> None:
        self.args = args
        self.eval_args = args.evaluation

    def run(self, run: int, iteration: int):
        """Play two most recent models against each other."""
        if not self.eval_args:
            return

        model_evaluation = ModelEvaluation(
            iteration,
            self.args,
            num_games=self.eval_args.num_games,
            num_searches_per_turn=self.eval_args.num_searches_per_turn,
        )

        processes: list[mp.Process] = []

        # Spawn subprocesses for each evaluation
        if self.eval_args.dataset_path:
            p = mp.Process(
                target=_eval_vs_dataset,
                args=(run, model_evaluation, iteration, self.eval_args.dataset_path),
            )
            p.start()
            processes.append(p)

        for fn in [
            _eval_vs_previous,
            _eval_vs_five_previous,
            _eval_vs_ten_previous,
            _eval_vs_reference,
            _eval_vs_random,
        ]:
            p = mp.Process(target=fn, args=(run, model_evaluation, iteration, self.args.save_path))
            p.start()
            processes.append(p)

        # Wait for all to finish
        for p in processes:
            p.join()


def evaluate_iteration(iteration: int, run_id: int):
    from src.settings import TRAINING_ARGS
    from src.util.save_paths import model_save_path

    assert TRAINING_ARGS.evaluation, 'Evaluation process is not enabled. Set evaluation.enabled to True in the config.'
    assert iteration > 0, 'Iteration must be greater than 0.'

    if not model_save_path(iteration, TRAINING_ARGS.save_path).exists():
        return

    TRAINING_ARGS.evaluation.num_games = 4

    log(f'Running evaluation process for iteration {iteration}')
    run_evaluation_process(run_id, TRAINING_ARGS, iteration)


def __main():
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')

    import torch  # noqa

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    from src.settings import get_run_id

    run_id = get_run_id()

    for i in range(1, 100):
        evaluate_iteration(i, run_id)


if __name__ == '__main__':
    __main()
