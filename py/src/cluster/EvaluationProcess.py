import os
from torch import multiprocessing as mp
import torch

from src.eval.ModelEvaluationCpp import ModelEvaluation

from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import log_scalar, TensorboardWriter
from src.settings_common import USE_GPU
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
    with TensorboardWriter(run, 'evaluation_dataset', postfix_pid=False):
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


def _eval_vs_previous(
    run: int,
    model_evaluation: ModelEvaluation,
    iteration: int,
    save_path: str,
    how_many_previous: int,
):
    previous_model_path = model_save_path(iteration - how_many_previous, save_path)
    if not previous_model_path.exists():
        return

    with TensorboardWriter(run, f'evaluation_vs_{how_many_previous}_previous_model', postfix_pid=False):
        results = model_evaluation.play_two_models_search(previous_model_path)
        log(f'Results after playing {iteration} vs {iteration - how_many_previous}:', results)

        log_scalars(
            f'evaluation/vs_{how_many_previous}_previous_model',
            {
                'wins': results.wins,
                'losses': results.losses,
                'draws': results.draws,
            },
            iteration,
        )


def _eval_vs_iteration(
    run: int, model_evaluation: ModelEvaluation, iteration: int, save_path: str, current_iteration: int
):
    model_path = model_save_path(iteration, save_path)
    if not model_path.exists():
        return

    with TensorboardWriter(run, f'evaluation_vs_{iteration}_model', postfix_pid=False):
        results = model_evaluation.play_two_models_search(model_path)
        log(f'Results after playing vs model {iteration} at iteration {current_iteration}:', results)

        log_scalars(
            f'evaluation/vs_{iteration}_model',
            {
                'wins': results.wins,
                'losses': results.losses,
                'draws': results.draws,
            },
            current_iteration,
        )


def _eval_vs_reference(run: int, model_evaluation: ModelEvaluation, iteration: int, save_path: str):
    reference_model_path = save_path + '/reference_model.pt'
    if not os.path.exists(reference_model_path):
        return

    with TensorboardWriter(run, 'evaluation_vs_reference', postfix_pid=False):
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
    with TensorboardWriter(run, 'evaluation_vs_random', postfix_pid=False):
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


def _eval_policy_vs_random(run: int, model_evaluation: ModelEvaluation, iteration: int, save_path: str):
    with TensorboardWriter(run, 'evaluation_policy_vs_random', postfix_pid=False):
        results = model_evaluation.play_policy_vs_random()
        log(f'Results after playing the current policy only vs random at iteration {iteration}:', results)

        log_scalars(
            'evaluation/policy_vs_random',
            {
                'wins': results.wins,
                'losses': results.losses,
                'draws': results.draws,
            },
            iteration,
        )


def _eval_vs_stockfish(run: int, model_evaluation: ModelEvaluation, level: int, iteration: int):
    with TensorboardWriter(run, f'evaluation_vs_stockfish_level_{level}', postfix_pid=False):
        results = model_evaluation.play_vs_stockfish(level)
        log(f'Results after playing vs stockfish level {level} at iteration {iteration}:', results)

        log_scalars(
            f'evaluation/vs_stockfish_level_{level}',
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
            device_id=torch.cuda.device_count() - 1 if USE_GPU else 0,
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

        for how_many_previous in [5, 10]:
            p = mp.Process(
                target=_eval_vs_previous,
                args=(run, model_evaluation, iteration, self.args.save_path, how_many_previous),
            )
            p.start()
            processes.append(p)

        for fn in [_eval_vs_reference, _eval_vs_random, _eval_policy_vs_random]:
            p = mp.Process(target=fn, args=(run, model_evaluation, iteration, self.args.save_path))
            p.start()
            processes.append(p)

        for iter in range(10, self.args.num_iterations + 1, 10):
            if iter >= iteration:
                continue
            p = mp.Process(
                target=_eval_vs_iteration, args=(run, model_evaluation, iter, self.args.save_path, iteration)
            )
            p.start()
            processes.append(p)

        for level in (0, 1, 2, 3):
            p = mp.Process(target=_eval_vs_stockfish, args=(run, model_evaluation, level, iteration))
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
