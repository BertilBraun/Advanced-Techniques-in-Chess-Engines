from pathlib import Path
import subprocess
import torch
from torch import multiprocessing as mp

from src.eval.ModelEvaluationCpp import ModelEvaluation
from src.cluster.CudaProcess import start_process_on_cuda_device
from src.eval.ModelEvaluationPy import Results

from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import log_scalar, TensorboardWriter
from src.settings_common import USE_GPU
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.train.TrainingArgs import TrainingArgs
from src.util.save_paths import inference_model_path, model_save_path
from src.util.tensorboard import log_scalars
from src.experiment.evaluation_protocol import (
    EngineCondition,
    EngineSetting,
    MatchConditions,
    MatchReport,
    file_sha256,
    summarize_match,
    write_match_report,
)
from src.experiment.evaluation_schedule import (
    evaluation_device_for_task,
    select_historical_model_iterations,
)
from src.experiment.run_configuration import RunManifest


SOURCE_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_PROCESS_POLL_SECONDS = 1.0


def _activate_evaluation_device(model_evaluation: ModelEvaluation) -> None:
    if USE_GPU:
        torch.cuda.set_device(model_evaluation.device_id)


def _reap_evaluation_tasks(processes: list[mp.Process]) -> None:
    active_processes: list[mp.Process] = []
    failed_processes: list[mp.Process] = []
    for process in processes:
        if process.is_alive():
            active_processes.append(process)
            continue
        process.join()
        if process.exitcode != 0:
            failed_processes.append(process)
    processes[:] = active_processes
    if failed_processes:
        failures = ', '.join(f'{process.pid}: {process.exitcode}' for process in failed_processes)
        raise RuntimeError(f'Evaluation tasks exited unsuccessfully: {failures}.')


def _wait_for_evaluation_slot(processes: list[mp.Process], maximum_processes: int) -> None:
    while len(processes) >= maximum_processes:
        processes[0].join(timeout=EVALUATION_PROCESS_POLL_SECONDS)
        _reap_evaluation_tasks(processes)


def _wait_for_all_evaluation_tasks(processes: list[mp.Process]) -> None:
    while processes:
        processes[0].join(timeout=EVALUATION_PROCESS_POLL_SECONDS)
        _reap_evaluation_tasks(processes)


def _terminate_evaluation_tasks(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    processes.clear()


def _result_metrics(results: Results) -> dict[str, float | int]:
    total_games = results.wins + results.draws + results.losses
    if total_games <= 0:
        raise ValueError('Evaluation results must contain at least one game.')
    return {
        'wins': results.wins,
        'losses': results.losses,
        'draws': results.draws,
        'score': (results.wins + 0.5 * results.draws) / total_games,
    }


def _model_engine_condition(
    model_evaluation: ModelEvaluation,
    artifact_sha256: str,
    identifier: str,
) -> EngineCondition:
    evaluation = model_evaluation.args.evaluation
    if evaluation is None:
        raise ValueError('Evaluation settings are required for model engine provenance.')
    direct_inference = evaluation.direct_inference
    settings = (
        (
            EngineSetting(name='InferenceScheduler', value='direct_multi_tree'),
            EngineSetting(name='InferenceWorkers', value=str(direct_inference.inference_workers)),
            EngineSetting(name='InferenceBatchSize', value=str(direct_inference.inference_batch_size)),
            EngineSetting(
                name='OutstandingBatchesPerWorker',
                value=str(direct_inference.outstanding_batches_per_worker),
            ),
            EngineSetting(name='ParallelSearchesPerTree', value=str(evaluation.parallel_searches)),
        )
        if direct_inference is not None
        else ()
    )
    return EngineCondition(
        artifact_sha256=artifact_sha256,
        identifier=identifier,
        search_limit_name='mcts_root_visits',
        search_limit_value=model_evaluation.num_searches_per_turn,
        threads=1 if direct_inference is not None else evaluation.mcts_threads,
        settings=settings,
    )


def _evaluation_source_revision() -> str:
    completed = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=SOURCE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_evaluation_process(
    run: int,
    args: TrainingArgs,
    iteration: int,
    metrics_step: int | None = None,
) -> None:
    evaluation_process = EvaluationProcess(args)
    with log_exceptions('Evaluation process'):
        evaluation_process.run(run, iteration, metrics_step)


def _eval_vs_dataset(
    run: int,
    model_evaluation: ModelEvaluation,
    iteration: int,
    dataset_path: str,
    metrics_step: int,
) -> None:
    _activate_evaluation_device(model_evaluation)
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
            metrics_step,
        )
        log_scalar('evaluation/value_mse_loss', avg_value_loss, metrics_step)


def _eval_vs_previous(
    run: int,
    model_evaluation: ModelEvaluation,
    iteration: int,
    save_path: str,
    how_many_previous: int,
    metrics_step: int,
) -> None:
    _activate_evaluation_device(model_evaluation)
    previous_model_path = model_save_path(iteration - how_many_previous, save_path)
    if not previous_model_path.exists():
        return

    with TensorboardWriter(run, f'evaluation_vs_{how_many_previous}_previous_model', postfix_pid=False):
        results = model_evaluation.play_two_models_search(previous_model_path)
        log(f'Results after playing {iteration} vs {iteration - how_many_previous}:', results)

        log_scalars(
            f'evaluation/vs_{how_many_previous}_previous_model',
            _result_metrics(results),
            metrics_step,
        )


def _eval_vs_iteration(
    run: int,
    model_evaluation: ModelEvaluation,
    iteration: int,
    save_path: str,
    current_iteration: int,
    metrics_step: int,
) -> None:
    _activate_evaluation_device(model_evaluation)
    model_path = model_save_path(iteration, save_path)
    if not inference_model_path(model_path).exists():
        return

    with TensorboardWriter(run, f'evaluation_vs_{iteration}_model', postfix_pid=False):
        results = model_evaluation.play_two_models_search(model_path)
        log(f'Results after playing vs model {iteration} at iteration {current_iteration}:', results)

        log_scalars(
            f'evaluation/vs_{iteration}_model',
            _result_metrics(results),
            metrics_step,
        )


def _eval_vs_reference(
    run: int,
    model_evaluation: ModelEvaluation,
    iteration: int,
    args: TrainingArgs,
    metrics_step: int,
) -> None:
    _activate_evaluation_device(model_evaluation)
    evaluation = args.evaluation
    if evaluation is None or evaluation.reference_model_path is None:
        return
    if evaluation.opening_suite_path is None or evaluation.raw_results_path is None:
        raise ValueError('Reference evaluation requires opening and raw-results paths.')

    reference_model_path = Path(evaluation.reference_model_path)
    candidate_model_path = model_save_path(iteration, args.save_path)
    reference_inference_path = inference_model_path(reference_model_path)
    candidate_inference_path = inference_model_path(candidate_model_path)
    run_manifest_path = Path(args.save_path) / 'run_manifest.json'
    run_manifest = RunManifest.model_validate_json(run_manifest_path.read_text(encoding='utf-8'))

    results, records = model_evaluation.play_two_models_paired(reference_model_path)
    summary = summarize_match(
        records,
        bootstrap_seed=evaluation.bootstrap_seed,
        bootstrap_samples=evaluation.bootstrap_samples,
    )
    with TensorboardWriter(run, 'evaluation_vs_reference', postfix_pid=False):
        log(f'Results after playing the current vs the reference at iteration {iteration}:', results)

        log_scalars(
            'evaluation/vs_reference_model',
            {
                **_result_metrics(results),
                'score_confidence_low': summary.score_confidence_low,
                'score_confidence_high': summary.score_confidence_high,
            },
            metrics_step,
        )

    opening_suite_path = Path(evaluation.opening_suite_path)
    condition = _model_engine_condition(
        model_evaluation,
        file_sha256(candidate_inference_path),
        f'candidate-model-{iteration}',
    )
    report = MatchReport(
        conditions=MatchConditions(
            source_revision=run_manifest.source_revision,
            evaluation_source_revision=_evaluation_source_revision(),
            opening_suite_path=str(opening_suite_path),
            opening_suite_sha256=file_sha256(opening_suite_path),
            candidate=condition,
            opponent=_model_engine_condition(
                model_evaluation,
                file_sha256(reference_inference_path),
                'archived-best-model-reexported-as-model-0',
            ),
            maximum_game_plies=evaluation.maximum_game_plies,
            bootstrap_seed=evaluation.bootstrap_seed,
            bootstrap_samples=evaluation.bootstrap_samples,
        ),
        summary=summary,
        games=records,
    )
    report_path = Path(evaluation.raw_results_path) / f'match-vs-archive-iteration-{iteration}.json'
    write_match_report(report_path, report)


def _eval_vs_stockfish_fixed(
    run: int,
    model_evaluation: ModelEvaluation,
    iteration: int,
    args: TrainingArgs,
    metrics_step: int,
) -> None:
    _activate_evaluation_device(model_evaluation)
    evaluation = args.evaluation
    if evaluation is None or evaluation.stockfish_binary_path is None:
        return
    if evaluation.opening_suite_path is None or evaluation.raw_results_path is None:
        raise ValueError('Stockfish evaluation requires opening and raw-results paths.')

    stockfish_binary_path = Path(evaluation.stockfish_binary_path)
    candidate_model_path = model_save_path(iteration, args.save_path)
    candidate_inference_path = inference_model_path(candidate_model_path)
    run_manifest_path = Path(args.save_path) / 'run_manifest.json'
    run_manifest = RunManifest.model_validate_json(run_manifest_path.read_text(encoding='utf-8'))
    if run_manifest.stockfish_binary_sha256 is None:
        raise ValueError('Run manifest does not identify the Stockfish binary.')

    results, records, engine_name = model_evaluation.play_vs_stockfish_fixed_nodes(
        binary_path=stockfish_binary_path,
        nodes_per_move=evaluation.stockfish_nodes_per_move,
        threads=evaluation.stockfish_threads,
        hash_mib=evaluation.stockfish_hash_mib,
    )
    summary = summarize_match(
        records,
        bootstrap_seed=evaluation.bootstrap_seed,
        bootstrap_samples=evaluation.bootstrap_samples,
    )
    with TensorboardWriter(run, 'evaluation_vs_stockfish_fixed_nodes', postfix_pid=False):
        log(f'Results after playing the current model vs {engine_name}:', results)
        log_scalars(
            'evaluation/vs_stockfish_fixed_nodes',
            {
                **_result_metrics(results),
                'score_confidence_low': summary.score_confidence_low,
                'score_confidence_high': summary.score_confidence_high,
            },
            metrics_step,
        )

    opening_suite_path = Path(evaluation.opening_suite_path)
    report = MatchReport(
        conditions=MatchConditions(
            source_revision=run_manifest.source_revision,
            evaluation_source_revision=_evaluation_source_revision(),
            opening_suite_path=str(opening_suite_path),
            opening_suite_sha256=file_sha256(opening_suite_path),
            candidate=_model_engine_condition(
                model_evaluation,
                file_sha256(candidate_inference_path),
                f'candidate-model-{iteration}',
            ),
            opponent=EngineCondition(
                artifact_sha256=run_manifest.stockfish_binary_sha256,
                identifier=engine_name,
                search_limit_name='nodes_per_move',
                search_limit_value=evaluation.stockfish_nodes_per_move,
                threads=evaluation.stockfish_threads,
                settings=(
                    EngineSetting(name='Hash', value=str(evaluation.stockfish_hash_mib)),
                    EngineSetting(name='Ponder', value='false'),
                ),
            ),
            maximum_game_plies=evaluation.maximum_game_plies,
            bootstrap_seed=evaluation.bootstrap_seed,
            bootstrap_samples=evaluation.bootstrap_samples,
        ),
        summary=summary,
        games=records,
    )
    report_path = Path(evaluation.raw_results_path) / f'match-vs-stockfish-fixed-nodes-iteration-{iteration}.json'
    write_match_report(report_path, report)


def _eval_vs_random(
    run: int,
    model_evaluation: ModelEvaluation,
    iteration: int,
    _: str,
    metrics_step: int,
) -> None:
    _activate_evaluation_device(model_evaluation)
    with TensorboardWriter(run, 'evaluation_vs_random', postfix_pid=False):
        results = model_evaluation.play_vs_random()
        log(f'Results after playing vs random at iteration {iteration}:', results)

        log_scalars(
            'evaluation/vs_random',
            _result_metrics(results),
            metrics_step,
        )


def _eval_policy_vs_random(
    run: int,
    model_evaluation: ModelEvaluation,
    iteration: int,
    save_path: str,
    metrics_step: int,
) -> None:
    _activate_evaluation_device(model_evaluation)
    with TensorboardWriter(run, 'evaluation_policy_vs_random', postfix_pid=False):
        results = model_evaluation.play_policy_vs_random()
        log(f'Results after playing the current policy only vs random at iteration {iteration}:', results)

        log_scalars(
            'evaluation/policy_vs_random',
            _result_metrics(results),
            metrics_step,
        )


def _eval_vs_stockfish(
    run: int,
    model_evaluation: ModelEvaluation,
    level: int,
    iteration: int,
    metrics_step: int,
) -> None:
    _activate_evaluation_device(model_evaluation)
    with TensorboardWriter(run, f'evaluation_vs_stockfish_level_{level}', postfix_pid=False):
        results = model_evaluation.play_vs_stockfish(level)
        log(f'Results after playing vs stockfish level {level} at iteration {iteration}:', results)

        log_scalars(
            f'evaluation/vs_stockfish_level_{level}',
            _result_metrics(results),
            metrics_step,
        )


class EvaluationProcess:
    """This class provides functionallity to evaluate the model against itself and other models to collect performance metrics for the model. The results are logged to tensorboard."""

    def __init__(self, args: TrainingArgs) -> None:
        self.args = args
        self.eval_args = args.evaluation

    def run(self, run: int, iteration: int, metrics_step: int | None = None) -> None:
        """Play two most recent models against each other."""
        if not self.eval_args:
            return

        processes: list[mp.Process] = []
        resolved_metrics_step = iteration if metrics_step is None else metrics_step
        max_concurrent_tasks = self.eval_args.max_concurrent_tasks
        evaluation_task_index = 0

        def create_model_evaluation() -> tuple[ModelEvaluation, int]:
            nonlocal evaluation_task_index
            device_cycle = self.args.cluster.evaluation_device_cycle
            physical_device_id = evaluation_device_for_task(device_cycle, evaluation_task_index) if USE_GPU else 0
            evaluation_task_index += 1
            return (
                ModelEvaluation(
                    iteration,
                    self.args,
                    device_id=0,
                    num_games=self.eval_args.num_games,
                    num_searches_per_turn=self.eval_args.num_searches_per_turn,
                ),
                physical_device_id,
            )

        def start_process(process: mp.Process, physical_device_id: int) -> None:
            _reap_evaluation_tasks(processes)
            _wait_for_evaluation_slot(processes, max_concurrent_tasks)
            if USE_GPU:
                start_process_on_cuda_device(process, physical_device_id)
            else:
                process.start()
            processes.append(process)

        try:
            # Spawn subprocesses for each evaluation
            if self.eval_args.dataset_path:
                model_evaluation, physical_device_id = create_model_evaluation()
                p = mp.Process(
                    target=_eval_vs_dataset,
                    args=(run, model_evaluation, iteration, self.eval_args.dataset_path, resolved_metrics_step),
                )
                start_process(p, physical_device_id)

            for how_many_previous in self.eval_args.previous_model_offsets:
                model_evaluation, physical_device_id = create_model_evaluation()
                p = mp.Process(
                    target=_eval_vs_previous,
                    args=(
                        run,
                        model_evaluation,
                        iteration,
                        self.args.save_path,
                        how_many_previous,
                        resolved_metrics_step,
                    ),
                )
                start_process(p, physical_device_id)

            if self.eval_args.reference_model_path is not None:
                model_evaluation, physical_device_id = create_model_evaluation()
                p = mp.Process(
                    target=_eval_vs_reference,
                    args=(run, model_evaluation, iteration, self.args, resolved_metrics_step),
                )
                start_process(p, physical_device_id)

            if self.eval_args.stockfish_binary_path is not None:
                model_evaluation, physical_device_id = create_model_evaluation()
                p = mp.Process(
                    target=_eval_vs_stockfish_fixed,
                    args=(run, model_evaluation, iteration, self.args, resolved_metrics_step),
                )
                start_process(p, physical_device_id)

            if self.eval_args.evaluate_random:
                for evaluation_function in [_eval_vs_random, _eval_policy_vs_random]:
                    model_evaluation, physical_device_id = create_model_evaluation()
                    p = mp.Process(
                        target=evaluation_function,
                        args=(
                            run,
                            model_evaluation,
                            iteration,
                            self.args.save_path,
                            resolved_metrics_step,
                        ),
                    )
                    start_process(p, physical_device_id)

            historical_model_iterations = select_historical_model_iterations(
                iteration,
                self.eval_args.historical_model_iterations,
                self.args.artifact_retention.milestone_inference_interval,
                self.eval_args.historical_model_rotation_period,
                self.eval_args.every_n_iterations,
            )
            for historical_iteration in historical_model_iterations:
                model_evaluation, physical_device_id = create_model_evaluation()
                p = mp.Process(
                    target=_eval_vs_iteration,
                    args=(
                        run,
                        model_evaluation,
                        historical_iteration,
                        self.args.save_path,
                        iteration,
                        resolved_metrics_step,
                    ),
                )
                start_process(p, physical_device_id)

            for level in self.eval_args.stockfish_skill_levels:
                model_evaluation, physical_device_id = create_model_evaluation()
                p = mp.Process(
                    target=_eval_vs_stockfish,
                    args=(run, model_evaluation, level, iteration, resolved_metrics_step),
                )
                start_process(p, physical_device_id)

            _wait_for_all_evaluation_tasks(processes)
        except BaseException:
            _terminate_evaluation_tasks(processes)
            raise


def evaluate_iteration(iteration: int, run_id: int) -> None:
    from src.settings import TRAINING_ARGS
    from src.util.save_paths import model_save_path

    assert TRAINING_ARGS.evaluation, 'Evaluation process is not enabled. Set evaluation.enabled to True in the config.'
    assert iteration > 0, 'Iteration must be greater than 0.'

    if not model_save_path(iteration, TRAINING_ARGS.save_path).exists():
        return

    log(f'Running evaluation process for iteration {iteration}')
    run_evaluation_process(run_id, TRAINING_ARGS, iteration)


def __main() -> None:
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
