import argparse
import os
import random
from pathlib import Path
from time import monotonic

os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for MKL
# os.environ['TORCH_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# This ensures, that the seperate processes spawned by torch.multiprocessing do not interfere with each other by using more than one core. Since we are using as many processes as cores for workers, we need to limit the number of threads to 1 for each process. Otherwise, we would use more than one core per process, which would lead to a lot of context switching and slow down the training.

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-config', required=True, type=Path)
    parser.add_argument('--expected-source-revision', required=True)
    parser.add_argument('--approval-file', required=True, type=Path)
    return parser.parse_args()


if __name__ == '__main__':
    command_line_arguments = parse_arguments()
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')

    import torch  # noqa

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    from src.experiment.configuration import (
        load_experiment_configuration,
        write_resolved_experiment,
    )
    from src.runtime import USE_GPU
    from src.util.log import log
    from src.util.profiler import start_gpu_usage_logger
    from src.cluster.CommanderProcess import CommanderProcess
    from src.util.tensorboard import (
        TensorboardWriter,
        configure_tensorboard_run_directory,
        get_run_id,
        log_text,
    )
    from src.experiment.run import prepare_experiment_training_run
    from src.experiment.resource_telemetry import start_resource_telemetry
    from src.experiment.progress_telemetry import (
        RunOutcomeStatus,
        write_run_outcome,
    )

    experiment = load_experiment_configuration(command_line_arguments.run_config.resolve())
    training = experiment.training
    configure_tensorboard_run_directory(experiment.run.tensorboard_run_directory)

    random.seed(training.random_seed)
    np.random.seed(training.random_seed)
    torch.manual_seed(training.random_seed)
    torch.cuda.manual_seed_all(training.random_seed)

    log('Starting training')
    log('Training on:', 'GPU' if USE_GPU else 'CPU')
    log('Training args:')
    log(training, use_pprint=True)

    run_id = get_run_id()
    log(f'Run ID: {run_id}')

    run_started_at = monotonic()
    manifest = prepare_experiment_training_run(
        experiment,
        command_line_arguments.expected_source_revision,
        command_line_arguments.approval_file,
    )
    write_resolved_experiment(Path(training.save_path) / 'resolved-experiment.json', experiment)
    log('Resolved run manifest:')
    log(manifest.model_dump(), use_pprint=True)

    resource_telemetry = start_resource_telemetry(
        output_path=Path(training.save_path),
        started_at=run_started_at,
        cost_currency=training.limits.cost_currency,
        hourly_price=training.limits.hourly_price,
        interval_seconds=training.limits.resource_telemetry_interval_seconds,
    )

    gpu_usage_logger = start_gpu_usage_logger(run_id)

    with TensorboardWriter(run_id, 'training_args', postfix_pid=False):
        import pprint

        log_text('TrainingArgs', pprint.PrettyPrinter(indent=4).pformat(training))

    if experiment.game == 'chess':
        from src.games.chess.training import ChessImplementation

        game = ChessImplementation(experiment)
    else:
        from src.games.go.training_runtime import GoImplementation

        game = GoImplementation(experiment)
    commander = CommanderProcess(run_id, game, run_started_at)
    outcome_path = Path(training.save_path) / 'run-outcome.json'
    try:
        commander.run()
    except Exception as error:
        write_run_outcome(
            outcome_path,
            RunOutcomeStatus.FAILED,
            str(error),
            run_started_at,
            training.limits.cost_currency,
            training.limits.hourly_price,
            commander.latest_completed_model_version,
        )
        raise
    finally:
        gpu_usage_logger.stop()
        resource_telemetry.stop()

    outcome_status = RunOutcomeStatus.STOPPED if commander.final_stop_reason is not None else RunOutcomeStatus.COMPLETED
    write_run_outcome(
        outcome_path,
        outcome_status,
        commander.final_stop_reason,
        run_started_at,
        training.limits.cost_currency,
        training.limits.hourly_price,
        commander.latest_completed_model_version,
    )

    log('Training finished')
