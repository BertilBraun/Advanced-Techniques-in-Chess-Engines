import src.environ_setup  # noqa # isort:skip # This import is necessary for setting up the environment variables

import argparse
import os
import random
from pathlib import Path
from time import monotonic

os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for MKL
# os.environ['TORCH_NUM_THREADS'] = '1'

# This ensures, that the seperate processes spawned by torch.multiprocessing do not interfere with each other by using more than one core. Since we are using as many processes as cores for workers, we need to limit the number of threads to 1 for each process. Otherwise, we would use more than one core per process, which would lead to a lot of context switching and slow down the training.

import numpy as np
import torch  # noqa


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-config', required=True, type=Path)
    parser.add_argument('--expected-source-revision', required=True)
    parser.add_argument('--approval-file', required=True, type=Path)
    return parser.parse_args()


if __name__ == '__main__':
    command_line_arguments = parse_arguments()
    os.environ['ALPHAZERO_EXPERIMENT_PATH'] = str(command_line_arguments.run_config.resolve())

    import torch.multiprocessing as mp

    mp.set_start_method('spawn')

    import torch  # noqa

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    from src.settings import EXPERIMENT, TRAINING_ARGS, USE_GPU, get_run_id
    from src.util.log import log
    from src.util.profiler import start_gpu_usage_logger
    from src.settings import TensorboardWriter, log_text
    from src.cluster.CommanderProcess import CommanderProcess
    from src.util.tensorboard import configure_tensorboard_run_directory
    from src.experiment.chess_experiment import write_resolved_experiment
    from src.experiment.chess_run import prepare_experiment_training_run
    from src.experiment.resource_telemetry import start_resource_telemetry
    from src.experiment.progress_telemetry import (
        RunOutcomeStatus,
        write_run_outcome,
    )

    experiment = EXPERIMENT
    run_configuration = experiment.run
    configure_tensorboard_run_directory(run_configuration.tensorboard_run_directory)

    random.seed(TRAINING_ARGS.random_seed)
    np.random.seed(TRAINING_ARGS.random_seed)
    torch.manual_seed(TRAINING_ARGS.random_seed)
    torch.cuda.manual_seed_all(TRAINING_ARGS.random_seed)

    log('Starting training')
    log('Training on:', 'GPU' if USE_GPU else 'CPU')
    log('Training args:')
    log(TRAINING_ARGS, use_pprint=True)

    run = get_run_id()
    log(f'Run ID: {run}')

    run_started_at = monotonic()
    manifest = prepare_experiment_training_run(
        experiment,
        command_line_arguments.expected_source_revision,
        command_line_arguments.approval_file,
    )
    write_resolved_experiment(Path(TRAINING_ARGS.save_path) / 'resolved-experiment.json', experiment)
    log('Resolved run manifest:')
    log(manifest.model_dump(), use_pprint=True)

    resource_telemetry = start_resource_telemetry(
        output_path=Path(TRAINING_ARGS.save_path),
        started_at=run_started_at,
        cost_currency=TRAINING_ARGS.limits.cost_currency,
        hourly_price=TRAINING_ARGS.limits.hourly_price,
        interval_seconds=TRAINING_ARGS.limits.resource_telemetry_interval_seconds,
    )

    gpu_usage_logger = start_gpu_usage_logger(run)

    with TensorboardWriter(run, 'training_args', postfix_pid=False):
        import pprint

        log_text('TrainingArgs', pprint.PrettyPrinter(indent=4).pformat(TRAINING_ARGS))

    if experiment.game == 'chess':
        from src.games.chess.training import ChessTrainingGame

        training_game = ChessTrainingGame(experiment)
    else:
        from src.games.go.training_runtime import GoTrainingGame

        training_game = GoTrainingGame(experiment)
    commander = CommanderProcess(run, training_game, run_started_at)
    training_results = commander.run()
    outcome_path = Path(TRAINING_ARGS.save_path) / 'run-outcome.json'
    try:
        for _ in training_results:
            pass
    except Exception as error:
        write_run_outcome(
            outcome_path,
            RunOutcomeStatus.FAILED,
            str(error),
            run_started_at,
            TRAINING_ARGS.limits.cost_currency,
            TRAINING_ARGS.limits.hourly_price,
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
        TRAINING_ARGS.limits.cost_currency,
        TRAINING_ARGS.limits.hourly_price,
        commander.latest_completed_model_version,
    )

    log('Training finished')
