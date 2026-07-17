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


torch.autograd.set_detect_anomaly(True)


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

    from src.settings import TRAINING_ARGS, USE_GPU, get_run_id
    from src.util.log import log
    from src.util.profiler import start_gpu_usage_logger
    from src.settings import TensorboardWriter, log_text
    from src.cluster.CommanderProcess import CommanderProcess
    from src.experiment.run_configuration import (
        apply_run_configuration,
        load_run_configuration,
        prepare_training_run,
    )
    from src.experiment.resource_telemetry import start_resource_telemetry
    from src.experiment.progress_telemetry import (
        RunOutcomeStatus,
        write_run_outcome,
    )

    run_configuration = load_run_configuration(command_line_arguments.run_config)
    apply_run_configuration(TRAINING_ARGS, run_configuration)

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
    manifest = prepare_training_run(
        TRAINING_ARGS,
        run_configuration,
        command_line_arguments.expected_source_revision,
        command_line_arguments.approval_file,
    )
    log('Resolved run manifest:')
    log(manifest.model_dump(), use_pprint=True)

    start_resource_telemetry(
        output_path=Path(TRAINING_ARGS.save_path),
        started_at=run_started_at,
        hourly_price_eur=run_configuration.budget.hourly_price_eur,
        interval_seconds=run_configuration.safety.telemetry_interval_seconds,
    )

    start_gpu_usage_logger(run)

    # if a function on_startup is defined, call it
    if TRAINING_ARGS.on_startup is not None:
        log('Calling on_startup function...')
        TRAINING_ARGS.on_startup()

    with TensorboardWriter(run, 'training_args', postfix_pid=False):
        import pprint

        log_text('TrainingArgs', pprint.PrettyPrinter(indent=4).pformat(TRAINING_ARGS))

    commander = CommanderProcess(run, TRAINING_ARGS, run_started_at)
    outcome_path = Path(TRAINING_ARGS.save_path) / 'run-outcome.json'
    try:
        for _ in commander.run():
            pass
    except Exception as error:
        write_run_outcome(
            outcome_path,
            RunOutcomeStatus.FAILED,
            str(error),
            run_started_at,
            run_configuration.budget.hourly_price_eur,
            commander.latest_completed_iteration,
        )
        raise

    outcome_status = RunOutcomeStatus.STOPPED if commander.final_stop_reason is not None else RunOutcomeStatus.COMPLETED
    write_run_outcome(
        outcome_path,
        outcome_status,
        commander.final_stop_reason,
        run_started_at,
        run_configuration.budget.hourly_price_eur,
        commander.latest_completed_iteration,
    )

    log('Training finished')
