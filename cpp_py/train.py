from pathlib import Path
import subprocess

from src.cluster.TrainerProcess import TrainerProcess
from src.util.exceptions import log_exceptions
from src.util.profiler import start_cpu_usage_logger
from src.util.save_paths import get_latest_model_iteration


if __name__ == '__main__':
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

    log('Starting training')
    log('Training on:', 'GPU' if USE_GPU else 'CPU')
    log('Training args:')
    log(TRAINING_ARGS, use_pprint=True)

    run = get_run_id()
    log(f'Run ID: {run}')

    with TensorboardWriter(run, 'training_args', postfix_pid=False):
        import pprint

        log_text('TrainingArgs', pprint.PrettyPrinter(indent=4).pformat(TRAINING_ARGS))

    # TODO make sure to Write the first model to "save_path/model_1.pt" before starting the C++ self players

    self_play_process = subprocess.Popen(
        [
            'build/AlphaZeroSelfPlay',
            str(run),
            str(TRAINING_ARGS.save_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    start_gpu_usage_logger(run)
    start_cpu_usage_logger(run, 'self_play', self_play_process.pid)

    Path(TRAINING_ARGS.save_path).mkdir(parents=True, exist_ok=True)

    log('Setting up connections...')
    # Trainer and Commander
    trainer_device_id = torch.cuda.device_count() - 1
    trainer_process = TrainerProcess(TRAINING_ARGS, run, trainer_device_id)

    log('Connections set up.')

    starting_iteration = get_latest_model_iteration(TRAINING_ARGS.save_path)
    log(f'Starting training at iteration {starting_iteration}.')

    with log_exceptions('Commander process'), TensorboardWriter(run, 'trainer', postfix_pid=False):
        for iteration in range(starting_iteration, TRAINING_ARGS.num_iterations):
            training_stats = trainer_process.train(iteration)
            log(f'Trainer finished at iteration {iteration}.')
            log(f'Iteration {iteration}: {training_stats}')

            # start EvaluationProcess
            Process(target=run_evaluation_process, args=(run, TRAINING_ARGS, iteration), daemon=True).start()
