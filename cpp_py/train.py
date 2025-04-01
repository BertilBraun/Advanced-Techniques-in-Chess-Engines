if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')

    import torch  # noqa

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    import subprocess
    from pathlib import Path

    from src.cluster.EvaluationProcess import run_evaluation_process
    from src.cluster.TrainerProcess import TrainerProcess
    from src.util.exceptions import log_exceptions
    from src.util.profiler import start_cpu_usage_logger, start_gpu_usage_logger
    from src.settings import TRAINING_ARGS, USE_GPU, NUM_GPUS, get_run_id, TensorboardWriter, log_text
    from src.util.save_paths import get_latest_model_iteration
    from src.util.log import log

    log('Starting training')
    log('Training on:', 'GPU' if USE_GPU else 'CPU')
    log('Training args:')
    log(TRAINING_ARGS, use_pprint=True)

    run = get_run_id()
    log(f'Run ID: {run}')

    with TensorboardWriter(run, 'training_args', postfix_pid=False):
        import pprint

        log_text('TrainingArgs', pprint.PrettyPrinter(indent=4).pformat(TRAINING_ARGS))

    Path(TRAINING_ARGS.save_path).mkdir(parents=True, exist_ok=True)

    # Trainer
    trainer_device_id = torch.cuda.device_count() - 1
    trainer_process = TrainerProcess(TRAINING_ARGS, run, trainer_device_id)

    starting_iteration = get_latest_model_iteration(TRAINING_ARGS.save_path)
    log(f'Starting training at iteration {starting_iteration}.')

    # make sure to Write the first model to "save_path/model_1.jit.pt" before starting the C++ self players
    trainer_process.ensure_model_exists(starting_iteration)

    log('Starting self play process.')
    self_play_process = subprocess.Popen(
        [
            'build/AlphaZeroSelfPlay',
            str(run),
            str(TRAINING_ARGS.save_path),
            str(TRAINING_ARGS.cluster.num_self_play_nodes_on_cluster),
            str(NUM_GPUS - 1),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    log(f'Self play process started with PID: {self_play_process.pid}')

    start_gpu_usage_logger(run)
    start_cpu_usage_logger(run, 'self_play', self_play_process.pid)

    with log_exceptions('Commander process'), TensorboardWriter(run, 'trainer', postfix_pid=False):
        for iteration in range(starting_iteration, TRAINING_ARGS.num_iterations):
            training_stats = trainer_process.train(iteration)
            log(f'Trainer finished at iteration {iteration}.')
            log(f'Iteration {iteration}: {training_stats}')

            # start EvaluationProcess
            mp.Process(target=run_evaluation_process, args=(run, TRAINING_ARGS, iteration), daemon=True).start()
