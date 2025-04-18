if __name__ == '__main__':
    import torch.multiprocessing as mp

    # set the start method to spawn for multiprocessing
    # this is required for the C++ self play process
    # and should be set before importing torch.multiprocessing
    # otherwise it will not work on Windows
    mp.set_start_method('spawn')

import torch  # noqa
import AlphaZeroCpp


def self_play_main_loop(run: int):
    from src.settings import TRAINING_ARGS, NUM_SELF_PLAY_GPUS

    # start the self play main loop
    AlphaZeroCpp.self_play_main(
        run,
        TRAINING_ARGS.save_path,
        TRAINING_ARGS.cluster.num_self_play_nodes_on_cluster,
        NUM_SELF_PLAY_GPUS,
    )


def main():
    import torch.multiprocessing as mp

    # mp.set_start_method('spawn')

    import torch  # noqa

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    from pathlib import Path

    from src.settings import TRAINING_ARGS, USE_GPU, get_run_id, TensorboardWriter, log_text
    from src.cluster.EvaluationProcess import run_evaluation_process
    from src.cluster.TrainerProcess import TrainerProcess
    from src.util.exceptions import log_exceptions
    from src.util.profiler import start_cpu_usage_logger, start_gpu_usage_logger
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

    # start a process which imports AlphaZeroCpp and starts the self play main loop
    self_play_process = mp.Process(
        target=self_play_main_loop,
        args=(run,),
        daemon=True,
    )
    self_play_process.start()
    assert self_play_process.pid is not None, 'Failed to start self play process.'
    log(f'Self play process started with PID: {self_play_process.pid}')

    start_gpu_usage_logger(run)
    start_cpu_usage_logger(run, 'self_play', self_play_process.pid)

    with log_exceptions('Commander process'), TensorboardWriter(run, 'trainer', postfix_pid=False):
        for iteration in range(starting_iteration, TRAINING_ARGS.num_iterations):
            log(f'Starting training at iteration {iteration}.')
            training_stats = trainer_process.train(iteration)
            log(f'Trainer finished at iteration {iteration}.')
            log(f'Iteration {iteration}: {training_stats}')

            # start EvaluationProcess
            eval_process = mp.Process(target=run_evaluation_process, args=(run, TRAINING_ARGS, iteration))
            eval_process.start()
            eval_process.join()
            log(f'Evaluation process finished at iteration {iteration}.')


if __name__ == '__main__':
    main()
