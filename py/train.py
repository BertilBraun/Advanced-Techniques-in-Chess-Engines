import src.environ_setup  # noqa # isort:skip # This import is necessary for setting up the environment variables


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.set_start_method('spawn')

    import torch  # noqa

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    from src.util.log import log
    from src.settings import TRAINING_ARGS, USE_GPU
    from src.util.profiler import start_usage_logger
    from src.settings import tensorboard_writer, log_text
    from src.cluster.CommanderProcess import CommanderProcess

    log('Starting training')
    log('Training on:', 'GPU' if USE_GPU else 'CPU')
    log('Training args:')
    log(TRAINING_ARGS, use_pprint=True)

    start_usage_logger()

    with tensorboard_writer():
        import pprint

        log_text('TrainingArgs', pprint.PrettyPrinter(indent=4).pformat(TRAINING_ARGS))

    commander = CommanderProcess(TRAINING_ARGS)
    for iteration, stats in commander.run():
        log(f'Trainer finished at iteration {iteration}.')
        log(f'Iteration {iteration}: {stats}')

    log('Training finished')
