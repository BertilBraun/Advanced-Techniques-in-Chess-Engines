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
    from src.cluster.CommanderProcess import CommanderProcess

    log('Starting training')
    log('Training on:', 'GPU' if USE_GPU else 'CPU')
    log('Training args:')
    log(TRAINING_ARGS, use_pprint=True)

    start_usage_logger()

    commander = CommanderProcess(
        num_self_play_nodes=TRAINING_ARGS.cluster.num_self_play_nodes_on_cluster,
        num_inference_nodes=TRAINING_ARGS.cluster.num_inference_nodes_on_cluster,
    )

    commander.run()
    log('Training finished')
