import src.environ_setup  # noqa # isort:skip # This import is necessary for setting up the environment variables

import os

os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for MKL
os.environ['TORCH_NUM_THREADS'] = '1'

# This ensures, that the seperate processes spawned by torch.multiprocessing do not interfere with each other by using more than one core. Since we are using as many processes as cores for workers, we need to limit the number of threads to 1 for each process. Otherwise, we would use more than one core per process, which would lead to a lot of context switching and slow down the training.

import torch  # noqa

from AlphaZeroCpp import init

init()

torch.manual_seed(42)  # Set the random seed for PyTorch
torch.set_num_threads(1)  # Limit the number of threads to 1 for PyTorch
# torch.set_num_interop_threads(1)  # Limit the number of inter-op threads to 1 for PyTorch

torch.autograd.set_detect_anomaly(True)

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

    start_gpu_usage_logger(run)

    # if a function on_startup is defined, call it
    if TRAINING_ARGS.on_startup is not None:
        log('Calling on_startup function...')
        TRAINING_ARGS.on_startup()

    with TensorboardWriter(run, 'training_args', postfix_pid=False):
        import pprint

        log_text('TrainingArgs', pprint.PrettyPrinter(indent=4).pformat(TRAINING_ARGS))

    commander = CommanderProcess(run, TRAINING_ARGS)
    commander.run()

    log('Training finished')
