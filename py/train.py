import src.environ_setup  # noqa # isort:skip # This import is necessary for setting up the environment variables

import os

os.environ['OMP_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for OpenMP
os.environ['MKL_NUM_THREADS'] = '1'  # Limit the number of threads to 1 for MKL

import torch

torch.manual_seed(42)  # Set the random seed for PyTorch
torch.set_num_threads(1)  # Limit the number of threads to 1 for PyTorch
torch.set_num_interop_threads(1)  # Limit the number of inter-op threads to 1 for PyTorch

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

    with TensorboardWriter(run, 'training_args', postfix_pid=False):
        import pprint

        log_text('TrainingArgs', pprint.PrettyPrinter(indent=4).pformat(TRAINING_ARGS))

    commander = CommanderProcess(run, TRAINING_ARGS)
    for iteration, stats in commander.run():
        log(f'Trainer finished at iteration {iteration}.')

    log('Training finished')
