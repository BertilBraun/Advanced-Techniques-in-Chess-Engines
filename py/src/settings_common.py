import os
import torch
from src.train.TrainingArgs import OptimizerType
from src.util import lerp
from src.util.tensorboard import *

USE_GPU = torch.cuda.is_available()
# Note CPU only seems to work for float32, on the GPU float16 and bfloat16 give no descerable difference in speed
TORCH_DTYPE = torch.bfloat16 if USE_GPU else torch.float32


def get_run_id():
    for run in range(10000):
        log_folder = f'logs/run_{run}'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            return run

    raise Exception('Could not find a free log folder')


LOG_FOLDER = 'logs'
SAVE_PATH = 'training_data'

NUM_GPUS = torch.cuda.device_count()
SELF_PLAYERS_PER_NODE = 45  # TODO 32?
NUM_SELF_PLAYERS = (NUM_GPUS - 1) * SELF_PLAYERS_PER_NODE + (2 * SELF_PLAYERS_PER_NODE) // 3
NUM_SELF_PLAYERS = max(1, NUM_SELF_PLAYERS)
NUM_SELF_PLAYERS = max(8, NUM_SELF_PLAYERS)  # TODO remove?

# Leave 2 CPUs for Trainer, Dataloader, Evaluation, etc.
NUM_SELF_PLAYERS = min(NUM_SELF_PLAYERS, multiprocessing.cpu_count() - 4)
NUM_SELF_PLAYERS = 10  # TODO remove?

PLAY_C_PARAM = 1.0


def sampling_window(current_iteration: int) -> int:
    """A slowly increasing sampling window, where the size of the window would start off small, and then slowly increase as the model generation count increased. This allowed us to quickly phase out very early data before settling to our fixed window size. We began with a window size of 4, so that by model 5, the first (and worst) generation of data was phased out. We then increased the history size by one every two models, until we reached our full 20 model history size at generation 35."""
    return min(max(5, 5 + (current_iteration - 5) // 5), 12)


def learning_rate(current_iteration: int, optimizer: OptimizerType) -> float:
    # SGD based on https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/training_go.py
    if optimizer == 'sgd':
        if current_iteration < 5:
            return 0.1
        elif current_iteration < 10:
            return 0.01
        elif current_iteration < 20:
            return 0.002
        elif current_iteration < 30:
            return 0.0002
        return 0.00002

    # AdamW
    if optimizer == 'adamw':
        if current_iteration < 5:
            return 0.002
        elif current_iteration < 100:
            return 0.001
        return 3 * 10**-4

    raise ValueError(f'Optimizer type {optimizer} not supported. Supported types: adamw, sgd')


def learning_rate_scheduler(batch_percentage: float, base_lr: float) -> float:
    """1 Cycle learning rate policy.
    Ramp up from lr/10 to lr over 50% of the batches, then ramp down to lr/10 over the remaining 50% of the batches.
    Do this for each epoch separately.
    """
    return base_lr  # NOTE from some small scale testing, the one cycle policy seems to be detrimental to the training process
    min_lr = base_lr / 10

    if batch_percentage < 0.5:
        return lerp(min_lr, base_lr, batch_percentage * 2)
    else:
        return lerp(base_lr, min_lr, (batch_percentage - 0.5) * 2)
