import os
import torch
from src.util.tensorboard import *

USE_GPU = torch.cuda.is_available()


def get_run_id() -> int:
    for run in range(10000):
        log_folder = f'logs/run_{run}'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            return run

    raise Exception('Could not find a free log folder')


LOG_FOLDER = 'logs'
