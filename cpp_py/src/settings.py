import os
import torch
from src.train.TrainingArgs import ClusterParams, EvaluationParams, NetworkParams, TrainingArgs, TrainingParams
from src.util import clamp, lerp

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.util.tensorboard import *  # noqa # Load tensorboard after setting the environment variable

USE_GPU = torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count()
# Note CPU only seems to work for float32, on the GPU float16 and bfloat16 give no descerable difference in speed
TORCH_DTYPE = torch.bfloat16 if USE_GPU else torch.float32


ACTION_SIZE = 1968  # The number of possible moves in chess (including castling and en passant)
BOARD_LENGTH = 8  # The length of the chess board (8x8)
BOARD_SIZE = BOARD_LENGTH * BOARD_LENGTH  # The total number of squares on the chess board (64)
ENCODING_CHANNELS = 12 + 1 + 1  # 12 for pieces + 1 for castling rights + 1 for color


def get_run_id():
    for run in range(10000):
        log_folder = f'logs/run_{run}'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            return run

    raise Exception('Could not find a free log folder')


LOG_FOLDER = 'logs'
SAVE_PATH = 'training_data'


def sampling_window(current_iteration: int) -> int:
    """A slowly increasing sampling window, where the size of the window would start off small, and then slowly increase as the model generation count increased. This allowed us to quickly phase out very early data before settling to our fixed window size. We began with a window size of 4, so that by model 5, the first (and worst) generation of data was phased out. We then increased the history size by one every two models, until we reached our full 20 model history size at generation 35."""
    return min(3, 3 + (current_iteration - 5) // 5, 12)


def learning_rate(current_iteration: int) -> float:
    if current_iteration < 8:
        return 0.2
    if current_iteration < 20:
        return 0.02
    if current_iteration < 30:
        return 0.008
    if current_iteration < 50:
        return 0.002
    return 0.0002
    if current_iteration < 20:
        return 0.2
    if current_iteration < 80:
        return 0.02
    if current_iteration < 120:
        return 0.008
    if current_iteration < 180:
        return 0.002
    return 0.0002

    base_lr = 0.2
    lr_decay = 0.9
    return base_lr * (lr_decay ** (current_iteration / 4))


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


def ensure_eval_dataset_exists(dataset_path: str) -> None:
    from src.games import ChessDatabase

    if not os.path.exists(dataset_path + '_stats.json'):
        out_paths = ChessDatabase.process_month(2024, 10, num_games_per_month=50)
        assert len(out_paths) == 1
        assert dataset_path == str(out_paths[0])


SELF_PLAYERS_PER_NODE = 24
NUM_SELF_PLAY_GPUS = max(NUM_GPUS, 1)
NUM_SELF_PLAYERS = NUM_SELF_PLAY_GPUS * SELF_PLAYERS_PER_NODE
NUM_SELF_PLAYERS = clamp(NUM_SELF_PLAYERS, 1, multiprocessing.cpu_count() - 5)

network = NetworkParams(num_layers=15, hidden_size=128)
training = TrainingParams(
    num_epochs=2,
    batch_size=2048,
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
    max_num_sample_repetitions=3,
)
evaluation = EvaluationParams(
    num_games=40,
    every_n_iterations=1,
    dataset_path='reference/chess_database/iteration_202410/0',
)
ensure_eval_dataset_exists(evaluation.dataset_path)

PARALLEL_GAMES = 16

if not USE_GPU:  # TODO remove
    PARALLEL_GAMES = 2

TRAINING_ARGS = TrainingArgs(
    num_iterations=70,
    save_path=SAVE_PATH + '/chess',
    num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS + 1,
    network=network,
    cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
    training=training,
    evaluation=evaluation,
)
