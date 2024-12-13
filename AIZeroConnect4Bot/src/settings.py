import torch
from AIZeroConnect4Bot.src.util import lerp
from AIZeroConnect4Bot.src.TrainingArgs import TrainingArgs


ROW_COUNT = 7
COLUMN_COUNT = 8
NUM_RES_BLOCKS = 10
NUM_HIDDEN = 128
ENCODING_CHANNELS = 1  # 6+6 for chess, 1 for connect4
CELL_STATES = 3  # 3 for connect4, 2 for chess
ACTION_SIZE = COLUMN_COUNT
AVERAGE_NUM_MOVES_PER_GAME = 20

TORCH_DTYPE = torch.bfloat16  # if torch.cuda.is_available() else torch.float32


def sampling_window(current_iteration: int) -> int:
    """A slowly increasing sampling window, where the size of the window would start off small, and then slowly increase as the model generation count increased. This allowed us to quickly phase out very early data before settling to our fixed window size. We began with a window size of 4, so that by model 5, the first (and worst) generation of data was phased out. We then increased the history size by one every two models, until we reached our full 20 model history size at generation 35."""
    if current_iteration < 5:
        return 4
    return min(4 + (current_iteration - 5) // 2, 20)


def learning_rate(current_iteration: int) -> float:
    if current_iteration < 10:
        return 0.2
    if current_iteration < 20:
        return 0.02
    return 0.002


def learning_rate_scheduler(batch_percentage: float, base_lr: float) -> float:
    """1 Cycle learning rate policy.
    Ramp up from lr/10 to lr over 50% of the batches, then ramp down to lr/10 over the remaining 50% of the batches.
    Do this for each epoch separately.
    """
    min_lr = base_lr / 10

    if batch_percentage < 0.5:
        return lerp(min_lr, base_lr, batch_percentage * 2)
    else:
        return lerp(base_lr, min_lr, (batch_percentage - 0.5) * 2)


# Chess training args
# ALPHA_ZERO_TRAINING_ARGS = TrainingArgs(
#     num_iterations=200,
#     num_self_play_iterations=500_000,
#     num_parallel_games=128,  # unknown
#     num_iterations_per_turn=1600,
#     num_epochs=2,  # unknown
#     batch_size=2048,
#     temperature=1.0,
#     dirichlet_epsilon=0.25,
#     dirichlet_alpha=0.3,
#     c_param=4.0,  # unknown
# )


# Test training args to verify the implementation
TRAINING_ARGS = TrainingArgs(
    num_iterations=50,
    num_self_play_iterations=7040,  # parallel games * 55
    num_parallel_games=128,
    num_iterations_per_turn=800,
    num_epochs=6,
    batch_size=64,
    temperature=1.0,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=1,
    c_param=4.0,
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
    save_path='AIZeroConnect4Bot/training_data',
    num_train_nodes_on_cluster=0,
    num_self_play_nodes_on_cluster=4,
)

# Test training args to verify the implementation
# TRAINING_ARGS = TrainingArgs(
#     num_iterations=50,
#     num_self_play_iterations=64,
#     num_parallel_games=4,
#     num_iterations_per_turn=600,
#     num_epochs=5,
#     batch_size=64,
#     temperature=1.0,
#     dirichlet_epsilon=0.25,
#     dirichlet_alpha=1,
#     c_param=4.0,
#     sampling_window=sampling_window,
#     learning_rate=learning_rate,
#     learning_rate_scheduler=learning_rate_scheduler,
#     save_path='AIZeroConnect4Bot/training_data',
# )
