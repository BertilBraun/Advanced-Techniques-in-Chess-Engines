from AIZeroConnect4Bot.src.TrainingArgs import TrainingArgs


ROW_COUNT = 7
COLUMN_COUNT = 8
NUM_RES_BLOCKS = 4
NUM_HIDDEN = 256
ENCODING_CHANNELS = 1  # 6+6 for chess, 1 for connect4
ACTION_SIZE = COLUMN_COUNT


ALPHA_ZERO_TRAINING_ARGS = TrainingArgs(
    num_iterations=200,
    num_self_play_iterations=500_000,
    num_parallel_games=128,  # unknown
    num_iterations_per_turn=1600,
    num_epochs=20,  # unknown
    batch_size=2048,
    temperature=1.0,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.03,
    c_param=2.0,  # unknown
)

TRAINING_ARGS = TrainingArgs(
    num_iterations=200,
    num_self_play_iterations=32000,
    num_parallel_games=64,
    num_iterations_per_turn=200,
    num_epochs=20,
    num_separate_nodes_on_cluster=2,
    batch_size=64,
    temperature=1.0,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.03,
    c_param=2.0,
)

TRAINING_ARGS = TrainingArgs(
    num_iterations=200,
    num_self_play_iterations=512,
    num_parallel_games=64,
    num_iterations_per_turn=200,
    num_epochs=20,
    num_separate_nodes_on_cluster=2,
    batch_size=64,
    temperature=1.0,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.03,
    c_param=2.0,
)

# Test training args to verify the implementation
TRAINING_ARGS = TrainingArgs(
    num_iterations=20,
    num_self_play_iterations=1,
    num_parallel_games=1,
    num_iterations_per_turn=200,
    num_epochs=6,
    num_separate_nodes_on_cluster=1,
    batch_size=64,
    temperature=1.0,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.3,
    c_param=2.0,
    save_path='AIZeroConnect4Bot/training_data',
)
