from AIZeroConnect4Bot.src.TrainingArgs import TrainingArgs


ROW_COUNT = 7
COLUMN_COUNT = 8
NUM_RES_BLOCKS = 10
NUM_HIDDEN = 128
ENCODING_CHANNELS = 1  # 6+6 for chess, 1 for connect4
ACTION_SIZE = COLUMN_COUNT


# Chess training args
ALPHA_ZERO_TRAINING_ARGS = TrainingArgs(
    num_iterations=200,
    num_self_play_iterations=500_000,
    num_parallel_games=128,  # unknown
    num_iterations_per_turn=1600,
    num_epochs=2,  # unknown
    batch_size=2048,
    temperature=1.0,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=0.3,
    c_param=4.0,  # unknown
)


# Test training args to verify the implementation
TRAINING_ARGS = TrainingArgs(
    num_iterations=20,
    num_self_play_iterations=6_000,
    num_parallel_games=128,
    num_iterations_per_turn=600,
    num_epochs=3,
    num_separate_nodes_on_cluster=1,
    batch_size=64,
    temperature=1.0,
    dirichlet_epsilon=0.25,
    dirichlet_alpha=1,
    c_param=4.0,
    save_path='AIZeroConnect4Bot/training_data',
)
