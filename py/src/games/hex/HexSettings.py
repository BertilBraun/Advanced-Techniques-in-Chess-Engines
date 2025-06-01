from src.settings_common import *

from src.train.TrainingArgs import (
    ClusterParams,
    EvaluationParams,
    MCTSParams,
    NetworkParams,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)

from src.games.hex.HexGame import HexGame, HexMove
from src.games.hex.HexBoard import SIZE, HexBoard
from src.games.hex.HexVisuals import HexVisuals

CurrentGameMove = HexMove
CurrentGame = HexGame()
CurrentBoard = HexBoard
CurrentGameVisuals = HexVisuals()

network = NetworkParams(num_layers=12, hidden_size=128)

training = TrainingParams(
    num_epochs=2,
    optimizer='adamw',  # 'sgd',
    batch_size=512,  # TODO 2048,
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
)
evaluation = EvaluationParams(
    num_searches_per_turn=1,
    num_games=100,
    every_n_iterations=1,
)

PARALLEL_GAMES = 32
NUM_SEARCHES_PER_TURN = 1200
MIN_VISIT_COUNT = 4  # TODO 1 or 2?

TRAINING_ARGS = TrainingArgs(
    num_iterations=350,
    save_path=SAVE_PATH + '/hex',
    num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS,
    network=network,
    self_play=SelfPlayParams(
        num_parallel_games=PARALLEL_GAMES,
        num_moves_after_which_to_play_greedy=SIZE * 6,  # even number - no bias towards white
        result_score_weight=0.4,
        resignation_threshold=-1.0,  # TODO -0.9,
        temperature=1.0,
        num_games_after_which_to_write=4,
        portion_of_samples_to_keep=0.2,
        mcts=MCTSParams(
            num_searches_per_turn=NUM_SEARCHES_PER_TURN,  # based on https://arxiv.org/pdf/1902.10565
            num_parallel_searches=4,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=10 / (SIZE * SIZE),
            c_param=2.5,  # Higher to encourage exploration without adding too much noise through Dirichlet noise
            min_visit_count=MIN_VISIT_COUNT,
        ),
    ),
    cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
    training=training,
    evaluation=evaluation,
)
