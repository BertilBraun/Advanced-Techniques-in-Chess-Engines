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
from src.games.hex.HexBoard import HexBoard
from src.games.hex.HexVisuals import HexVisuals

CurrentGameMove = HexMove
CurrentGame = HexGame()
CurrentBoard = HexBoard
CurrentGameVisuals = HexVisuals()

network = NetworkParams(num_layers=12, hidden_size=96)
training = TrainingParams(
    num_epochs=1,
    optimizer='adamw',  # 'sgd',
    batch_size=512,  # TODO 2048,
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
)
evaluation = EvaluationParams(
    num_searches_per_turn=60,
    num_games=30,
    every_n_iterations=1,
)

PARALLEL_GAMES = 32
NUM_SEARCHES_PER_TURN = 500
MIN_VISIT_COUNT = 1  # TODO 1 or 2?

TRAINING_ARGS = TrainingArgs(
    num_iterations=120,
    save_path=SAVE_PATH + '/hex',
    num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS * 2,
    network=network,
    self_play=SelfPlayParams(
        num_parallel_games=PARALLEL_GAMES,
        num_moves_after_which_to_play_greedy=24,  # even number - no bias towards white
        result_score_weight=0.15,
        resignation_threshold=-1.0,  # TODO -0.9,
        temperature=1.0,  # based on https://github.com/QueensGambit/CrazyAra/blob/19e37d034cce086947f3fdbeca45af885959bead/DeepCrazyhouse/configs/rl_config.py#L45
        num_games_after_which_to_write=1,
        mcts=MCTSParams(
            num_searches_per_turn=NUM_SEARCHES_PER_TURN,  # based on https://arxiv.org/pdf/1902.10565
            num_parallel_searches=4,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.1,
            c_param=1.7,  # Based on MiniGO Paper
            min_visit_count=MIN_VISIT_COUNT,
            full_search_probability=1.0,  # TODO? 0.2,  # Based on Paper "Accelerating Self-Play Learning in GO"
        ),
    ),
    cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
    training=training,
    evaluation=evaluation,
)
