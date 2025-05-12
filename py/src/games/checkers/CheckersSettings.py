from src.train.TrainingArgs import (
    ClusterParams,
    EvaluationParams,
    MCTSParams,
    NetworkParams,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)
from src.settings_common import *


from src.games.checkers.CheckersGame import CheckersGame, CheckersMove
from src.games.checkers.CheckersBoard import CheckersBoard
from src.games.checkers.CheckersVisuals import CheckersVisuals

CurrentGameMove = CheckersMove
CurrentGame = CheckersGame()
CurrentBoard = CheckersBoard
CurrentGameVisuals = CheckersVisuals()

network = NetworkParams(num_layers=10, hidden_size=128)
training = TrainingParams(
    num_epochs=2,
    batch_size=256,
    optimizer='adamw',  # 'sgd',
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
)

PARALLEL_GAMES = 64

TRAINING_ARGS = TrainingArgs(
    num_iterations=100,
    save_path=SAVE_PATH + '/checkers',
    num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS,
    network=network,
    self_play=SelfPlayParams(
        num_parallel_games=PARALLEL_GAMES,
        num_moves_after_which_to_play_greedy=10,
        mcts=MCTSParams(
            num_searches_per_turn=320,  # 200, based on https://arxiv.org/pdf/1902.10565
            num_parallel_searches=4,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.2,  # Average of 50 moves possible per turn -> 10/50 = 0.2
            c_param=2,
            full_search_probability=0.2,  # Based on Paper "Accelerating Self-Play Learning in GO"
        ),
    ),
    cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
    training=training,
    evaluation=EvaluationParams(
        num_searches_per_turn=60,
        num_games=20,
        every_n_iterations=10,
    ),
)
