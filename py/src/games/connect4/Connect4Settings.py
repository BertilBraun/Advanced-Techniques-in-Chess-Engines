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


from src.games.connect4.Connect4Game import Connect4Game, Connect4Move
from src.games.connect4.Connect4Board import Connect4Board
from src.games.connect4.Connect4Visuals import Connect4Visuals

CurrentGameMove = Connect4Move
CurrentGame = Connect4Game()
CurrentBoard = Connect4Board
CurrentGameVisuals = Connect4Visuals()

PARALLEL_GAMES = 128

TRAINING_ARGS = TrainingArgs(
    num_iterations=100,
    save_path=SAVE_PATH + '/connect4',
    num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS,
    network=NetworkParams(num_layers=9, hidden_size=64),
    self_play=SelfPlayParams(
        num_parallel_games=PARALLEL_GAMES,
        temperature=1.25,
        num_moves_after_which_to_play_greedy=10,
        mcts=MCTSParams(
            num_searches_per_turn=600,
            num_parallel_searches=8,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3,
            c_param=4,
            min_visit_count=2,
        ),
    ),
    cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
    training=TrainingParams(
        num_epochs=1,
        batch_size=128,
        optimizer='adamw',  # 'sgd',
        sampling_window=sampling_window,
        learning_rate=learning_rate,
        learning_rate_scheduler=learning_rate_scheduler,
    ),
    evaluation=EvaluationParams(
        num_searches_per_turn=60,
        num_games=20,
        every_n_iterations=10,
        dataset_path='reference/memory_0_connect4_database.hdf5',
    ),
)
