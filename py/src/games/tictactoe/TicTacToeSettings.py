import os
from pathlib import Path
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


from src.games.tictactoe.TicTacToeGame import TicTacToeGame, TicTacToeMove
from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
from src.games.tictactoe.TicTacToeVisuals import TicTacToeVisuals


def ensure_eval_dataset_exists(dataset_path: str) -> None:
    if not os.path.exists(dataset_path):
        from src.games.tictactoe.TicTacToeDatabase import generate_database

        generate_database(Path(dataset_path))


CurrentGameMove = TicTacToeMove
CurrentGame = TicTacToeGame()
CurrentBoard = TicTacToeBoard
CurrentGameVisuals = TicTacToeVisuals()


def sampling_window(current_iteration: int) -> int:
    return 3


# Test training args to verify the implementation
TRAINING_ARGS = TrainingArgs(
    num_iterations=12,
    save_path=SAVE_PATH + '/tictactoe',
    num_games_per_iteration=300,
    network=NetworkParams(num_layers=8, hidden_size=16),
    self_play=SelfPlayParams(
        temperature=1.25,
        num_parallel_games=1,  # TODO 5,
        num_moves_after_which_to_play_greedy=5,
        result_score_weight=0.15,
        mcts=MCTSParams(
            num_searches_per_turn=200,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=1.0,
            c_param=2,
            num_parallel_searches=1,  # TODO 2,
            min_visit_count=2,
        ),
    ),
    cluster=ClusterParams(num_self_play_nodes_on_cluster=1),
    training=TrainingParams(
        num_epochs=1,
        batch_size=128,
        optimizer='adamw',
        sampling_window=sampling_window,
        learning_rate=learning_rate,
        learning_rate_scheduler=learning_rate_scheduler,
    ),
    evaluation=EvaluationParams(
        num_searches_per_turn=60,
        num_games=30,
        every_n_iterations=1,
        dataset_path='reference/memory_0_tictactoe_database.hdf5',
    ),
)

if TRAINING_ARGS.evaluation and TRAINING_ARGS.evaluation.dataset_path:
    ensure_eval_dataset_exists(TRAINING_ARGS.evaluation.dataset_path)
# TEST_TRAINING_ARGS = TrainingArgs(
#     num_iterations=50,
#     num_self_play_games_per_iteration=2,
#     num_parallel_games=1,
#     num_epochs=3,
#     batch_size=16,
#     temperature=1.25,
#     mcts_num_searches_per_turn=200,
#     mcts_dirichlet_epsilon=0.25,
#     mcts_dirichlet_alpha=lambda _: 0.3,
#     mcts_c_param=2,
#     nn_hidden_size=NN_HIDDEN_SIZE,
#     nn_num_layers=NN_NUM_LAYERS,
#     sampling_window=sampling_window,
#     learning_rate=learning_rate,
#     learning_rate_scheduler=learning_rate_scheduler,
#     save_path=SAVE_PATH,
#     num_self_play_nodes_on_cluster=1,
#     num_train_nodes_on_cluster=0,
# )
