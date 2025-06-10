import os
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

from src.games.chess.ChessGame import ChessGame, ChessMove
from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.ChessVisuals import ChessVisuals


def on_startup():
    if evaluation is None:
        return
    ensure_eval_dataset_exists(evaluation.dataset_path)


def ensure_eval_dataset_exists(dataset_path: str | None) -> None:
    if dataset_path and not os.path.exists(dataset_path):
        from src.games.chess import ChessDatabase

        out_paths = ChessDatabase.process_month(2024, 10, num_games_per_month=50)
        assert len(out_paths) == 1
        out_paths[0].rename(dataset_path)


CurrentGameMove = ChessMove
CurrentGame = ChessGame()
CurrentBoard = ChessBoard
CurrentGameVisuals = ChessVisuals()

network = NetworkParams(num_layers=7, hidden_size=96)
training = TrainingParams(
    num_epochs=2,
    optimizer='adamw',  # 'sgd',
    batch_size=512,  # TODO 2048,
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
)
evaluation = EvaluationParams(
    num_searches_per_turn=32,
    num_games=100,
    every_n_iterations=1,
    dataset_path='reference/memory_0_chess_database.hdf5',
)

NUM_SELF_PLAYERS = 2
NUM_THREADS = int(64 // NUM_SELF_PLAYERS * 1.5)
PARALLEL_GAMES = NUM_THREADS
NUM_SEARCHES_PER_TURN = 800
MIN_VISIT_COUNT = 2
PARALLEL_SEARCHES = 4

USE_CPP = True

if not USE_GPU:  # TODO remove
    PARALLEL_GAMES = 2
    NUM_SEARCHES_PER_TURN = 40
    MIN_VISIT_COUNT = 1
    PARALLEL_SEARCHES = 2
    evaluation = None  # No evaluation in CPU mode

    NUM_SEARCHES_PER_TURN = 800  # TODO remove
    MIN_VISIT_COUNT = 2
    PARALLEL_SEARCHES = 4
    network = NetworkParams(num_layers=2, hidden_size=32)
    NUM_SELF_PLAYERS = 2

TRAINING_ARGS = TrainingArgs(
    num_iterations=300,
    save_path=SAVE_PATH + '/chess',
    num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS * 8,
    network=network,
    self_play=SelfPlayParams(
        num_parallel_games=PARALLEL_GAMES,
        num_moves_after_which_to_play_greedy=60,  # even number - no bias towards white
        result_score_weight=0.1,  # TODO increase?
        resignation_threshold=-0.95,
        temperature=1.0,
        num_games_after_which_to_write=4,
        portion_of_samples_to_keep=0.7,
        only_store_sampled_moves=True,
        mcts=MCTSParams(
            num_searches_per_turn=NUM_SEARCHES_PER_TURN,
            num_parallel_searches=PARALLEL_SEARCHES,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3,  # Based on AZ Paper
            c_param=1.7,  # Based on MiniGO Paper
            min_visit_count=MIN_VISIT_COUNT,
            node_reuse_discount=0.5,
            num_threads=NUM_THREADS,
        ),
    ),
    cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
    training=training,
    evaluation=evaluation,
    on_startup=on_startup,
)
