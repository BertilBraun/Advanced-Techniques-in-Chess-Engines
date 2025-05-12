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

network = NetworkParams(num_layers=15, hidden_size=128)
training = TrainingParams(
    num_epochs=1,
    optimizer='adamw',  # 'sgd',
    batch_size=1024,  # TODO 2048,
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
)
evaluation = EvaluationParams(
    num_searches_per_turn=60,
    num_games=40,
    every_n_iterations=1,
    dataset_path='reference/memory_0_chess_database.hdf5',
)
ensure_eval_dataset_exists(evaluation.dataset_path)

PARALLEL_GAMES = 32
NUM_SEARCHES_PER_TURN = 640
MIN_VISIT_COUNT = 0  # TODO 1 or 2?

if not USE_GPU:  # TODO remove
    PARALLEL_GAMES = 2
    NUM_SEARCHES_PER_TURN = 640
    MIN_VISIT_COUNT = 1

TRAINING_ARGS = TrainingArgs(
    num_iterations=120,
    save_path=SAVE_PATH + '/chess',
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
            dirichlet_alpha=0.3,  # Based on AZ Paper
            c_param=1.7,  # Based on MiniGO Paper
            min_visit_count=MIN_VISIT_COUNT,
            full_search_probability=0.2,  # Based on Paper "Accelerating Self-Play Learning in GO"
        ),
    ),
    cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
    training=training,
    evaluation=evaluation,
)
