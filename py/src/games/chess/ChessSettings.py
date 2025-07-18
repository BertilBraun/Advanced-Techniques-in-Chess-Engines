import os
from src.train.TrainingArgs import (
    ClusterParams,
    EvaluationParams,
    GatingParams,
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

network = NetworkParams(num_layers=8, hidden_size=96, se_positions=(1, 3))
training = TrainingParams(
    num_epochs=1,
    optimizer='adamw',  # 'sgd',
    batch_size=1024,  # 2048, # 4096, # 512??
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
    num_workers=16,
)
evaluation = EvaluationParams(
    num_searches_per_turn=64,
    num_games=100,
    every_n_iterations=1,
    dataset_path='reference/memory_0_chess_database.hdf5',
)

if USE_GATING := False:
    gating = GatingParams(
        num_games=100,
        num_searches_per_turn=64,
        ignore_draws=True,
        gating_threshold=0.5,
    )
else:
    gating = None

NUM_SELF_PLAYERS_PER_GPU = 16
NUM_SELF_PLAYERS = NUM_SELF_PLAYERS_PER_GPU * (torch.cuda.device_count() - 1) + (NUM_SELF_PLAYERS_PER_GPU // 2)
NUM_THREADS = multiprocessing.cpu_count() // NUM_SELF_PLAYERS * 3
PARALLEL_GAMES = NUM_THREADS * 32
NUM_SEARCHES_PER_TURN = 600  # More searches? 500-800? # NOTE: if KL divergence between policy and mcts policy is < 0.2 then add more searches
MIN_VISIT_COUNT = 1
PARALLEL_SEARCHES = 4

USE_CPP = True

if not USE_GPU:
    PARALLEL_GAMES = 2
    NUM_SEARCHES_PER_TURN = 128
    MIN_VISIT_COUNT = 1
    PARALLEL_SEARCHES = 2
    evaluation = None  # No evaluation in CPU mode


TRAINING_ARGS = TrainingArgs(
    num_iterations=300,
    save_path=SAVE_PATH + '/chess',
    num_games_per_iteration=5000,
    network=network,
    self_play=SelfPlayParams(
        num_parallel_games=PARALLEL_GAMES,
        num_moves_after_which_to_play_greedy=50,  # even number - no bias towards white
        result_score_weight=0.1,
        resignation_threshold=-5.0,  # TODO -0.9 or so
        starting_temperature=1.3,  # Decays to 0.1 up to num_moves_after_which_to_play_greedy
        num_games_after_which_to_write=2,
        portion_of_samples_to_keep=0.75,  # To not keep all symmetries
        only_store_sampled_moves=True,
        game_outcome_discount_per_move=0.005,  # Discount per move for game outcome
        mcts=MCTSParams(
            num_searches_per_turn=NUM_SEARCHES_PER_TURN,
            num_parallel_searches=PARALLEL_SEARCHES,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3,  # Based on AZ Paper
            c_param=1.5,  # Range 1.25-1.5
            min_visit_count=MIN_VISIT_COUNT,
            percentage_of_node_visits_to_keep=0.6,
            num_threads=NUM_THREADS,
            fast_searches_proportion_of_full_searches=1 / 4,
            playout_cap_randomization=0.25,
        ),
    ),
    cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
    training=training,
    evaluation=evaluation,
    gating=gating,
    on_startup=on_startup,
)
