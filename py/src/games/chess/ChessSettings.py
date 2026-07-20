import os
from src.train.TrainingArgs import (
    ClusterParams,
    EvaluationParams,
    GatingParams,
    MCTSParams,
    NetworkParams,
    SEPlacement,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)

from src.settings_common import *

from src.games.chess.ChessGame import ChessGame, ChessMove
from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.ChessVisuals import ChessVisuals


def on_startup() -> None:
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

network = NetworkParams(
    num_layers=12,
    hidden_size=112,
    se_placement=SEPlacement.EVERY_SECOND_BLOCK,
)
training = TrainingParams(
    num_epochs=1,
    optimizer='adamw',  # 'sgd',
    batch_size=1024,
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
    num_workers=16,
)
evaluation = EvaluationParams(
    num_searches_per_turn=64,
    num_games=100,
    every_n_iterations=1,
    evaluate_initial_checkpoint=False,
    max_concurrent_tasks=1,
    dataset_path='reference/memory_0_chess_database.hdf5',
    reference_model_path=None,
    opening_suite_path=None,
    raw_results_path=None,
    maximum_game_plies=200,
    bootstrap_seed=0,
    bootstrap_samples=10_000,
    mcts_threads=1,
    previous_model_offsets=(5, 10),
    historical_model_iterations=tuple(range(10, 301, 10)),
    historical_model_rotation_period=1,
    stockfish_skill_levels=(0, 1, 2, 3),
    stockfish_binary_path=None,
    stockfish_nodes_per_move=1_000,
    stockfish_threads=1,
    stockfish_hash_mib=128,
    evaluate_random=True,
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

NUM_SELF_PLAYERS = 1
NUM_THREADS = 1
PARALLEL_GAMES = 2
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
        inference_cache_capacity=250_000,
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
    cluster=ClusterParams(
        trainer_device_id=max(0, torch.cuda.device_count() - 1),
        evaluation_device_cycle=(max(0, torch.cuda.device_count() - 1),),
        self_play_device_ids=(max(0, torch.cuda.device_count() - 1),),
        self_play_tensorboard_processes=1,
        trainer_cpu_threads=1,
        trainer_interop_threads=1,
        self_play_node_ids_to_pause_during_training=(),
        max_concurrent_evaluations=1,
    ),
    training=training,
    run_limits=DEFAULT_RUNTIME_LIMITS,
    artifact_retention=DEFAULT_ARTIFACT_RETENTION,
    evaluation=evaluation,
    gating=gating,
    random_seed=0,
    self_play_search_warmup_iterations=15,
    self_play_value_warmup_iterations=30,
    self_play_endgame_shortcut_fade_iterations=50,
    on_startup=on_startup,
)
