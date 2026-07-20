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
        starting_temperature=1.25,
        num_parallel_games=1,  # TODO 5,
        inference_cache_capacity=250_000,
        use_inference_cache=True,
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
    cluster=ClusterParams(
        trainer_device_id=max(0, NUM_GPUS - 1),
        evaluation_device_cycle=(max(0, NUM_GPUS - 1),),
        self_play_device_ids=(max(0, NUM_GPUS - 1),),
        self_play_tensorboard_processes=1,
        trainer_cpu_threads=1,
        trainer_interop_threads=1,
        self_play_node_ids_to_pause_during_training=(),
        max_concurrent_evaluations=1,
    ),
    training=TrainingParams(
        num_epochs=1,
        batch_size=128,
        optimizer='adamw',
        sampling_window=sampling_window,
        learning_rate=learning_rate,
        learning_rate_scheduler=learning_rate_scheduler,
    ),
    run_limits=DEFAULT_RUNTIME_LIMITS,
    artifact_retention=DEFAULT_ARTIFACT_RETENTION,
    evaluation=EvaluationParams(
        num_searches_per_turn=60,
        num_games=30,
        every_n_iterations=1,
        evaluate_initial_checkpoint=False,
        max_concurrent_tasks=1,
        inference_cache_capacity=250_000,
        dataset_path='reference/memory_0_tictactoe_database.hdf5',
        reference_model_path=None,
        opening_suite_path=None,
        raw_results_path=None,
        maximum_game_plies=200,
        bootstrap_seed=0,
        bootstrap_samples=10_000,
        mcts_threads=1,
        previous_model_offsets=(5, 10),
        historical_model_iterations=tuple(range(10, 13, 10)),
        historical_model_rotation_period=1,
        stockfish_skill_levels=(),
        stockfish_binary_path=None,
        stockfish_nodes_per_move=1_000,
        stockfish_threads=1,
        stockfish_hash_mib=128,
        evaluate_random=True,
    ),
    random_seed=0,
    self_play_search_warmup_iterations=1,
    self_play_value_warmup_iterations=2,
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
