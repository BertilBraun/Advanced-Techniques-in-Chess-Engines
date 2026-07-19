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
        inference_cache_capacity=250_000,
        num_moves_after_which_to_play_greedy=10,
        mcts=MCTSParams(
            num_searches_per_turn=320,  # 200, based on https://arxiv.org/pdf/1902.10565
            num_parallel_searches=4,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.2,  # Average of 50 moves possible per turn -> 10/50 = 0.2
            c_param=2,
        ),
    ),
    cluster=ClusterParams(
        trainer_device_id=max(0, NUM_GPUS - 1),
        evaluation_device_cycle=(max(0, NUM_GPUS - 1),),
        self_play_device_ids=(max(0, NUM_GPUS - 1),) * NUM_SELF_PLAYERS,
        self_play_tensorboard_processes=1,
        trainer_cpu_threads=1,
        trainer_interop_threads=1,
        self_play_node_ids_to_pause_during_training=(),
        max_concurrent_evaluations=1,
    ),
    training=training,
    run_limits=DEFAULT_RUNTIME_LIMITS,
    artifact_retention=DEFAULT_ARTIFACT_RETENTION,
    evaluation=EvaluationParams(
        num_searches_per_turn=60,
        num_games=20,
        every_n_iterations=10,
        evaluate_initial_checkpoint=False,
        max_concurrent_tasks=1,
        dataset_path=None,
        reference_model_path=None,
        opening_suite_path=None,
        raw_results_path=None,
        maximum_game_plies=200,
        bootstrap_seed=0,
        bootstrap_samples=10_000,
        mcts_threads=1,
        previous_model_offsets=(5, 10),
        historical_model_iterations=tuple(range(10, 101, 10)),
        historical_model_rotation_period=1,
        stockfish_skill_levels=(),
        stockfish_binary_path=None,
        stockfish_nodes_per_move=1_000,
        stockfish_threads=1,
        stockfish_hash_mib=128,
        evaluate_random=True,
    ),
    random_seed=0,
    self_play_search_warmup_iterations=5,
    self_play_value_warmup_iterations=10,
)
