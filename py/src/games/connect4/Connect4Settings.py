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
        inference_cache_capacity=250_000,
        use_inference_cache=True,
        starting_temperature=1.25,
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
    cluster=ClusterParams(
        trainer_device_type='cuda' if USE_GPU else 'cpu',
        trainer_process_group_backend='nccl' if USE_GPU else 'gloo',
        trainer_rank_zero_device_id=max(0, NUM_GPUS - 1),
        trainer_ddp_device_ids=(max(0, NUM_GPUS - 1),),
        evaluation_device_cycle=(max(0, NUM_GPUS - 1),),
        self_play_device_ids=(max(0, NUM_GPUS - 1),) * NUM_SELF_PLAYERS,
        self_play_tensorboard_processes=1,
        trainer_cpu_threads=1,
        trainer_interop_threads=1,
        self_play_node_ids_to_pause_during_training=(),
        max_concurrent_evaluations=1,
    ),
    training=TrainingParams(
        num_epochs=1,
        global_batch_size=128,
        local_batch_size=128,
        optimizer='adamw',  # 'sgd',
        sampling_window=sampling_window,
        learning_rate=learning_rate,
        learning_rate_scheduler=learning_rate_scheduler,
    ),
    run_limits=DEFAULT_RUNTIME_LIMITS,
    artifact_retention=DEFAULT_ARTIFACT_RETENTION,
    evaluation=EvaluationParams(
        num_searches_per_turn=60,
        num_games=20,
        every_n_iterations=10,
        evaluate_initial_checkpoint=False,
        max_concurrent_tasks=1,
        inference_cache_capacity=250_000,
        use_inference_cache=True,
        dataset_path='reference/memory_0_connect4_database.hdf5',
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
)
