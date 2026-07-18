from src.settings_common import *

from src.train.TrainingArgs import (
    ClusterParams,
    EvaluationParams,
    MCTSParams,
    NetworkParams,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)

from src.games.hex.HexGame import HexGame, HexMove
from src.games.hex.HexBoard import SIZE, HexBoard
from src.games.hex.HexVisuals import HexVisuals

CurrentGameMove = HexMove
CurrentGame = HexGame()
CurrentBoard = HexBoard
CurrentGameVisuals = HexVisuals()

network = NetworkParams(num_layers=12, hidden_size=128)

training = TrainingParams(
    num_epochs=2,
    optimizer='adamw',  # 'sgd',
    batch_size=512,  # TODO 2048,
    sampling_window=sampling_window,
    learning_rate=learning_rate,
    learning_rate_scheduler=learning_rate_scheduler,
)
evaluation = EvaluationParams(
    num_searches_per_turn=1,
    num_games=100,
    every_n_iterations=1,
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
    historical_model_iterations=tuple(range(10, 351, 10)),
    stockfish_skill_levels=(),
    stockfish_binary_path=None,
    stockfish_nodes_per_move=1_000,
    stockfish_threads=1,
    stockfish_hash_mib=128,
    evaluate_random=True,
)

PARALLEL_GAMES = 32
NUM_SEARCHES_PER_TURN = 1200
MIN_VISIT_COUNT = 4  # TODO 1 or 2?

TRAINING_ARGS = TrainingArgs(
    num_iterations=350,
    save_path=SAVE_PATH + '/hex',
    num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS,
    network=network,
    self_play=SelfPlayParams(
        num_parallel_games=PARALLEL_GAMES,
        inference_cache_capacity=250_000,
        num_moves_after_which_to_play_greedy=SIZE * 6,  # even number - no bias towards white
        result_score_weight=0.4,
        resignation_threshold=-1.0,  # TODO -0.9,
        starting_temperature=1.0,
        num_games_after_which_to_write=4,
        portion_of_samples_to_keep=0.2,
        mcts=MCTSParams(
            num_searches_per_turn=NUM_SEARCHES_PER_TURN,  # based on https://arxiv.org/pdf/1902.10565
            num_parallel_searches=4,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=10 / (SIZE * SIZE),
            c_param=2.5,  # Higher to encourage exploration without adding too much noise through Dirichlet noise
            min_visit_count=MIN_VISIT_COUNT,
        ),
    ),
    cluster=ClusterParams(
        trainer_device_id=max(0, NUM_GPUS - 1),
        evaluation_device_id=max(0, NUM_GPUS - 1),
        self_play_device_ids=(max(0, NUM_GPUS - 1),) * NUM_SELF_PLAYERS,
        trainer_cpu_threads=1,
        trainer_interop_threads=1,
        pause_self_play_during_training=False,
        max_concurrent_evaluations=1,
    ),
    training=training,
    run_limits=DEFAULT_RUNTIME_LIMITS,
    artifact_retention=DEFAULT_ARTIFACT_RETENTION,
    evaluation=evaluation,
    random_seed=0,
    self_play_search_warmup_iterations=18,
    self_play_value_warmup_iterations=35,
)
