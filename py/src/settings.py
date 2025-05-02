import os
from pathlib import Path
import torch
from src.train.TrainingArgs import (
    ClusterParams,
    EvaluationParams,
    MCTSParams,
    NetworkParams,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)
from src.util import lerp
from src.util.tensorboard import *

USE_GPU = torch.cuda.is_available()
# Note CPU only seems to work for float32, on the GPU float16 and bfloat16 give no descerable difference in speed
TORCH_DTYPE = torch.bfloat16 if USE_GPU else torch.float32


def get_run_id():
    for run in range(10000):
        log_folder = f'logs/run_{run}'
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
            return run

    raise Exception('Could not find a free log folder')


LOG_FOLDER = 'logs'
SAVE_PATH = 'training_data'

PLAY_C_PARAM = 1.0


def sampling_window(current_iteration: int) -> int:
    """A slowly increasing sampling window, where the size of the window would start off small, and then slowly increase as the model generation count increased. This allowed us to quickly phase out very early data before settling to our fixed window size. We began with a window size of 4, so that by model 5, the first (and worst) generation of data was phased out. We then increased the history size by one every two models, until we reached our full 20 model history size at generation 35."""
    return min(3, 3 + (current_iteration - 5) // 5, 12)


def learning_rate(current_iteration: int) -> float:
    # SGD based on https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/training_go.py
    if current_iteration < 10:
        return 0.1
    return 0.01

    # AdamW
    if current_iteration < 10:
        return 0.001
    return 0.0005

    if current_iteration < 20:
        return 0.2
    if current_iteration < 50:
        return 0.02
    if current_iteration < 80:
        return 0.008
    if current_iteration < 100:
        return 0.002
    return 0.0002

    base_lr = 0.2
    lr_decay = 0.9
    return base_lr * (lr_decay ** (current_iteration / 4))


def learning_rate_scheduler(batch_percentage: float, base_lr: float) -> float:
    """1 Cycle learning rate policy.
    Ramp up from lr/10 to lr over 50% of the batches, then ramp down to lr/10 over the remaining 50% of the batches.
    Do this for each epoch separately.
    """
    return base_lr  # NOTE from some small scale testing, the one cycle policy seems to be detrimental to the training process
    min_lr = base_lr / 10

    if batch_percentage < 0.5:
        return lerp(min_lr, base_lr, batch_percentage * 2)
    else:
        return lerp(base_lr, min_lr, (batch_percentage - 0.5) * 2)


if False:
    from src.games.connect4.Connect4Game import Connect4Game, Connect4Move
    from src.games.connect4.Connect4Board import Connect4Board
    from src.games.connect4.Connect4Visuals import Connect4Visuals

    CurrentGameMove = Connect4Move
    CurrentGame = Connect4Game()
    CurrentBoard = Connect4Board
    CurrentGameVisuals = Connect4Visuals()

    NUM_GPUS = torch.cuda.device_count()
    SELF_PLAYERS_PER_NODE = 24
    # Assuming 12 parallel self players per node and 6 additional self players on the training GPU
    NUM_SELF_PLAYERS = (NUM_GPUS - 1) * SELF_PLAYERS_PER_NODE + SELF_PLAYERS_PER_NODE // 2

    network = NetworkParams(num_layers=9, hidden_size=64)

    PARALLEL_GAMES = 128

    TRAINING_ARGS = TrainingArgs(
        num_iterations=100,
        save_path=SAVE_PATH + '/connect4',
        num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS,
        network=network,
        self_play=SelfPlayParams(
            num_parallel_games=PARALLEL_GAMES,
            temperature=1.25,
            num_moves_after_which_to_play_greedy=10,
            mcts=MCTSParams(
                num_searches_per_turn=600,
                num_parallel_searches=8,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=dirichlet_alpha,
                c_param=4,
                min_visit_count=2,
            ),
        ),
        cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
        training=TrainingParams(
            num_epochs=1,
            batch_size=128,
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
    # TODO remove
    TRAINING_ARGS = TrainingArgs(
        num_iterations=9,
        save_path=SAVE_PATH + '/connect4',
        num_games_per_iteration=200,
        network=network,
        self_play=SelfPlayParams(
            num_parallel_games=32,
            temperature=1.25,
            result_score_weight=0,  # 0.15,
            num_moves_after_which_to_play_greedy=15,
            mcts=MCTSParams(
                num_searches_per_turn=600,
                num_parallel_searches=2,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=1.0,
                c_param=2,
                min_visit_count=1,
            ),
        ),
        cluster=ClusterParams(num_self_play_nodes_on_cluster=1),
        training=TrainingParams(
            num_epochs=1,
            batch_size=256,
            sampling_window=sampling_window,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
        ),
    )

elif True:

    def ensure_eval_dataset_exists(dataset_path: str) -> None:
        if not os.path.exists(dataset_path):
            from src.games.chess import ChessDatabase

            out_paths = ChessDatabase.process_month(2024, 10, num_games_per_month=50)
            assert len(out_paths) == 1
            out_paths[0].rename(dataset_path)

    from src.games.chess.ChessGame import ChessGame, ChessMove
    from src.games.chess.ChessBoard import ChessBoard
    from src.games.chess.ChessVisuals import ChessVisuals

    CurrentGameMove = ChessMove
    CurrentGame = ChessGame()
    CurrentBoard = ChessBoard
    CurrentGameVisuals = ChessVisuals()

    NUM_GPUS = torch.cuda.device_count()
    SELF_PLAYERS_PER_NODE = 42  # TODO 32?
    NUM_SELF_PLAYERS = (NUM_GPUS - 1) * SELF_PLAYERS_PER_NODE + (2 * SELF_PLAYERS_PER_NODE) // 3
    NUM_SELF_PLAYERS = max(1, NUM_SELF_PLAYERS)

    NUM_SELF_PLAYERS = min(NUM_SELF_PLAYERS, multiprocessing.cpu_count() - 3)

    network = NetworkParams(num_layers=15, hidden_size=64)
    training = TrainingParams(
        num_epochs=1,
        optimizer='sgd',
        batch_size=512,  # TODO 2048,
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

    PARALLEL_GAMES = 8
    NUM_SEARCHES_PER_TURN = 320  # TODO 640
    MIN_VISIT_COUNT = 0  # TODO 1 or 2?

    if not USE_GPU:  # TODO remove
        PARALLEL_GAMES = 2
        NUM_SEARCHES_PER_TURN = 640
        MIN_VISIT_COUNT = 1

    TRAINING_ARGS = TrainingArgs(
        num_iterations=12,  # 120
        save_path=SAVE_PATH + '/chess',
        num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS * 2,
        network=network,
        self_play=SelfPlayParams(
            num_parallel_games=PARALLEL_GAMES,
            num_moves_after_which_to_play_greedy=25,
            result_score_weight=0.0,  # TODO 0.15,
            resignation_threshold=-1.0,  # TODO -0.9,
            temperature=1.0,  # based on https://github.com/QueensGambit/CrazyAra/blob/19e37d034cce086947f3fdbeca45af885959bead/DeepCrazyhouse/configs/rl_config.py#L45
            num_games_after_which_to_write=1,
            mcts=MCTSParams(
                num_searches_per_turn=NUM_SEARCHES_PER_TURN,  # based on https://arxiv.org/pdf/1902.10565
                num_parallel_searches=8,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=0.3,  # Based on AZ Paper
                c_param=1.7,  # Based on MiniGO Paper
                min_visit_count=MIN_VISIT_COUNT,
            ),
        ),
        cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
        training=training,
        evaluation=evaluation,
    )


elif False:
    from src.games.checkers.CheckersGame import CheckersGame, CheckersMove
    from src.games.checkers.CheckersBoard import CheckersBoard
    from src.games.checkers.CheckersVisuals import CheckersVisuals

    CurrentGameMove = CheckersMove
    CurrentGame = CheckersGame()
    CurrentBoard = CheckersBoard
    CurrentGameVisuals = CheckersVisuals()

    NUM_GPUS = torch.cuda.device_count()
    SELF_PLAYERS_PER_NODE = 12
    # Assuming 12 parallel self players per node and 6 additional self players on the training GPU
    NUM_SELF_PLAYERS = (NUM_GPUS - 1) * SELF_PLAYERS_PER_NODE + SELF_PLAYERS_PER_NODE // 2
    NUM_SELF_PLAYERS = max(1, NUM_SELF_PLAYERS)

    network = NetworkParams(num_layers=10, hidden_size=128)
    training = TrainingParams(
        num_epochs=2,
        batch_size=256,
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
            num_moves_after_which_to_play_greedy=10,
            mcts=MCTSParams(
                num_searches_per_turn=320,  # 200, based on https://arxiv.org/pdf/1902.10565
                num_parallel_searches=4,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=0.2,  # Average of 50 moves possible per turn -> 10/50 = 0.2
                c_param=2,
            ),
        ),
        cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
        training=training,
        evaluation=EvaluationParams(
            num_searches_per_turn=60,
            num_games=20,
            every_n_iterations=10,
        ),
    )
    # TODO remove
    TEST_TRAINING_ARGS = TrainingArgs(
        num_iterations=25,
        save_path=SAVE_PATH + '/checkers',
        num_games_per_iteration=32,
        network=network,
        self_play=SelfPlayParams(
            num_parallel_games=2,
            num_games_after_which_to_write=4,
            num_moves_after_which_to_play_greedy=10,
            mcts=MCTSParams(
                num_searches_per_turn=64,
                num_parallel_searches=4,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=dirichlet_alpha,
                c_param=2,
            ),
        ),
        cluster=ClusterParams(num_self_play_nodes_on_cluster=1),
        training=training,
    )

elif True:

    def ensure_eval_dataset_exists(dataset_path: str) -> None:
        if not os.path.exists(dataset_path):
            from src.games.tictactoe.TicTacToeDatabase import generate_database

            generate_database(Path(dataset_path))

    from src.games.tictactoe.TicTacToeGame import TicTacToeGame, TicTacToeMove
    from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
    from src.games.tictactoe.TicTacToeVisuals import TicTacToeVisuals

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

    if TRAINING_ARGS.evaluation is not None:
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
