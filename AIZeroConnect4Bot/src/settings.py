import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src.alpha_zero.train.TrainingArgs import (
    ClusterParams,
    EvaluationParams,
    MCTSParams,
    NetworkParams,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)

USE_PROFILING = True
USE_GPU = torch.cuda.is_available()
TORCH_DTYPE = torch.bfloat16 if USE_GPU else torch.float32

LOG_FOLDER = 'AIZeroConnect4Bot/logs'
SAVE_PATH = 'AIZeroConnect4Bot/training_data'
TESTING = True

DEDUPLICATE_EACH_ITERATION = True  # Deduplicate the dataset after each iteration
LOG_HISTOGRAMS = False

PLAY_C_PARAM = 1.0

_TB_SUMMARY = SummaryWriter(LOG_FOLDER)


def log_scalar(name: str, value: float, iteration: int) -> None:
    _TB_SUMMARY.add_scalar(name, value, iteration)


def log_histogram(name: str, values: torch.Tensor | np.ndarray, iteration: int) -> None:
    if not LOG_HISTOGRAMS:
        return
    values = values.reshape(-1)
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    _TB_SUMMARY.add_histogram(name, values, iteration)


def sampling_window(current_iteration: int) -> int:
    """A slowly increasing sampling window, where the size of the window would start off small, and then slowly increase as the model generation count increased. This allowed us to quickly phase out very early data before settling to our fixed window size. We began with a window size of 4, so that by model 5, the first (and worst) generation of data was phased out. We then increased the history size by one every two models, until we reached our full 20 model history size at generation 35."""
    if current_iteration < 20:
        return 20
    return min(20 + (current_iteration - 20) // 8, 40)


def learning_rate(current_iteration: int) -> float:
    base_lr = 0.025
    lr_decay = 0.85
    return base_lr * (lr_decay ** (current_iteration / 4))
    # if current_iteration < 10:
    #    return 0.01
    if current_iteration < 10:
        return 0.005
    if current_iteration < 30:
        return 0.002
    if current_iteration < 50:
        return 0.0005
    return 0.0002


# Chess training args
# ALPHA_ZERO_TRAINING_ARGS = TrainingArgs(
#     num_iterations=200,
#     num_self_play_iterations=500_000,
#     num_parallel_games=128,  # unknown
#     num_iterations_per_turn=1600,
#     num_epochs=2,  # unknown
#     batch_size=2048,
#     temperature=1.0,
#     dirichlet_epsilon=0.25,
#     dirichlet_alpha=0.3,
#     c_param=4.0,  # unknown
# )

if True:
    from src.games.connect4.Connect4Game import Connect4Game, Connect4Move
    from src.games.connect4.Connect4Board import Connect4Board
    from src.games.connect4.Connect4Visuals import Connect4Visuals

    CurrentGameMove = Connect4Move
    CurrentGame = Connect4Game()
    CurrentBoard = Connect4Board
    CurrentGameVisuals = Connect4Visuals()

    NUM_NODES = 2
    NUM_TRAINERS = 1
    NUM_SELF_PLAYERS = NUM_NODES * 8 - NUM_TRAINERS  # Assuming 8 parallel self players per node

    NN_HIDDEN_SIZE = 128
    NN_NUM_LAYERS = 9

    PARALLEL_GAMES = 64

    TRAINING_ARGS = TrainingArgs(
        num_iterations=100,
        save_path=SAVE_PATH + '/connect4',
        mcts=MCTSParams(
            num_searches_per_turn=600,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=lambda _: 1.0,
            c_param=4,
        ),
        network=NetworkParams(
            num_layers=NN_NUM_LAYERS,
            hidden_size=NN_HIDDEN_SIZE,
        ),
        self_play=SelfPlayParams(
            temperature=1.25,
            num_parallel_games=PARALLEL_GAMES,
            num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS,
        ),
        cluster=ClusterParams(
            num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS,
            num_train_nodes_on_cluster=NUM_TRAINERS,
        ),
        training=TrainingParams(
            num_epochs=1,
            batch_size=128,
            sampling_window=sampling_window,
            learning_rate=learning_rate,
        ),
        evaluation=EvaluationParams(
            num_searches_per_turn=60,
            num_games=30,
            every_n_iterations=3,
        ),
    )
    # TODO remove
    TEST_TRAINING_ARGS = TrainingArgs(
        num_iterations=25,
        save_path=SAVE_PATH + '/connect4',
        mcts=MCTSParams(
            num_searches_per_turn=60,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=lambda _: 0.3,
            c_param=2,
        ),
        network=NetworkParams(
            num_layers=NN_NUM_LAYERS,
            hidden_size=NN_HIDDEN_SIZE,
        ),
        self_play=SelfPlayParams(
            temperature=1.25,
            num_parallel_games=4,
            num_games_per_iteration=4 * 2,
        ),
        cluster=ClusterParams(
            num_self_play_nodes_on_cluster=1,
            num_train_nodes_on_cluster=0,
        ),
        training=TrainingParams(
            num_epochs=4,
            batch_size=128,
            sampling_window=sampling_window,
            learning_rate=learning_rate,
        ),
    )


elif False:
    from src.games.checkers.CheckersGame import CheckersGame, CheckersMove
    from src.games.checkers.CheckersBoard import CheckersBoard
    from src.games.checkers.CheckersVisuals import CheckersVisuals

    CurrentGameMove = CheckersMove
    CurrentGame = CheckersGame()
    CurrentBoard = CheckersBoard()
    CurrentGameVisuals = CheckersVisuals()
    NUM_RES_BLOCKS = 10
    NUM_HIDDEN = 128
elif True:
    from src.games.tictactoe.TicTacToeGame import TicTacToeGame, TicTacToeMove
    from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
    from src.games.tictactoe.TicTacToeVisuals import TicTacToeVisuals

    CurrentGameMove = TicTacToeMove
    CurrentGame = TicTacToeGame()
    CurrentBoard = TicTacToeBoard
    CurrentGameVisuals = TicTacToeVisuals()

    NN_HIDDEN_SIZE = 64
    NN_NUM_LAYERS = 4

    # Test training args to verify the implementation
    if torch.cuda.is_available() and not TESTING:
        NUM_NODES = 8
        NUM_TRAINERS = 1
        NUM_SELF_PLAYERS = NUM_NODES - NUM_TRAINERS
        PARALLEL_GAMES = 128  # Approximately 5min for 128 games
        TRAINING_ARGS = None
    else:
        # Test training args to verify the implementation
        TRAINING_ARGS = TrainingArgs(
            num_iterations=12,
            save_path=SAVE_PATH + '/tictactoe',
            mcts=MCTSParams(
                num_searches_per_turn=60,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=lambda _: 0.3,
                c_param=2,
            ),
            network=NetworkParams(
                num_layers=NN_NUM_LAYERS,
                hidden_size=NN_HIDDEN_SIZE,
            ),
            self_play=SelfPlayParams(
                temperature=1.25,
                num_parallel_games=128,
                num_games_per_iteration=128 * 4 * 2,
            ),
            cluster=ClusterParams(
                num_self_play_nodes_on_cluster=1,
                num_train_nodes_on_cluster=0,
            ),
            training=TrainingParams(
                num_epochs=4,
                batch_size=64,
                sampling_window=lambda _: 3,
                learning_rate=lambda _: 0.001,
            ),
            evaluation=EvaluationParams(
                num_searches_per_turn=60,
                num_games=30,
                every_n_iterations=3,
            ),
        )
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
