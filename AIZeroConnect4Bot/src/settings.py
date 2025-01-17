import multiprocessing
import numpy as np
import torch
from tensorboardX import SummaryWriter
from src.alpha_zero.train.TrainingArgs import (
    ClusterParams,
    EvaluationParams,
    InferenceParams,
    MCTSParams,
    NetworkParams,
    SelfPlayParams,
    TrainingArgs,
    TrainingParams,
)
from src.util import lerp

USE_PROFILING = True
USE_GPU = torch.cuda.is_available()
# Note CPU only seems to work for float32, on the GPU float16 and bfloat16 give no descerable difference in speed
TORCH_DTYPE = torch.bfloat16 if USE_GPU else torch.float32

LOG_FOLDER = 'AIZeroConnect4Bot/logs'
SAVE_PATH = 'AIZeroConnect4Bot/training_data'
TESTING = True

LOG_HISTOGRAMS = True  # Log any histograms to tensorboard - not sure, might be really slow, not sure though

PLAY_C_PARAM = 1.0

_TB_SUMMARY = SummaryWriter(LOG_FOLDER + f'/{multiprocessing.current_process().pid}')


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
    if current_iteration < 5:
        return 5
    return min(5 + (current_iteration - 5) // 6, 15)


def learning_rate(current_iteration: int) -> float:
    base_lr = 0.025
    lr_decay = 0.95
    return base_lr * (lr_decay ** (current_iteration / 4))


def learning_rate_scheduler(batch_percentage: float, base_lr: float) -> float:
    """1 Cycle learning rate policy.
    Ramp up from lr/10 to lr over 50% of the batches, then ramp down to lr/10 over the remaining 50% of the batches.
    Do this for each epoch separately.
    """
    min_lr = base_lr / 10

    if batch_percentage < 0.5:
        return lerp(min_lr, base_lr, batch_percentage * 2)
    else:
        return lerp(base_lr, min_lr, (batch_percentage - 0.5) * 2)


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

    network = NetworkParams(num_layers=9, hidden_size=128)

    PARALLEL_GAMES = 128

    def dirichlet_alpha(iteration: int) -> float:
        return 1.0

    TRAINING_ARGS = TrainingArgs(
        num_iterations=100,
        save_path=SAVE_PATH + '/connect4',
        num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS,
        network=network,
        inference=InferenceParams(batch_size=128),
        self_play=SelfPlayParams(
            num_parallel_games=PARALLEL_GAMES,
            temperature=1.25,
            mcts=MCTSParams(
                num_searches_per_turn=600,
                num_parallel_searches=8,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=dirichlet_alpha,
                c_param=4,
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
        ),
    )
    # TODO remove
    TRAINING_ARGS = TrainingArgs(
        num_iterations=25,
        save_path=SAVE_PATH + '/connect4',
        num_games_per_iteration=32,
        network=network,
        inference=InferenceParams(batch_size=128),
        self_play=SelfPlayParams(
            num_parallel_games=128,
            temperature=1.25,
            mcts=MCTSParams(
                num_searches_per_turn=100,
                num_parallel_searches=8,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=dirichlet_alpha,
                c_param=2,
            ),
        ),
        cluster=ClusterParams(num_self_play_nodes_on_cluster=1),
        training=TrainingParams(
            num_epochs=2,
            batch_size=32,
            sampling_window=sampling_window,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
        ),
    )

elif True:
    from src.games.chess.ChessGame import ChessGame, ChessMove
    from src.games.chess.ChessBoard import ChessBoard
    from src.games.chess.ChessVisuals import ChessVisuals

    CurrentGameMove = ChessMove
    CurrentGame = ChessGame()
    CurrentBoard = ChessBoard
    CurrentGameVisuals = ChessVisuals()

    NUM_GPUS = torch.cuda.device_count()
    SELF_PLAYERS_PER_NODE = 15
    NUM_SELF_PLAYERS = (NUM_GPUS - 1) * SELF_PLAYERS_PER_NODE + SELF_PLAYERS_PER_NODE // 2
    NUM_SELF_PLAYERS = max(1, NUM_SELF_PLAYERS)

    network = NetworkParams(num_layers=12, hidden_size=128)
    training = TrainingParams(
        num_epochs=2,
        batch_size=256,
        sampling_window=sampling_window,
        learning_rate=learning_rate,
        learning_rate_scheduler=learning_rate_scheduler,
    )
    inference = InferenceParams(batch_size=128)
    evaluation = EvaluationParams(
        num_searches_per_turn=60,
        num_games=40,
        every_n_iterations=5,
    )

    PARALLEL_GAMES = 64

    def dirichlet_alpha(iteration: int) -> float:
        return 0.3

    TRAINING_ARGS = TrainingArgs(
        num_iterations=100,
        save_path=SAVE_PATH + '/chess',
        num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS // 2,
        network=network,
        inference=inference,
        self_play=SelfPlayParams(
            num_parallel_games=PARALLEL_GAMES,
            num_moves_after_which_to_play_greedy=25,
            mcts=MCTSParams(
                num_searches_per_turn=240,  # based on https://arxiv.org/pdf/1902.10565
                num_parallel_searches=4,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=dirichlet_alpha,
                c_param=2,
            ),
        ),
        cluster=ClusterParams(num_self_play_nodes_on_cluster=NUM_SELF_PLAYERS),
        training=training,
        evaluation=evaluation,
    )
    # TODO remove
    TEST_TRAINING_ARGS = TrainingArgs(
        num_iterations=25,
        save_path=SAVE_PATH + '/chess',
        num_games_per_iteration=32,
        network=network,
        inference=inference,
        self_play=SelfPlayParams(
            num_parallel_games=64,
            num_moves_after_which_to_play_greedy=25,
            mcts=MCTSParams(
                num_searches_per_turn=100,
                num_parallel_searches=4,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=dirichlet_alpha,
                c_param=2,
            ),
        ),
        cluster=ClusterParams(num_self_play_nodes_on_cluster=1),
        training=training,
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
    inference = InferenceParams(batch_size=128)

    PARALLEL_GAMES = 64

    def dirichlet_alpha(iteration: int) -> float:
        return 0.2  # Average of 50 moves possible per turn -> 10/50 = 0.2

    TRAINING_ARGS = TrainingArgs(
        num_iterations=100,
        save_path=SAVE_PATH + '/checkers',
        num_games_per_iteration=PARALLEL_GAMES * NUM_SELF_PLAYERS,
        network=network,
        inference=inference,
        self_play=SelfPlayParams(
            num_parallel_games=PARALLEL_GAMES,
            num_moves_after_which_to_play_greedy=10,
            mcts=MCTSParams(
                num_searches_per_turn=320,  # 200, based on https://arxiv.org/pdf/1902.10565
                num_parallel_searches=4,
                dirichlet_epsilon=0.25,
                dirichlet_alpha=dirichlet_alpha,
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
        inference=inference,
        self_play=SelfPlayParams(
            num_parallel_games=2,
            num_samples_after_which_to_write=4,
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
    from src.games.tictactoe.TicTacToeGame import TicTacToeGame, TicTacToeMove
    from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
    from src.games.tictactoe.TicTacToeVisuals import TicTacToeVisuals

    CurrentGameMove = TicTacToeMove
    CurrentGame = TicTacToeGame()
    CurrentBoard = TicTacToeBoard
    CurrentGameVisuals = TicTacToeVisuals()

    NN_HIDDEN_SIZE = 64
    NN_NUM_LAYERS = 4

    def dirichlet_alpha(iteration: int) -> float:
        return 0.3

    def sampling_window(current_iteration: int) -> int:
        return 3

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
            num_games_per_iteration=32,
            inference=InferenceParams(
                batch_size=128,
            ),
            network=NetworkParams(
                num_layers=NN_NUM_LAYERS,
                hidden_size=NN_HIDDEN_SIZE,
            ),
            self_play=SelfPlayParams(
                temperature=1.25,
                num_parallel_games=128,
                num_moves_after_which_to_play_greedy=5,
                mcts=MCTSParams(
                    num_searches_per_turn=60,
                    dirichlet_epsilon=0.25,
                    dirichlet_alpha=dirichlet_alpha,
                    c_param=2,
                    num_parallel_searches=2,
                ),
            ),
            cluster=ClusterParams(
                num_self_play_nodes_on_cluster=1,
            ),
            training=TrainingParams(
                num_epochs=4,
                batch_size=64,
                sampling_window=sampling_window,
                learning_rate=learning_rate,
                learning_rate_scheduler=learning_rate_scheduler,
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
