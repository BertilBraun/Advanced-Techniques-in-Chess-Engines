import torch
from src.util import lerp
from src.train.TrainingArgs import TrainingArgs

TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

LOG_FOLDER = 'AIZeroConnect4Bot/logs'
SAVE_PATH = 'AIZeroConnect4Bot/training_data'
TESTING = False

PLAY_C_PARAM = 1.0
VALUE_OUTPUT_HEADS = 1


def sampling_window(current_iteration: int) -> int:
    """A slowly increasing sampling window, where the size of the window would start off small, and then slowly increase as the model generation count increased. This allowed us to quickly phase out very early data before settling to our fixed window size. We began with a window size of 4, so that by model 5, the first (and worst) generation of data was phased out. We then increased the history size by one every two models, until we reached our full 20 model history size at generation 35."""
    if current_iteration < 5:
        return 4
    return min(4 + (current_iteration - 5) // 2, 20)


def learning_rate(current_iteration: int) -> float:
    # if current_iteration < 10:
    #    return 0.01
    if current_iteration < 10:
        return 0.005
    if current_iteration < 30:
        return 0.002
    if current_iteration < 50:
        return 0.0005
    return 0.0002


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


def dirichlet_alpha(iteration: int) -> float:
    # if iteration < 50:  # TODO
    #     return 0.01
    return 0.3


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

    CURRENT_GAME_MOVE = Connect4Move
    CURRENT_GAME = Connect4Game()
    CURRENT_BOARD = Connect4Board()
    CURRENT_GAME_VISUALS = Connect4Visuals()

    # Test training args to verify the implementation
    if torch.cuda.is_available() and not TESTING:
        NUM_NODES = 8
        NUM_TRAINERS = 1
        NUM_SELF_PLAY_NODES = NUM_NODES - NUM_TRAINERS
        PARALLEL_GAMES = 128  # Approximately 5min for 128 games
        TRAINING_ARGS = TrainingArgs(
            num_iterations=50,
            num_self_play_games_per_iteration=PARALLEL_GAMES * 2 * NUM_SELF_PLAY_NODES,
            num_parallel_games=PARALLEL_GAMES,
            num_searches_per_turn=800,
            num_epochs=5,
            batch_size=16,
            temperature=1.0,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=1,
            c_param=4.0,
            sampling_window=sampling_window,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            save_path=SAVE_PATH,
            num_train_nodes_on_cluster=1,
            num_self_play_nodes_on_cluster=NUM_SELF_PLAY_NODES,
        )
    else:
        # Test training args to verify the implementation
        TRAINING_ARGS = TrainingArgs(
            num_iterations=50,
            num_self_play_games_per_iteration=128,
            num_parallel_games=64,
            num_searches_per_turn=200,
            num_epochs=2,
            batch_size=16,
            temperature=1.0,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=1,
            c_param=4.0,
            sampling_window=sampling_window,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            save_path=SAVE_PATH,
            num_self_play_nodes_on_cluster=1,
            num_train_nodes_on_cluster=0,
        )
        # Test training args to verify the implementation
        TRAINING_ARGS = TrainingArgs(
            num_iterations=8,
            num_self_play_games_per_iteration=128 * 4 * 2,
            num_parallel_games=128,
            num_searches_per_turn=600,
            num_epochs=4,
            batch_size=128,
            temperature=1.25,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=lambda _: 0.3,
            c_param=2,
            sampling_window=lambda _: 1,
            learning_rate=lambda _: 0.001,
            learning_rate_scheduler=lambda _, lr: lr,
            save_path=SAVE_PATH + '/connect4',
            num_self_play_nodes_on_cluster=1,
            num_train_nodes_on_cluster=0,
        )

elif False:
    from src.games.checkers.CheckersGame import CheckersGame, CheckersMove
    from src.games.checkers.CheckersBoard import CheckersBoard
    from src.games.checkers.CheckersVisuals import CheckersVisuals

    CURRENT_GAME_MOVE = CheckersMove
    CURRENT_GAME = CheckersGame()
    CURRENT_BOARD = CheckersBoard()
    CURRENT_GAME_VISUALS = CheckersVisuals()
elif True:
    from src.games.tictactoe.TicTacToeGame import TicTacToeGame, TicTacToeMove
    from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard
    from src.games.tictactoe.TicTacToeVisuals import TicTacToeVisuals

    CURRENT_GAME_MOVE = TicTacToeMove
    CURRENT_GAME = TicTacToeGame()
    CURRENT_BOARD = TicTacToeBoard
    CURRENT_GAME_VISUALS = TicTacToeVisuals()

    # Test training args to verify the implementation
    if torch.cuda.is_available() and not TESTING:
        NUM_NODES = 8
        NUM_TRAINERS = 1
        NUM_SELF_PLAY_NODES = NUM_NODES - NUM_TRAINERS
        PARALLEL_GAMES = 128  # Approximately 5min for 128 games
        TRAINING_ARGS = None
    else:
        # Test training args to verify the implementation
        TRAINING_ARGS = TrainingArgs(
            num_iterations=4,
            num_self_play_games_per_iteration=128 * 4 * 2,
            num_parallel_games=128,
            num_searches_per_turn=60,
            num_epochs=4,
            batch_size=64,
            temperature=1.25,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=lambda _: 0.3,
            c_param=2,
            sampling_window=lambda _: 1,
            learning_rate=lambda _: 0.001,
            learning_rate_scheduler=lambda _, lr: lr,
            save_path=SAVE_PATH + '/tictactoe',
            num_self_play_nodes_on_cluster=1,
            num_train_nodes_on_cluster=0,
        )
        TRAINING_ARGS = TrainingArgs(
            num_iterations=50,
            num_self_play_games_per_iteration=2,
            num_parallel_games=1,
            num_searches_per_turn=200,
            num_epochs=3,
            batch_size=16,
            temperature=1.25,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=lambda _: 0.3,
            c_param=2,
            sampling_window=sampling_window,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            save_path=SAVE_PATH,
            num_self_play_nodes_on_cluster=1,
            num_train_nodes_on_cluster=0,
        )

    args = TRAINING_ARGS
