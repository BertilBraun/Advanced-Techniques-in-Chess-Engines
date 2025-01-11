import asyncio
import torch

from src.util.log import log
from src.settings import TRAINING_ARGS, CurrentGame, CurrentGameVisuals
from src.games.GUI import BaseGridGameGUI
from src.eval.GameManager import GameManager
from src.eval.HumanPlayer import HumanPlayer
from src.eval.AlphaZeroBot import AlphaZeroBot
from src.util.save_paths import get_latest_model_iteration


class CommonHumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        _, rows, cols = CurrentGame.representation_shape
        gui = BaseGridGameGUI(rows, cols)
        super().__init__(gui, CurrentGameVisuals)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    iteration = get_latest_model_iteration(TRAINING_ARGS.num_iterations, TRAINING_ARGS.save_path)

    MAX_TIME_TO_THINK = 1.0

    game_manager = GameManager(
        AlphaZeroBot(iteration, max_time_to_think=MAX_TIME_TO_THINK),
        CommonHumanPlayer(),
    )

    result = asyncio.run(game_manager.play_game())
    log('Final Result - Winner:', result)
    log('Board:')
    log(game_manager.board.board)
