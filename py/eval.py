import asyncio
import torch

from src.eval.TournamentManager import TournamentManager
from src.util.log import log
from src.settings import TRAINING_ARGS, CurrentGame, CurrentBoard, CurrentGameVisuals
from src.eval.GUI import BaseGridGameGUI
from src.eval.GameManager import GameManager
from src.eval.HumanPlayer import HumanPlayer
from src.eval.AlphaZeroBot import AlphaZeroBot
from src.util.save_paths import get_latest_model_iteration


class CommonHumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        _, rows, cols = CurrentGame.representation_shape
        if hasattr(CurrentBoard(), 'board_dimensions'):
            rows, cols = CurrentBoard().board_dimensions
        gui = BaseGridGameGUI(rows, cols)
        super().__init__(gui, CurrentGameVisuals)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    iteration = get_latest_model_iteration(TRAINING_ARGS.num_iterations, TRAINING_ARGS.save_path)

    HUMAN_PLAY = True

    if HUMAN_PLAY:
        MAX_TIME_TO_THINK = 1.0
        NETWORK_ONLY = True

        game_manager = GameManager(
            CommonHumanPlayer(),
            AlphaZeroBot(iteration, max_time_to_think=MAX_TIME_TO_THINK, network_eval_only=NETWORK_ONLY),
        )

        result = asyncio.run(game_manager.play_game())
        log('Game over. Result:', result)
        log(game_manager.board)
    else:
        tournament_manager = TournamentManager(
            AlphaZeroBot(iteration, max_time_to_think=0.1),
            AlphaZeroBot(iteration, network_eval_only=True),
            num_games=10,
        )

        log('Playing game...')
        results = asyncio.run(tournament_manager.play_games())
        log('Results:', results)
