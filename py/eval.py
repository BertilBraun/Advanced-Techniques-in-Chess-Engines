import torch

from src.eval.TournamentManager import TournamentManager
from src.games.chess.comparison_bots.StockfishBot import ChessStockfishBot
from src.util.log import log
from src.settings import TRAINING_ARGS, CurrentGame, CurrentBoard, CurrentGameVisuals, CURRENT_GAME
from src.eval.GameManager import GameManager
from src.eval.HumanPlayer import HumanPlayer
from src.eval.AlphaZeroBotCpp import AlphaZeroBot
from src.util.save_paths import get_latest_model_iteration, model_save_path

if CURRENT_GAME == 'hex':
    from src.eval.HexGUI import HexGridGameGUI as GUI
else:
    from src.eval.GridGUI import BaseGridGameGUI as GUI


class CommonHumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        _, rows, cols = CurrentGame.representation_shape
        if hasattr(CurrentBoard(), 'board_dimensions'):
            rows, cols = CurrentBoard().board_dimensions  # type: ignore
        gui = GUI(rows, cols)
        super().__init__(gui, CurrentGameVisuals)  # type: ignore


def alpha_zero_bot_factory(game_index: int) -> AlphaZeroBot:
    return AlphaZeroBot(
        current_model_path=model_path,
        device_id=game_index % max(1, torch.cuda.device_count()),
        max_time_to_think=0.5,
        network_eval_only=False,
    )


def stockfish_bot_factory(_: int) -> ChessStockfishBot:
    return ChessStockfishBot(skill_level=0, max_time_to_think=0.1)


iteration = get_latest_model_iteration(TRAINING_ARGS.save_path)
model_path = str(model_save_path(iteration, TRAINING_ARGS.save_path))

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    HUMAN_PLAY = True

    if HUMAN_PLAY:
        MAX_TIME_TO_THINK = 1.0
        NETWORK_ONLY = False

        game_manager = GameManager(
            CommonHumanPlayer(),
            AlphaZeroBot(model_path, device_id=0, max_time_to_think=MAX_TIME_TO_THINK, network_eval_only=NETWORK_ONLY),
        )

        result = game_manager.play_game()
        log('Game over. Result:', result)
        log(game_manager.board)
    else:
        tournament_manager = TournamentManager(
            alpha_zero_bot_factory,
            stockfish_bot_factory,
            num_games=10,
        )

        log('Playing game...')
        results = tournament_manager.play_games()
        log('Results:', results)
