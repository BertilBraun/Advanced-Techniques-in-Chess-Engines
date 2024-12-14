from AIZeroConnect4Bot.src.util import load_json
from AIZeroConnect4Bot.src.util.log import log
from AIZeroConnect4Bot.src.settings import SAVE_PATH
from AIZeroConnect4Bot.src.games.GUI import BaseGridGameGUI
from AIZeroConnect4Bot.src.eval.GameManager import GameManager
from AIZeroConnect4Bot.src.eval.HumanPlayer import HumanPlayer
from AIZeroConnect4Bot.src.eval.AlphaZeroBot import AlphaZeroBot


class Connect4HumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        from AIZeroConnect4Bot.src.games.connect4.Connect4Board import Connect4Board
        from AIZeroConnect4Bot.src.games.connect4.Connect4Visuals import Connect4Visuals

        rows, cols = Connect4Board().board_dimensions
        gui = BaseGridGameGUI(rows, cols, checkered=False)
        visuals = Connect4Visuals()
        super().__init__(gui, visuals)


class CheckersHumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        from AIZeroConnect4Bot.src.games.checkers.CheckersVisuals import CheckersVisuals

        gui = BaseGridGameGUI(8, 8)
        visuals = CheckersVisuals()
        super().__init__(gui, visuals)


if __name__ == '__main__':
    last_training_config = load_json(f'{SAVE_PATH}/last_training_config.json')
    model_path = last_training_config['model']

    game_manager = GameManager(
        Connect4HumanPlayer(),
        AlphaZeroBot(model_path, max_time_to_think=5.0),
    )

    log('Final Result - Winner:', game_manager.play_game())
