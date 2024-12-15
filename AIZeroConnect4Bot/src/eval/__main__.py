from AIZeroConnect4Bot.src.games.checkers.CheckersGame import CheckersGame
from AIZeroConnect4Bot.src.games.connect4.Connect4Game import Connect4Game
from AIZeroConnect4Bot.src.util import load_json
from AIZeroConnect4Bot.src.util.log import log
from AIZeroConnect4Bot.src.settings import CURRENT_GAME, SAVE_PATH
from AIZeroConnect4Bot.src.games.GUI import BaseGridGameGUI
from AIZeroConnect4Bot.src.eval.GameManager import GameManager
from AIZeroConnect4Bot.src.eval.HumanPlayer import HumanPlayer
from AIZeroConnect4Bot.src.eval.AlphaZeroBot import AlphaZeroBot


class Connect4HumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        from AIZeroConnect4Bot.src.games.connect4.Connect4Visuals import Connect4Visuals

        _, rows, cols = Connect4Game().representation_shape
        gui = BaseGridGameGUI(rows, cols, checkered=False)
        visuals = Connect4Visuals()
        super().__init__(gui, visuals)


class CheckersHumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        from AIZeroConnect4Bot.src.games.checkers.CheckersVisuals import CheckersVisuals

        _, rows, cols = CheckersGame().representation_shape
        gui = BaseGridGameGUI(rows, cols)
        visuals = CheckersVisuals()
        super().__init__(gui, visuals)


if __name__ == '__main__':
    try:
        last_training_config = load_json(f'{SAVE_PATH}/last_training_config.json')
        model_path = last_training_config['model']
    except FileNotFoundError:
        model_path = None

    MAX_TIME_TO_THINK = 1.0

    if isinstance(CURRENT_GAME, Connect4Game):
        game_manager = GameManager(
            Connect4HumanPlayer(),
            AlphaZeroBot(model_path, max_time_to_think=MAX_TIME_TO_THINK),
        )
    elif isinstance(CURRENT_GAME, CheckersGame):
        game_manager = GameManager(
            CheckersHumanPlayer(),
            AlphaZeroBot(model_path, max_time_to_think=MAX_TIME_TO_THINK),
        )

    log('Final Result - Winner:', game_manager.play_game())
