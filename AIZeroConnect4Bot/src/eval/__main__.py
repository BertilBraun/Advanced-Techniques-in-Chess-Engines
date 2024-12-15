from AIZeroConnect4Bot.src.util import load_json
from AIZeroConnect4Bot.src.util.log import log
from AIZeroConnect4Bot.src.settings import CURRENT_GAME, CURRENT_GAME_VISUALS, SAVE_PATH
from AIZeroConnect4Bot.src.games.GUI import BaseGridGameGUI
from AIZeroConnect4Bot.src.eval.GameManager import GameManager
from AIZeroConnect4Bot.src.eval.HumanPlayer import HumanPlayer
from AIZeroConnect4Bot.src.eval.AlphaZeroBot import AlphaZeroBot


class CommonHumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        _, rows, cols = CURRENT_GAME.representation_shape
        gui = BaseGridGameGUI(rows, cols)
        super().__init__(gui, CURRENT_GAME_VISUALS)


if __name__ == '__main__':
    try:
        last_training_config = load_json(f'{SAVE_PATH}/last_training_config.json')
        model_path = last_training_config['model']
    except FileNotFoundError:
        model_path = None

    MAX_TIME_TO_THINK = 1.0

    game_manager = GameManager(
        CommonHumanPlayer(),
        AlphaZeroBot(model_path, max_time_to_think=MAX_TIME_TO_THINK),
    )

    log('Final Result - Winner:', game_manager.play_game())
