from src.util import load_json
from src.util.log import log
from src.settings import CurrentGame, CurrentGameVisuals, SAVE_PATH
from src.games.GUI import BaseGridGameGUI
from src.eval.GameManager import GameManager
from src.eval.HumanPlayer import HumanPlayer
from src.eval.AlphaZeroBot import AlphaZeroBot


class CommonHumanPlayer(HumanPlayer):
    def __init__(self) -> None:
        """Initializes the human player."""
        _, rows, cols = CurrentGame.representation_shape
        gui = BaseGridGameGUI(rows, cols)
        super().__init__(gui, CurrentGameVisuals)


if __name__ == '__main__':
    try:
        last_training_config = load_json(f'{SAVE_PATH}/last_training_config.json')
        model_path = last_training_config['model']
    except FileNotFoundError:
        model_path = None

    MAX_TIME_TO_THINK = 1.0

    game_manager = GameManager(
        AlphaZeroBot(model_path, max_time_to_think=MAX_TIME_TO_THINK),
        CommonHumanPlayer(),
    )

    log('Final Result - Winner:', game_manager.play_game())
