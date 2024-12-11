import torch

from Framework.GameManager import GameManager
from Framework.ExampleBots.HumanPlayer import HumanPlayer
from AIZeroConnect4Bot.eval.AlphaZeroBot import AlphaZeroBot

if __name__ == '__main__':
    last_training_config = torch.load('last_training_config.pt')
    model_path = last_training_config['model']

    game_manager = GameManager(HumanPlayer(), AlphaZeroBot(model_path))

    result, thinking_time = game_manager.play_game()

    print(result)
