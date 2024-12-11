import json
import torch

from abc import ABC, abstractmethod

from AIZeroConnect4Bot.src.Board import Board, Move


class Bot(ABC):
    def __init__(self, name: str) -> None:
        """Initializes the bot with a name."""
        self.name = name

    @abstractmethod
    def think(self, board: Board) -> Move:
        """This method is called when it's the bot's turn to move. It should return the move that the bot wants to make."""
        raise NotImplementedError('Subclasses must implement this method')


class GameManager:
    def __init__(self, white: Bot, black: Bot) -> None:
        """Initializes the game manager with two players."""
        self.board = Board()
        self.players = {1: white, -1: black}

    def play_game(self, verify_moves=True):
        """Manages the gameplay loop until the game is over or a player quits."""
        while not self.board.is_game_over():
            print(self.board.get_board_state())
            move = self.players[self.board.current_player].think(self.board)

            if verify_moves and move not in self.board.get_valid_moves():
                raise ValueError(f'Invalid move {move} for player {self.players[self.board.current_player].name}')

            self.board.make_move(move)
            self.board.switch_player()

        return self.board.check_winner()


class HumanPlayer(Bot):
    def __init__(self) -> None:
        """Initializes the human player."""
        super().__init__('HumanPlayer')

    def think(self, board: Board) -> Move:
        """Prompts the human player to make a move."""
        print(f'Enter the column number to make a move {board.get_valid_moves()}:')

        while True:
            try:
                move = int(input())
                if move in board.get_valid_moves():
                    return move
                else:
                    print('Invalid move. Please try again.')
            except ValueError:
                print('Invalid input. Please enter a number.')


if __name__ == '__main__':
    from AIZeroConnect4Bot.eval.AlphaZeroBot import AlphaZeroBot

    with open('last_training_config.json', 'r') as f:
        last_training_config = json.load(f)
    model_path = last_training_config['model']

    game_manager = GameManager(HumanPlayer(), AlphaZeroBot(model_path, num_iterations=1000))

    print('Final Result - Winner:', game_manager.play_game())
