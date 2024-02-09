from dataclasses import dataclass

from Framework import ChessBot, Color, Board, WHITE
from Framework.GameResult import GameResult


@dataclass
class ThinkingTime:
    white: float = 0.0
    black: float = 0.0

    def update(self, color: Color, time: float) -> None:
        if color == WHITE:
            self.white += time
        else:
            self.black += time


class GameManager:
    def __init__(self, white: ChessBot, black: ChessBot) -> None:
        """Initializes the game manager with two players."""
        self.board = Board()
        self.white = white
        self.black = black
        self.thinking_time = ThinkingTime()

    def play_game(self, verify_moves=True) -> tuple[GameResult, ThinkingTime]:
        """Manages the gameplay loop until the game is over or a player quits."""
        while not self.board.is_game_over():
            current_player = self.white if self.board.turn == WHITE else self.black

            current_player.restart_clock()
            move = current_player.think(self.board)

            if verify_moves and move not in self.board.legal_moves:
                raise ValueError(f'Invalid move {move} for player {current_player.name}')

            self.thinking_time.update(self.board.turn, current_player.time_elapsed)

            self.board.push(move)

        return GameResult.from_board(self.board), self.thinking_time
