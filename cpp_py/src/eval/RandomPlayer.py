import random

from src.eval.Bot import Bot
from src.settings import ChessMove, ChessBoard


class RandomPlayer(Bot):
    def __init__(self) -> None:
        """Initializes the random player."""
        super().__init__('RandomPlayer', max_time_to_think=0.0)

    def think(self, board: ChessBoard) -> ChessMove:
        return random.choice(board.get_valid_moves())
