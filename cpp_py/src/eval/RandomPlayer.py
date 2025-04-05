import random
import chess

from src.eval.Bot import Bot


class RandomPlayer(Bot):
    def __init__(self) -> None:
        """Initializes the random player."""
        super().__init__('RandomPlayer', max_time_to_think=0.0)

    def think(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))
