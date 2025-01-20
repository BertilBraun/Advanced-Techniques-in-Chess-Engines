import chess.engine

from chess import *  # type: ignore
from src.eval.Bot import Bot

from py.src.games.chess.comparison_bots.util import *


class ChessStockfishBot(Bot):
    def __init__(self) -> None:
        """Initializes the Stockfish player."""
        super().__init__('Stockfish', max_time_to_think=0.2)  # 200ms of thinking time
        self.engine = chess.engine.SimpleEngine.popen_uci('stockfish')
        self.limit = chess.engine.Limit(time=self.max_time_to_think)

    def cleanup(self) -> None:
        """Cleans up the Stockfish player."""
        self.engine.quit()

    def think(self, board: Board) -> Move:
        """Selects a move based on stockfish's evaluation."""

        result = self.engine.play(board, self.limit)
        if not result.move:
            raise ValueError('Stockfish did not return a move.')

        return result.move
