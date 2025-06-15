import chess.engine

from chess import *  # type: ignore
from src.eval.Bot import Bot

from src.games.chess.ChessBoard import ChessBoard
from src.games.chess.comparison_bots.util import *


class ChessStockfishBot(Bot):
    def __init__(self, skill_level: int = 4, max_time_to_think: float = 0.2) -> None:
        """Initializes the Stockfish player."""
        self.engine = chess.engine.SimpleEngine.popen_uci('stockfish')
        super().__init__('Stockfish', max_time_to_think=max_time_to_think)
        self.engine.configure({'Skill Level': skill_level})

        self.limit = chess.engine.Limit(time=self.max_time_to_think)

    def cleanup(self) -> None:
        """Cleans up the Stockfish player."""
        self.engine.quit()

    def think(self, board: ChessBoard) -> Move:
        """Selects a move based on stockfish's evaluation."""

        result = self.engine.play(board.board, self.limit)
        move = result.move

        if not move:
            raise ValueError('Stockfish did not return a move.')

        if move.promotion is not None:
            # Handle promotion moves
            move = Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        return move
