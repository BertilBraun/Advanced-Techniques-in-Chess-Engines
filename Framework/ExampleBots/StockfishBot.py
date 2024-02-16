import chess.engine

from Framework.ChessBot import ChessBot, Board, Move, TIME_TO_THINK


class StockfishBot(ChessBot):
    def __init__(self) -> None:
        """Initializes the Stockfish player."""
        super().__init__('Stockfish')
        self.engine = chess.engine.SimpleEngine.popen_uci('stockfish')
        self.limit = chess.engine.Limit(time=TIME_TO_THINK)

    def cleanup(self) -> None:
        """Cleans up the Stockfish player."""
        self.engine.quit()

    def think(self, board: Board) -> Move:
        """Selects a move based on stockfish's evaluation."""

        result = self.engine.play(board, self.limit)
        if not result.move:
            raise ValueError('Stockfish did not return a move.')

        return result.move
