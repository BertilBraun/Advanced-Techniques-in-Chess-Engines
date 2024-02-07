import chess

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class GameResult:
    winner: Optional[chess.Color]
    result: Literal["1-0", "0-1", "1/2-1/2", "unfinished"]

    @staticmethod
    def from_board(board: chess.Board) -> 'GameResult':
        if board.is_checkmate():
            result = "1-0" if board.turn == chess.BLACK else "0-1"
            return GameResult(board.turn, result)
        
        if board.is_game_over():
            return GameResult(None, "1/2-1/2")

        return GameResult(None, "unfinished")
