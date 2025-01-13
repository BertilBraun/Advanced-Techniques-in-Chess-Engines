from __future__ import annotations

import chess
from typing import Optional, List

from src.games.Board import Board, Player

ChessMove = chess.Move


class ChessBoard(Board[ChessMove]):
    def __init__(self) -> None:
        super().__init__()
        self.board = chess.Board()

    def make_move(self, move: ChessMove) -> None:
        assert move in self.board.legal_moves
        self.board.push(move)
        self._switch_player()

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def check_winner(self) -> Optional[Player]:
        result = self.board.result()
        if result == '1-0':
            return 1
        elif result == '0-1':
            return -1
        else:
            return None

    def get_valid_moves(self) -> List[ChessMove]:
        return list(self.board.legal_moves)

    def copy(self) -> ChessBoard:
        game = ChessBoard()
        game.board = self.board.copy(stack=False)
        game.current_player = self.current_player
        return game

    def quick_hash(self) -> int:
        return hash(self.board.fen())
