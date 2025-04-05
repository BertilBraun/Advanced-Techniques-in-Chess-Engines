from __future__ import annotations

import chess
from typing import Literal, Optional, List

ChessMove = chess.Move
Player = Literal[-1, 1]



class ChessBoard:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.current_player = 1

    def make_move(self, move: ChessMove) -> None:
        assert move in self.board.legal_moves
        self.board.push(move)
        self.current_player *= -1

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

    def set_fen(self, fen: str) -> None:
        self.board.set_fen(fen)
        self.current_player = 1 if self.board.turn == chess.WHITE else -1

    @staticmethod
    def from_fen(fen: str) -> ChessBoard:
        board = ChessBoard()
        board.set_fen(fen)
        return board

    def __repr__(self) -> str:
        from src.settings import BOARD_LENGTH

        rows = []
        rows.append('  a b c d e f g h')
        for i in range(BOARD_LENGTH):
            row = [str(i + 1)]
            for j in range(BOARD_LENGTH):
                piece = self.board.piece_at(i * BOARD_LENGTH + j)
                row.append(piece.unicode_symbol(invert_color=True) if piece is not None else '.')
            row.append(f' {i + 1}')
            rows.append(' '.join(row))
        rows.append('  a b c d e f g h')
        return '\n'.join(rows)
