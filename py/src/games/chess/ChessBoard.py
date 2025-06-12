from __future__ import annotations

import chess
from typing import Optional, List

from src.games.Board import Board, Player

ChessMove = chess.Move

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
MAX_MATERIAL_VALUE = (
    PIECE_VALUE[chess.PAWN] * 8
    + PIECE_VALUE[chess.KNIGHT] * 2
    + PIECE_VALUE[chess.BISHOP] * 2
    + PIECE_VALUE[chess.ROOK] * 2
    + PIECE_VALUE[chess.QUEEN] * 1
)


class ChessBoard(Board[ChessMove]):
    def __init__(self) -> None:
        self.board = chess.Board()

    @property
    def current_player(self) -> Player:  # type: ignore
        return 1 if self.board.turn == chess.WHITE else -1

    def make_move(self, move: ChessMove) -> None:
        # TODO? assert move in self.board.legal_moves
        self.board.push(move)

    def is_game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def check_winner(self) -> Optional[Player]:
        """Check if the game is over and return the winner."""
        result = self.board.result(claim_draw=True)
        if result == '1-0':
            return 1
        elif result == '0-1':
            return -1
        else:
            return None

    def get_valid_moves(self) -> List[ChessMove]:
        legal_moves = list(self.board.legal_moves)
        # Filter out non-queen promotions
        return [move for move in legal_moves if not move.promotion or move.promotion == chess.QUEEN]

    def copy(self) -> ChessBoard:
        game = ChessBoard()
        game.board = self.board.copy(stack=False)
        return game

    def quick_hash(self) -> int:
        return hash(self.board.fen())

    def get_approximate_result_score(self) -> float:
        """Returns a score between -1 and 1, where 1 means white is winning, -1 means black is winning, and 0 is a draw."""
        mat_value = 0

        for piece in chess.PIECE_TYPES:
            white_value = len(self.board.pieces(piece, chess.WHITE)) * PIECE_VALUE[piece]
            black_value = len(self.board.pieces(piece, chess.BLACK)) * PIECE_VALUE[piece]
            mat_value += white_value - black_value

        return mat_value / MAX_MATERIAL_VALUE

    def set_fen(self, fen: str) -> None:
        self.board.set_fen(fen)

    @staticmethod
    def from_fen(fen: str) -> ChessBoard:
        board = ChessBoard()
        board.set_fen(fen)
        return board

    def __repr__(self) -> str:
        from src.games.chess.ChessGame import BOARD_LENGTH

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
