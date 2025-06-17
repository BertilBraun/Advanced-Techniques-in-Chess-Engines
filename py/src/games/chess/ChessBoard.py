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
        from src.games.chess.ChessGame import mirror_move_for_black

        # if castling move, map it back to a chess Move from a stockfish move
        piece_from = self.board.piece_at(move.from_square)
        piece_to = self.board.piece_at(move.to_square)
        if (
            move.promotion is None
            and piece_from
            and piece_to
            and piece_from.piece_type == chess.KING
            and piece_to.piece_type == chess.ROOK
            and piece_from.color == piece_to.color
        ):
            # Castling move is a special case, we need to handle it differently
            if move.from_square == chess.E1 and move.to_square == chess.A1:
                move = chess.Move(chess.E1, chess.C1)
            elif move.from_square == chess.E1 and move.to_square == chess.H1:
                move = chess.Move(chess.E1, chess.G1)
            elif move.from_square == chess.E8 and move.to_square == chess.A8:
                move = chess.Move(chess.E8, chess.C8)
            elif move.from_square == chess.E8 and move.to_square == chess.H8:
                move = chess.Move(chess.E8, chess.G8)

        assert move in self.get_valid_moves(), f'Invalid move: {move} for board:\n{self}'
        self.board.push(move)

    def is_game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True) or self._is_draw_by_insufficient_material()

    def _is_draw_by_insufficient_material(self) -> bool:
        """Check if the game is a draw by insufficient material. Copy to the one in Board.cpp"""
        if len(self.board.piece_map()) > 4:
            return False

        # Check for the following cases:
        # 1) King vs King
        if len(self.board.piece_map()) == 2:
            return True
        # 2) King and Bishop vs King
        if len(self.board.piece_map()) == 3 and len(self.board.piece_map(mask=self.board.bishops)) == 1:
            return True
        # 3) King and Knight vs King
        if len(self.board.piece_map()) == 3 and len(self.board.piece_map(mask=self.board.knights)) == 1:
            return True
        # 4) King and two Knights vs King
        if len(self.board.piece_map()) == 4 and len(self.board.piece_map(mask=self.board.knights)) == 2:
            return True
        return False

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
