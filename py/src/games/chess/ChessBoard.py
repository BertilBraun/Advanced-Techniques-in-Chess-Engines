from __future__ import annotations

import chess
from src.games.Board import Board, Player
from src.games.chess.repetition_history import REPETITION_HISTORY_PLIES

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
        assert move in self.get_valid_moves(), (
            f'Invalid move: {move} for board FEN: {self.board.fen()}:\n{self}\nValid moves: {self.get_valid_moves()}'
        )
        self.board.push(move)

    def is_game_over(self) -> bool:
        if self.board.is_game_over(claim_draw=False):
            return True
        return self.board.halfmove_clock >= 100 or self.board.is_repetition(3)

    def check_winner(self) -> Player | None:
        """Check if the game is over and return the winner."""
        outcome = self.board.outcome()
        if outcome is None or outcome.winner is None:
            return None
        return 1 if outcome.winner == chess.WHITE else -1

    def get_valid_moves(self) -> list[ChessMove]:
        return list(self.board.legal_moves)

    def copy(self) -> ChessBoard:
        game = ChessBoard()
        game.board = self.board.copy(stack=REPETITION_HISTORY_PLIES)
        return game

    def quick_hash(self) -> int:
        return hash(self.board.fen())

    def get_approximate_result_score(self) -> float:
        """
        Normalised score in [-1, 1].
        1   → White is completely winning (all the material belongs to White)
        0   → Material is exactly equal
        -1  → Black is completely winning
        """
        white_total = 0
        black_total = 0

        for piece in chess.PIECE_TYPES:
            value = PIECE_VALUE[piece]
            white_total += value * len(self.board.pieces(piece, chess.WHITE))
            black_total += value * len(self.board.pieces(piece, chess.BLACK))

        total_material = white_total + black_total  # what’s actually left
        if total_material == 0:  # bare kings
            return 0.0

        return (white_total - black_total) / total_material

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
