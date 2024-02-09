from dataclasses import dataclass

from chess import KING, Board, Move, PieceType


@dataclass
class ExtendedMove:
    move: Move = Move.null()
    move_piece_type: PieceType = 0
    capture_piece_type: PieceType | None = None
    score: float = 0

    def __init__(self, move: Move, board: Board, score: float) -> None:
        self.move = move
        self.move_piece_type = board.piece_type_at(move.from_square) or 0
        self.capture_piece_type = board.piece_type_at(move.to_square)
        self.score = score

    @property
    def from_square(self) -> int:
        return self.move.from_square

    @property
    def to_square(self) -> int:
        return self.move.to_square

    @property
    def is_capture(self) -> bool:
        return self.capture_piece_type != 0

    @property
    def is_castling(self) -> bool:
        return self.move_piece_type == KING and abs(self.from_square - self.to_square) == 2

    @property
    def is_promotion(self) -> bool:
        return self.move.promotion != 0

    @property
    def is_null(self) -> bool:
        return self.move == Move.null()

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, ExtendedMove):
            return (
                self.move == __value.move
                and self.move_piece_type == __value.move_piece_type
                and self.capture_piece_type == __value.capture_piece_type
            )
        if isinstance(__value, Move):
            return self.move == __value
        return False

    @classmethod
    def null(cls) -> 'ExtendedMove':
        return cls(Move.null(), Board(), 0)


def get_legal_moves(board: Board) -> list[ExtendedMove]:
    # Return a list of ExtendedMove objects for all legal moves in the current board state. The score is set to 0.
    return [ExtendedMove(move, board, 0) for move in board.legal_moves]
