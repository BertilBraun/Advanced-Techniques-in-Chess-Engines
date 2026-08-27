from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

BOARD_LENGTH = 8
BOARD_SQUARE_COUNT = BOARD_LENGTH * BOARD_LENGTH
PROMOTION_SOURCE_ROW = 6
PROMOTION_PIECE_COUNT = 4
KNIGHT_PROMOTION_INDEX = 3

# Row and column steps, in the order cpp/src/games/chess/encoding/ChessPolicyEncoding.cpp enumerates them;
# the action ids depend on the order, so these tuples are a contract with the native encoding.
RAY_DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
KNIGHT_STEPS = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
PROMOTION_PIECES = ('q', 'r', 'b', 'n')


@dataclass(frozen=True)
class ChessFromToActionTable:
    """The (from, to, promotion) triple behind every reduced-encoding chess action id."""

    from_squares: npt.NDArray[np.int64]
    to_squares: npt.NDArray[np.int64]
    promotion_indices: npt.NDArray[np.int64]

    @property
    def action_count(self) -> int:
        return len(self.from_squares)

    @property
    def from_to_indices(self) -> npt.NDArray[np.int64]:
        return self.from_squares * BOARD_SQUARE_COUNT + self.to_squares

    @property
    def promotion_action_ids(self) -> npt.NDArray[np.int64]:
        return np.flatnonzero(self.promotion_indices >= 0).astype(np.int64)

    @property
    def promotion_offset_indices(self) -> npt.NDArray[np.int64]:
        promotions = self.promotion_action_ids
        destination_columns = self.to_squares[promotions] % BOARD_LENGTH
        return destination_columns * PROMOTION_PIECE_COUNT + self.promotion_indices[promotions]


def _square(column: int, row: int) -> int:
    return row * BOARD_LENGTH + column


def build_chess_from_to_action_table() -> ChessFromToActionTable:
    from_squares: list[int] = []
    to_squares: list[int] = []
    promotion_indices: list[int] = []

    def add_move(from_square: int, to_square: int, promotion_index: int) -> None:
        from_squares.append(from_square)
        to_squares.append(to_square)
        promotion_indices.append(promotion_index)

    for from_square in range(BOARD_SQUARE_COUNT):
        row, column = divmod(from_square, BOARD_LENGTH)
        for row_step, column_step in RAY_DIRECTIONS:
            for distance in range(1, BOARD_LENGTH):
                to_row = row + row_step * distance
                to_column = column + column_step * distance
                if 0 <= to_row < BOARD_LENGTH and 0 <= to_column < BOARD_LENGTH:
                    add_move(from_square, _square(to_column, to_row), -1)
        for row_step, column_step in KNIGHT_STEPS:
            to_row = row + row_step
            to_column = column + column_step
            if 0 <= to_row < BOARD_LENGTH and 0 <= to_column < BOARD_LENGTH:
                add_move(from_square, _square(to_column, to_row), -1)
        if row == PROMOTION_SOURCE_ROW:
            for column_offset in (-1, 0, 1):
                to_column = column + column_offset
                if 0 <= to_column < BOARD_LENGTH:
                    to_square = _square(to_column, PROMOTION_SOURCE_ROW + 1)
                    for promotion_index in range(PROMOTION_PIECE_COUNT):
                        add_move(from_square, to_square, promotion_index)

    return ChessFromToActionTable(
        from_squares=np.array(from_squares, dtype=np.int64),
        to_squares=np.array(to_squares, dtype=np.int64),
        promotion_indices=np.array(promotion_indices, dtype=np.int64),
    )


CHESS_FROM_TO_ACTION_TABLE = build_chess_from_to_action_table()
