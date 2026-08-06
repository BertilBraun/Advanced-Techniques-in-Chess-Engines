from __future__ import annotations

from typing import TYPE_CHECKING

from src.games.GameVisuals import GameVisuals
from src.games.go.NativeGoBoard import NativeGoBoard

if TYPE_CHECKING:
    from src.eval.GridGUI import BaseGridGameGUI


class GoVisuals(GameVisuals[int]):
    def draw_pieces(self, board: NativeGoBoard, gui: BaseGridGameGUI) -> None:
        for point in board.black_points():
            row, column = divmod(point, board.board_size)
            gui.draw_circle(row, column, 'black')
        for point in board.white_points():
            row, column = divmod(point, board.board_size)
            gui.draw_circle(row, column, 'white')

    def is_two_click_game(self) -> bool:
        return False

    def get_moves_from_square(self, board: NativeGoBoard, row: int, col: int) -> list[tuple[int, int]]:
        return []

    def try_make_move(
        self,
        board: NativeGoBoard,
        from_cell: tuple[int, int] | None,
        to_cell: tuple[int, int],
    ) -> int | None:
        row, column = to_cell
        action = row * board.board_size + column
        return action if action in board.get_valid_moves() else None
