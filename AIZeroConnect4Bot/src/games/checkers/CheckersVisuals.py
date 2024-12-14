from typing import List, Optional, Tuple

from AIZeroConnect4Bot.src.games.Game import Board
from AIZeroConnect4Bot.src.games.GUI import BaseGridGameGUI
from AIZeroConnect4Bot.src.games.GameVisuals import GameVisuals
from AIZeroConnect4Bot.src.games.checkers.CheckersBoard import CheckersBoard, CheckersMove


class CheckersVisuals(GameVisuals[CheckersMove]):
    def draw_pieces(self, board: CheckersBoard, gui: BaseGridGameGUI) -> None:
        board_width, board_height = board.board_dimensions
        for row in range(board_height):
            for col in range(board_width):
                cell_index = row * board_width + col
                if board.black_kings & (1 << cell_index):
                    gui.draw_circle(row, col, 'black')
                elif board.black_pieces & (1 << cell_index):
                    gui.draw_circle(row, col, 'dark gray')
                elif board.white_kings & (1 << cell_index):
                    gui.draw_circle(row, col, 'white')
                elif board.white_pieces & (1 << cell_index):
                    gui.draw_circle(row, col, 'light gray')

    def is_two_click_game(self) -> bool:
        return True

    def get_moves_from_square(self, board: Board[CheckersMove], row: int, col: int) -> List[Tuple[int, int]]:
        valid_moves = board.get_valid_moves()
        board_width, board_height = board.board_dimensions
        from_index = row * board_width + col

        valid_moves_from_square = []
        for move_from, move_to in valid_moves:
            if move_from == from_index:
                valid_moves_from_square.append(move_to)

        return [(move // board_width, move % board_width) for move in valid_moves_from_square]

    def try_make_move(
        self,
        board: Board[CheckersMove],
        from_cell: Optional[Tuple[int, int]],
        to_cell: Tuple[int, int],
    ) -> Optional[CheckersMove]:
        assert from_cell is not None, 'from_cell should not be None'

        from_row, from_col = from_cell
        to_row, to_col = to_cell
        board_width, board_height = board.board_dimensions

        from_index = from_row * board_width + from_col
        to_index = to_row * board_width + to_col

        valid_moves = board.get_valid_moves()
        for move_from, move_to in valid_moves:
            if move_from == from_index and move_to == to_index:
                return (from_index, to_index)

        return None
