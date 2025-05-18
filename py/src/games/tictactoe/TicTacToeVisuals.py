from typing import List, Optional, Tuple

from src.games.Game import Board
from src.eval.GridGUI import BaseGridGameGUI
from src.games.GameVisuals import GameVisuals
from src.games.tictactoe.TicTacToeBoard import TicTacToeBoard, ROW_COUNT
from src.games.tictactoe.TicTacToeGame import TicTacToeMove


class TicTacToeVisuals(GameVisuals[TicTacToeMove]):
    def draw_pieces(self, board: TicTacToeBoard, gui: BaseGridGameGUI) -> None:
        for i, cell in enumerate(board.board):
            x, y = divmod(i, ROW_COUNT)
            if cell == 1:
                gui.draw_circle(x, y, 'red')
            elif cell == -1:
                gui.draw_circle(x, y, 'yellow')

    def is_two_click_game(self) -> bool:
        return False

    def get_moves_from_square(self, board: Board[TicTacToeMove], row: int, col: int) -> List[Tuple[int, int]]:
        assert False, 'TicTacToe is not a two-click game'

    def try_make_move(
        self,
        board: Board[TicTacToeMove],
        from_cell: Optional[Tuple[int, int]],
        to_cell: Tuple[int, int],
    ) -> Optional[TicTacToeMove]:
        assert from_cell is None, 'TicTacToe is not a two-click game'

        move = to_cell[0] * ROW_COUNT + to_cell[1]
        if move in board.get_valid_moves():
            return move
        return None
