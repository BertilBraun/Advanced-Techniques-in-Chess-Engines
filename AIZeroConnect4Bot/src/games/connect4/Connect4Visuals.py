from typing import List, Optional, Tuple

from src.games.Game import Board
from src.games.GUI import BaseGridGameGUI
from src.games.GameVisuals import GameVisuals
from src.games.connect4.Connect4Board import Connect4Board
from src.games.connect4.Connect4Game import Connect4Move


class Connect4Visuals(GameVisuals[Connect4Move]):
    def draw_pieces(self, board: Connect4Board, gui: BaseGridGameGUI) -> None:
        for i, row in enumerate(board.board):
            for j, cell in enumerate(row):
                if cell == 1:
                    gui.draw_circle(i, j, 'red')
                elif cell == -1:
                    gui.draw_circle(i, j, 'yellow')

    def is_two_click_game(self) -> bool:
        return False

    def get_moves_from_square(self, board: Board[Connect4Move], row: int, col: int) -> List[Tuple[int, int]]:
        assert False, 'Connect4 is not a two-click game'

    def try_make_move(
        self,
        board: Board[Connect4Move],
        from_cell: Optional[Tuple[int, int]],
        to_cell: Tuple[int, int],
    ) -> Optional[Connect4Move]:
        assert from_cell is None, 'Connect4 is not a two-click game'

        move = Connect4Move(to_cell[1])
        if move in board.get_valid_moves():
            return move
        return None
