from typing import List, Optional, Tuple

from src.eval.HexGUI import HexGridGameGUI
from src.games.GameVisuals import GameVisuals
from src.games.hex.HexBoard import HexBoard, SIZE
from src.games.Game import Board


class HexVisuals(GameVisuals[int]):
    def draw_pieces(self, board: HexBoard, gui: HexGridGameGUI) -> None:
        # Draw X for player 1 and O for player -1
        for r in range(SIZE):
            for c in range(SIZE):
                cell = board.board[r, c]
                if cell == 1:
                    gui.draw_hex_cell(r, c, 'blue')
                elif cell == -1:
                    gui.draw_hex_cell(r, c, 'green')

    def is_two_click_game(self) -> bool:
        # Single click selects destination only
        return False

    def get_moves_from_square(self, board: Board[int], row: int, col: int) -> List[Tuple[int, int]]:
        # Hex is single-click game, not two-click
        assert False, 'Hex is not a two-click game'

    def try_make_move(
        self, board: Board[int], from_cell: Optional[Tuple[int, int]], to_cell: Tuple[int, int]
    ) -> Optional[int]:
        # from_cell ignored
        r, c = to_cell
        idx = r * SIZE + c
        if idx in board.get_valid_moves():
            return idx
        return None
