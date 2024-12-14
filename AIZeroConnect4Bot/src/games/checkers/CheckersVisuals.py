from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Tuple, TypeVar

from AIZeroConnect4Bot.src.games.Game import Board
from AIZeroConnect4Bot.src.games.GUI import BaseGridGameGUI
from AIZeroConnect4Bot.src.games.GameVisuals import GameVisuals
from AIZeroConnect4Bot.src.games.checkers.CheckersGame import CheckersMove


class CheckersVisuals(GameVisuals[CheckersMove]):
    def draw_pieces(self, board: Board[CheckersMove], gui: BaseGridGameGUI) -> None:
        pass

    def is_two_click_game(self) -> bool:
        pass

    def get_moves_from_square(self, row: int, col: int) -> List[Tuple[int, int]]:
        pass

    def try_make_move(
        self,
        board: Board[CheckersMove],
        from_cell: Optional[Tuple[int, int]],
        to_cell: Tuple[int, int],
    ) -> Optional[CheckersMove]:
        pass
