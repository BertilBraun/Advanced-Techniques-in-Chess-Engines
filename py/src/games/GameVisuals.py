from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, List, Optional, Tuple, TypeVar

from src.games.Game import Board

if TYPE_CHECKING:
    from src.eval.GridGUI import BaseGridGameGUI

_Move = TypeVar('_Move')


class GameVisuals(ABC, Generic[_Move]):
    @abstractmethod
    def draw_pieces(self, board: Board[_Move], gui: BaseGridGameGUI) -> None:
        pass

    @abstractmethod
    def is_two_click_game(self) -> bool:
        pass

    @abstractmethod
    def get_moves_from_square(self, board: Board[_Move], row: int, col: int) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def try_make_move(
        self,
        board: Board[_Move],
        from_cell: Optional[Tuple[int, int]],
        to_cell: Tuple[int, int],
    ) -> Optional[_Move]:
        pass
