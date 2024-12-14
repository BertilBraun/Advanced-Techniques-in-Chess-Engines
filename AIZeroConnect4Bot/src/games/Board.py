from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Literal, Optional, TypeVar


_Move = TypeVar('_Move')
Player = Literal[-1, 1]


class Board(ABC, Generic[_Move]):
    def __init__(self) -> None:
        self.current_player: Player = 1

    @abstractmethod
    def make_move(self, move: _Move) -> None:
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        pass

    @abstractmethod
    def check_winner(self) -> Optional[Player]:
        pass

    @abstractmethod
    def get_valid_moves(self) -> list[_Move]:
        pass

    @abstractmethod
    def copy(self) -> Board[_Move]:
        pass

    def _switch_player(self) -> None:
        self.current_player = -self.current_player
