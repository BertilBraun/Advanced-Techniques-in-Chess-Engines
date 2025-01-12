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

    @classmethod
    def _cache(cls):
        def cache_decorator(method):
            def cached_method(self, *args, **kwargs):
                cache_name = f'_{method.__name__}_cached_result'
                if not hasattr(self, 'cache_attr_names'):
                    setattr(self, 'cache_attr_names', [])
                self.cache_attr_names.append(cache_name)
                if not hasattr(self, cache_name) or getattr(self, cache_name) is None:
                    setattr(self, cache_name, method(self, *args, **kwargs))

                return getattr(self, cache_name)

            return cached_method

        return cache_decorator

    def _invalidate_cache(self) -> None:
        if not hasattr(self, 'cache_attr_names'):
            return
        for attr in getattr(self, 'cache_attr_names'):
            setattr(self, attr, None)

    def _copy_cache(self, other: Board) -> None:
        if not hasattr(other, 'cache_attr_names'):
            return
        for attr in getattr(other, 'cache_attr_names'):
            setattr(self, attr, getattr(other, attr))
