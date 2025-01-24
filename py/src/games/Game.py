from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from src.games.Board import Board, Player  # noqa: F401

_Move = TypeVar('_Move')


class Game(ABC, Generic[_Move]):
    @property
    @abstractmethod
    def null_move(self) -> _Move:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        """The number of possible moves in the game."""
        pass

    @property
    @abstractmethod
    def representation_shape(self) -> tuple[int, int, int]:
        """(num_channels, height, width)"""
        pass

    @abstractmethod
    def get_canonical_board(self, board: Board[_Move]) -> np.ndarray:
        """Returns a canonical representation of the board from the perspective of the current player.
        No matter the current player, the board should always be from the perspective as if the player to move is 1.
        The board should be a numpy array with shape (num_channels, height, width) as returned by the `representation_shape` property."""
        pass

    @abstractmethod
    def encode_move(self, move: _Move) -> int:
        """Encodes a move into the index of the action in the policy vector."""
        pass

    @abstractmethod
    def decode_move(self, move: int) -> _Move:
        """Decodes an action index into a move."""
        pass

    def encode_moves(self, moves: list[_Move]) -> np.ndarray:
        encoded = np.zeros(self.action_size)
        for move in moves:
            encoded[self.encode_move(move)] = 1
        return encoded

    def decode_moves(self, moves: np.ndarray) -> list[_Move]:
        return [self.decode_move(i) for i in moves]

    @abstractmethod
    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns a list of symetric variations of the board and the corresponding action probabilities.
        The board is a numpy array with shape (num_channels, height, width) as returned by the `representation_shape` property.
        The action_probabilities is a numpy array with shape (action_size)."""
        pass

    @abstractmethod
    def get_initial_board(self) -> Board[_Move]:
        pass
