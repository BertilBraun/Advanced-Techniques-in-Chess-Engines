from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Literal, Optional, TypeVar

import numpy as np
import torch

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


class Game(ABC, Generic[_Move]):
    @property
    @abstractmethod
    def null_move(self) -> _Move:
        pass

    @property
    @abstractmethod
    def action_size(self) -> int:
        # The number of possible moves in the game.
        pass

    @property
    @abstractmethod
    def representation_shape(self) -> tuple[int, int, int]:
        # (num_channels, height, width)
        pass

    @property
    @abstractmethod
    def network_properties(self) -> tuple[int, int]:
        # (num_res_blocks, hidden_dim)
        pass

    @property
    @abstractmethod
    def average_num_moves_per_game(self) -> int:
        pass

    @abstractmethod
    def get_canonical_board(self, board: Board[_Move]) -> np.ndarray:
        # Returns a canonical representation of the board from the perspective of the current player.
        # No matter the current player, the board should always be from the perspective as if the player to move is 1.
        # The board should be a numpy array with shape (num_channels, height, width) as returned by the `representation_shape` property.
        pass

    @abstractmethod
    def encode_move(self, move: _Move) -> int:
        # Encodes a move into the index of the action in the policy vector.
        pass

    @abstractmethod
    def decode_move(self, move: int) -> _Move:
        # Decodes an action index into a move.
        pass

    def encode_moves(self, moves: list[_Move]) -> torch.Tensor:
        encoded = torch.zeros(self.action_size, dtype=torch.float)
        for move in moves:
            encoded[self.encode_move(move)] = 1
        return encoded

    def decode_moves(self, moves: np.ndarray) -> list[_Move]:
        return [self.decode_move(i) for i in moves.nonzero()[0]]

    @abstractmethod
    def hash_boards(self, boards: torch.Tensor) -> list[int]:
        # Hashes a batch of encoded canonical boards.
        # The input tensor has shape (batch_size, num_channels, height, width) also (batch_size, *representation_shape).
        pass

    @abstractmethod
    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # Returns a list of symetric variations of the board and the corresponding action probabilities.
        # The board is a numpy array with shape (num_channels, height, width) as returned by the `representation_shape` property.
        # The action_probabilities is a numpy array with shape (action_size).
        pass

    @abstractmethod
    def get_initial_board(self) -> Board[_Move]:
        pass
