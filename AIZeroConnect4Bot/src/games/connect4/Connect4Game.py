from __future__ import annotations

import torch
import numpy as np
from typing import List

from src.games.Game import Game
from src.games.connect4.Connect4Defines import (
    ROW_COUNT,
    COLUMN_COUNT,
    ACTION_SIZE,
    ENCODING_CHANNELS,
    NUM_RES_BLOCKS,
    NUM_HIDDEN,
    AVERAGE_NUM_MOVES_PER_GAME,
)
from src.games.connect4.Connect4Board import Connect4Board, Connect4Move
from src.util.ZobristHasher import ZobristHasher


class Connect4Game(Game[Connect4Move]):
    Hasher = ZobristHasher(ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT)

    @property
    def null_move(self) -> Connect4Move:
        return -1

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        return ENCODING_CHANNELS, ROW_COUNT, COLUMN_COUNT

    @property
    def network_properties(self) -> tuple[int, int]:
        return NUM_RES_BLOCKS, NUM_HIDDEN

    @property
    def average_num_moves_per_game(self) -> int:
        return AVERAGE_NUM_MOVES_PER_GAME

    def get_canonical_board(self, board: Connect4Board) -> np.ndarray:
        # TODO 3 channels for player 1, player 2 and empty cells?
        return (board.board * board.current_player).reshape(self.representation_shape)

    def hash_boards(self, boards: torch.Tensor) -> List[int]:
        assert boards.shape[1:] == self.representation_shape, f'Invalid shape: {boards.shape}'
        return self.Hasher.zobrist_hash_boards(boards)

    def encode_move(self, move: Connect4Move) -> int:
        assert 0 <= move < COLUMN_COUNT, f'Invalid move: {move}'
        return move

    def decode_move(self, move: int) -> Connect4Move:
        assert 0 <= move < COLUMN_COUNT, f'Invalid move: {move}'
        return move

    def symmetric_variations(
        self, board: np.ndarray, action_probabilities: np.ndarray
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        return [
            # Original board
            (board, action_probabilities),
            # Vertical flip
            # 1234 -> becomes -> 4321
            # 5678               8765
            (np.flip(board, axis=2), np.flip(action_probabilities)),
            # NOTE: The following implementations DO NOT WORK. They are incorrect. This would give wrong symmetries to train on.
            # Player flip
            # yield -board, action_probabilities, -result
            # Player flip and vertical flip
            # yield -board[:, ::-1], action_probabilities[::-1], -result
        ]

    def get_initial_board(self) -> Connect4Board:
        return Connect4Board()
