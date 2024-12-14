from __future__ import annotations

import torch
import numpy as np
from typing import List

from AIZeroConnect4Bot.src.games.Game import Board, Game
from AIZeroConnect4Bot.src.games.checkers.hashing import zobrist_hash_boards
from AIZeroConnect4Bot.src.games.checkers.CheckersBoard import CheckersBoard, CheckersMove


class CheckersGame(Game[CheckersMove]):
    @property
    def null_move(self) -> CheckersMove:
        return -1, -1

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def representation_shape(self) -> tuple[int, int, int]:
        ENCODING_CHANNELS = 4
        return ENCODING_CHANNELS, 8, 8

    @property
    def network_properties(self) -> tuple[int, int]:
        NUM_RES_BLOCKS = 10
        NUM_HIDDEN = 128
        return NUM_RES_BLOCKS, NUM_HIDDEN

    @property
    def average_num_moves_per_game(self) -> int:
        return 30

    def get_canonical_board(self, board: CheckersBoard) -> np.ndarray:
        return (board.board * board.current_player).reshape(self.representation_shape)

    def hash_boards(self, boards: torch.Tensor) -> List[int]:
        assert boards.shape[1:] == self.representation_shape, f'Invalid shape: {boards.shape}'
        return zobrist_hash_boards(boards)

    def encode_move(self, move: CheckersMove) -> int:
        assert 0 <= move < COLUMN_COUNT, f'Invalid move: {move}'
        return move

    def decode_move(self, move: int) -> CheckersMove:
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
            # TODO possibly doable? If also the action probabilities are flipped?
            # Player flip
            # yield -board, action_probabilities, -result
            # Player flip and vertical flip
            # yield -board[:, ::-1], action_probabilities[::-1], -result
        ]

    def get_initial_board(self) -> Board[CheckersMove]:
        return CheckersBoard()
